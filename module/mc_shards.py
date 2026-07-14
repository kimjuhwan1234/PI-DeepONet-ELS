# -*- coding: utf-8 -*-
"""전체 els3_dataset MC 재산출 — 독립 샤드 병렬(ProcessPool 회피, Windows OOM 방지).
 공용 엔진 module.mc 사용(40k 경로, 무보정 k=1 정본; kmap 있으면 sig_eff 버킷 vol 프리미엄).

 프로젝트 루트에서 실행(각 샤드 OMP_NUM_THREADS=1 자동):
   N=28
   seq 0 $((N-1)) | xargs -P $N -I {} python -m module.mc_shards shard {} $N   # 계산 (샤드별 진행 출력)
   python -m module.mc_shards monitor $N                                       # (선택) 다른 터미널: 전체 진행률+ETA
   python -m module.mc_shards combine $N

 모드:
   shard <k> <N>     : 인덱스 tasks[k::N] 처리 → data/cache/mccal_shard_<k>.json
                       (200개마다 scratch/mc_progress/shard_<k>.json 에 진행 heartbeat 기록)
   monitor <N> [초]  : 전 샤드 heartbeat 를 합산해 '누적 n/전체 · 처리율 · ETA · 완료샤드' 를 한 줄로 갱신(기본 5초)
   combine <N>       : 샤드 합쳐 data/els3_dataset.parquet 에 mc·MC입력·recent_margin 기록 + heartbeat 정리
   test              : 앞 30개 단일프로세스 검증(보정 배수·mc 확인)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 프로젝트 루트 (직접 실행 대비)
import json, time, gc, bisect
import numpy as np
import pandas as pd
from util import file_manager as fm
from module import mc as MC

FACE = 10000


def _tasks():
    df = pd.read_parquet(fm.source()).sort_values("isu_ord").reset_index(drop=True)
    tasks = [(i, r["item"], int(r["isu_ord"]), float(r["B"]), float(r["coupon"]),
              float(r["tenor"]), float(r["sig_eff"])) for i, r in df.iterrows()]
    return df, tasks


def _clean(x):
    return None if (isinstance(x, float) and np.isnan(x)) else float(x)


def combine(N=None, verbose=True):
    """캐시된 샤드 JSON(data/cache/mccal_shard_*.json)을 읽어 els3_dataset.parquet 에
    mc·MC입력·recent_margin 을 붙여 저장(재시뮬 없이 수 초). N=None 이면 존재하는 샤드 수 자동감지.
    노트북·CLI 공용. 반환: 저장된 df. (JSON 없으면 FileNotFoundError)"""
    if N is None:
        N = len(list(fm.CACHE.glob("mccal_shard_*.json")))
    if N == 0:
        raise FileNotFoundError(f"mccal_shard_*.json 없음 ({fm.CACHE}) — 먼저 shard 로 MC 를 계산하세요.")
    df, _ = _tasks(); res = {}
    for k in range(N):
        p = fm.CACHE / f"mccal_shard_{k}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p.name} 없음 — 샤드 0..{N-1} 이 모두 있어야 합니다(shard 재실행).")
        for ky, v in json.loads(p.read_text()).items():
            res[int(ky)] = v
    arr = {c: np.full(len(df), np.nan) for c in MC.MC_COLS}
    for i, v in res.items():
        for c, val in zip(MC.MC_COLS, v):
            arr[c][i] = (np.nan if val is None else val)
    for c in MC.MC_COLS:
        df[c] = arr[c].astype("float32")
    df["fair_minus_mc"] = (df["fair"] - df["mc"]).astype("float32")
    df["mc_krw"] = (df["mc"] * FACE).round().astype("float32")
    df["fair_krw"] = (df["fair"] * FACE).round().astype("float32")
    df["fair_minus_mc_krw"] = ((df["fair"] - df["mc"]) * FACE).round().astype("float32")
    df = df.sort_values("isu_ord").reset_index(drop=True)
    # recent_margin: 발행 전 90일 (fair-mc) 평균 (인과적)
    o = df["isu_ord"].tolist(); mg = (df["fair"] - df["mc"]).values; rm = np.zeros(len(df))
    for i in range(len(df)):
        hi = bisect.bisect_left(o, o[i]); lo = bisect.bisect_left(o, o[i] - 90)
        rm[i] = mg[lo:hi].mean() if hi > lo else (mg[:hi].mean() if hi > 0 else 0.0)
    df["recent_margin"] = rm.astype("float32")
    df.to_parquet(fm.source())
    pdir = fm.SCRATCH / "mc_progress"           # 진행 heartbeat 정리
    if pdir.exists():
        for f in pdir.glob("shard_*.json"):
            f.unlink()
    if verbose:
        print(f"combined | rows {len(df)} | mc {df.mc.mean():.4f}({df.mc_krw.mean():.0f}원) | "
              f"fair {df.fair.mean():.4f}({df.fair_krw.mean():.0f}원) | fair-mc {df.fair_minus_mc_krw.mean():+.0f}원 | "
              f"k mean {df.mc_k.mean():.3f} | nan {int(df.mc.isna().sum())}")
    return df


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kmap = MC.load_kmap()

    if mode == "shard":
        k = int(sys.argv[2]); N = int(sys.argv[3])
        mk = MC.load_market(); _, tasks = _tasks(); my = tasks[k::N]
        pdir = fm.SCRATCH / "mc_progress"; pdir.mkdir(parents=True, exist_ok=True)
        hb = pdir / f"shard_{k}.json"

        def _beat(done, finished=False):
            hb.write_text(json.dumps({"done": done, "total": len(my), "t": time.time(), "finished": finished}))

        out = {}; t0 = time.time(); _beat(0)
        for n_, (idx, it, iord, B, c, ten, se) in enumerate(my):
            r = MC.price_one(mk, kmap, it, iord, B, c, ten, se)
            out[str(idx)] = [_clean(v) for v in r]
            if (n_ + 1) % 200 == 0:
                gc.collect(); _beat(n_ + 1)
                print(f"shard{k}: {n_+1}/{len(my)} ({time.time()-t0:.0f}s)", flush=True)
        (fm.CACHE / f"mccal_shard_{k}.json").write_text(json.dumps(out))
        _beat(len(my), finished=True)
        print(f"shard{k} DONE {len(out)} in {time.time()-t0:.0f}s", flush=True)

    elif mode == "combine":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else None
        combine(N)
        print("DONE")

    elif mode == "monitor":
        N = int(sys.argv[2]); interval = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        df, _ = _tasks(); total = len(df)
        pdir = fm.SCRATCH / "mc_progress"
        t0 = time.time()
        while True:
            done = fin = 0
            for k in range(N):
                p = pdir / f"shard_{k}.json"
                if p.exists():
                    try:
                        d = json.loads(p.read_text())
                        done += int(d.get("done", 0)); fin += bool(d.get("finished"))
                    except Exception:
                        pass
            el = time.time() - t0; rate = done / el if el > 0 else 0.0
            eta = f"{(total-done)/rate/60:.1f}분" if rate > 0 else "?"
            print(f"\r진행 {done:,}/{total:,} ({100*done/max(total,1):4.1f}%) | {rate:6.1f}/s | "
                  f"ETA {eta} | 완료샤드 {fin}/{N}   ", end="", flush=True)
            if fin >= N or done >= total:
                print(f"\n완료: {done:,}/{total:,} in {el/60:.1f}분"); break
            time.sleep(interval)

    else:  # test
        mk = MC.load_market(); df, tasks = _tasks()
        t0 = time.time(); rows = [MC.price_one(mk, kmap, *t[1:]) for t in tasks[:30]]
        mc = np.array([r[0] for r in rows]); kk = np.array([r[8] for r in rows]); fair = df["fair"].values[:30]
        print(f"kmap: {kmap}")
        print(f"test 30 in {time.time()-t0:.0f}s | mc {np.nanmean(mc):.4f} | fair {np.nanmean(fair):.4f} | "
              f"fair-mc {np.nanmean(fair-mc)*FACE:+.0f}원 | k [{kk.min():.3f},{kk.max():.3f}] mean {kk.mean():.3f}")


if __name__ == "__main__":
    main()
