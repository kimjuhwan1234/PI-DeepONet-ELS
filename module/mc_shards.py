# -*- coding: utf-8 -*-
"""전체 els3_dataset 캘리브레이션 MC 재산출 — 독립 샤드 병렬(ProcessPool 회피, Windows OOM 방지).
 공용 엔진 module.mc 사용(40k 경로, sig_eff 버킷 vol 프리미엄 k = data/cache/calib_kmap.json).

 프로젝트 루트에서 실행(각 샤드 OMP_NUM_THREADS=1 자동):
   N=28
   seq 0 $((N-1)) | xargs -P $N -I {} python -m module.mc_shards shard {} $N
   python -m module.mc_shards combine $N

 모드:
   shard <k> <N>  : 인덱스 tasks[k::N] 처리 → data/cache/mccal_shard_<k>.json
   combine <N>    : 샤드 합쳐 data/els3_dataset.parquet 에 mc·MC입력·recent_margin 기록
   test           : 앞 30개 단일프로세스 검증(보정 배수·mc 확인)"""
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


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kmap = MC.load_kmap()

    if mode == "shard":
        k = int(sys.argv[2]); N = int(sys.argv[3])
        mk = MC.load_market(); _, tasks = _tasks(); my = tasks[k::N]
        out = {}; t0 = time.time()
        for n_, (idx, it, iord, B, c, ten, se) in enumerate(my):
            r = MC.price_one(mk, kmap, it, iord, B, c, ten, se)
            out[str(idx)] = [_clean(v) for v in r]
            if (n_ + 1) % 200 == 0:
                gc.collect(); print(f"shard{k}: {n_+1}/{len(my)} ({time.time()-t0:.0f}s)", flush=True)
        (fm.CACHE / f"mccal_shard_{k}.json").write_text(json.dumps(out))
        print(f"shard{k} DONE {len(out)} in {time.time()-t0:.0f}s", flush=True)

    elif mode == "combine":
        N = int(sys.argv[2]); df, _ = _tasks(); res = {}
        for k in range(N):
            d = json.loads((fm.CACHE / f"mccal_shard_{k}.json").read_text())
            for ky, v in d.items():
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
        print(f"combined | rows {len(df)} | mc {df.mc.mean():.4f}({df.mc_krw.mean():.0f}원) | "
              f"fair {df.fair.mean():.4f}({df.fair_krw.mean():.0f}원) | fair-mc {df.fair_minus_mc_krw.mean():+.0f}원 | "
              f"k mean {df.mc_k.mean():.3f} | nan {int(df.mc.isna().sum())}")
        print("DONE")

    else:  # test
        mk = MC.load_market(); df, tasks = _tasks()
        t0 = time.time(); rows = [MC.price_one(mk, kmap, *t[1:]) for t in tasks[:30]]
        mc = np.array([r[0] for r in rows]); kk = np.array([r[8] for r in rows]); fair = df["fair"].values[:30]
        print(f"kmap: {kmap}")
        print(f"test 30 in {time.time()-t0:.0f}s | mc {np.nanmean(mc):.4f} | fair {np.nanmean(fair):.4f} | "
              f"fair-mc {np.nanmean(fair-mc)*FACE:+.0f}원 | k [{kk.min():.3f},{kk.max():.3f}] mean {kk.mean():.3f}")


if __name__ == "__main__":
    main()
