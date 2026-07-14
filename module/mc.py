# -*- coding: utf-8 -*-
"""MC 이론가 엔진 (미래에셋 36449 방식) — 노트북(1_MC_recompute)·병렬 샤드 스크립트 공용(DRY).
 일별 GBM + 촐레스키(180일 역사 상관) + 180일 역사 변동성(cache px_*) + KRW Nelson-Siegel 할인 + q=0 + 일별 KI.
 곡선·vol·corr 정의는 module.features 사용.

 캘리브레이션(선택): 역사 변동성은 내재변동성보다 낮아 MC가 공정가치를 상회한다. 관측 입력(sig_eff)의
 연속 함수 k(sig_eff)=piecewise-linear(=data/cache/calib_kmap.json)를 σ에 곱해 편향을 줄인다(불연속 없음).
 kmap=None 이면 미보정(pre-calibration) 버전 = 원래 MC.
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from util import file_manager as fm
from . import features as F

NPATH = 40_000
PCHUNK = 2_000
MC_COLS = ["mc", "mc_vol1", "mc_vol2", "mc_vol3", "mc_rho12", "mc_rho13", "mc_rho23", "mc_r_krw", "mc_k"]


def _safe(t):
    return "px_" + t.replace("^", "_").replace(".", "_") + ".parquet"


def load_market():
    """cache/raw → 시장데이터 dict(RET 로그수익률, KRW 곡선, 상품→기초자산·행사가 매핑)."""
    CA, RAW = fm.CACHE, fm.RAW
    mapping = json.loads((CA / "udly_ticker_map.json").read_text(encoding="utf-8"))
    RET = {}
    for _uid, t in mapping.items():
        if not t:
            continue
        f = CA / _safe(t)
        if t in RET or not f.exists():
            continue
        s = pd.read_parquet(f)["close"].dropna(); s = s[~s.index.duplicated()]
        RET[t] = np.log(s).diff()
    KRW = pd.read_parquet(CA / "krw_curve.parquet")[["call", "m3", "y10"]]
    if not (CA / "_u3map.json").exists() or not (CA / "_strk_by.json").exists():
        ud = pd.read_csv(RAW / "LAKE_V2_DART_UDLY_INFO.csv", low_memory=False)
        sc = pd.read_csv(RAW / "LAKE_V2_DART_SCHD_INFO.csv", low_memory=False)
        (CA / "_u3map.json").write_text(json.dumps(ud.groupby("ITEM_CD")["UDLY_ID"].apply(list).to_dict()), encoding="utf-8")
        sc1 = sc[sc.SCHD_TYPE == 1].sort_values(["ITEM_CD", "SEQ"])
        (CA / "_strk_by.json").write_text(json.dumps(sc1.groupby("ITEM_CD")["STRK_1"].apply(lambda s: [float(x) / 100 for x in s]).to_dict()), encoding="utf-8")
    return dict(mapping=mapping, RET=RET, KRW=KRW,
                u3map=json.loads((CA / "_u3map.json").read_text(encoding="utf-8")),
                strk_by=json.loads((CA / "_strk_by.json").read_text(encoding="utf-8")))


def load_kmap(path=None):
    """캘리브레이션 매핑 로딩. 파일 없으면 None(미보정)."""
    p = fm.CACHE / "calib_kmap.json" if path is None else path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def calib_k(sig_eff, kmap):
    """관측 입력(sig_eff) → vol 프리미엄 배수 k. 연속 piecewise-linear 보간(knot 밖은 끝값으로 클램프).
     kmap=None 이면 1.0(미보정). (버킷 계단 버전은 불연속으로 DeepONet 앵커 학습을 해쳐 연속형으로 대체.)"""
    if kmap is None:
        return 1.0
    return float(np.interp(float(sig_eff), kmap["x"], kmap["y"]))


def mc_daily(sigs, corr, beta, B, strikes, c, ten, n=NPATH, seed=0, chunk=PCHUNK):
    """미래에셋 방식 일별 MC. 비연속-out cumsum 누수 회피 위해 연속 exp(cumsum) 사용."""
    nobs = len(strikes); N = int(round(ten * 365)); dt = 1 / 365
    obs_day = np.clip(np.round(np.arange(1, nobs + 1) * (ten / nobs) * 365).astype(int), 1, N); obs_t = obs_day / 365.0
    times = np.arange(1, N + 1) * dt; DF = np.exp(-F.zero_curve(beta, times) * times)
    lnDF = np.concatenate([[0.0], np.log(DF)]); fdt = (-(np.diff(lnDF))).astype(np.float32)
    L = F.chol_psd(np.asarray(corr)).astype(np.float32); sig = np.asarray(sigs, np.float32)
    dtf = np.float32(dt); sq = np.float32(np.sqrt(dt)); drift = fdt[:, None] - (0.5 * sig ** 2 * dtf)[None, :]
    rng = np.random.default_rng(seed); tot = 0.0; done = 0
    while done < n:
        m = min(chunk, n - done)
        Z = rng.standard_normal((m, N, 3), dtype=np.float32) @ L.T
        incr = drift[None, :, :] + sig[None, None, :] * sq * Z
        R = np.exp(np.cumsum(incr, axis=1)); WP = R.min(axis=2); ki = (WP < B).any(axis=1)
        alive = np.ones(m, bool); pay = np.zeros(m)
        for j in range(nobs):
            oi = obs_day[j]; hit = alive & (WP[:, oi - 1] >= strikes[j])
            pay[hit] = (1 + c * obs_t[j]) * DF[oi - 1]; alive[hit] = False
        wT = WP[alive, -1]; kA = ki[alive]; pay[alive] = np.where(~kA, 1.0, wT) * DF[-1]
        tot += pay.sum(); done += m
        del Z, incr, R, WP
    return tot / n


def price_one(mk, kmap, item, iord, B, c, ten, sig_eff, n=NPATH, seed=0):
    """상품 하나: (mc, vol1..3, rho12/13/23, r_krw, k). 실제 3기초자산 180일 역사 vol/corr + 캘리브레이션 배수 k.
     vol1..3 은 관측(미보정) 역사변동성이며 MC 내부에서는 k·σ 를 사용(k=적용배수)."""
    dt = pd.Timestamp(date.fromordinal(int(iord)))
    beta = F.krw_beta(mk["KRW"].asof(dt).values); strikes = mk["strk_by"].get(item)
    ts = [mk["mapping"].get(x) for x in mk["u3map"].get(item, [])]; ts = [t for t in ts if t]
    nan8 = (np.nan,) * 7
    if beta is None or not strikes:
        return (np.nan, *nan8)
    rkrw = float(F.zero_curve(beta, np.array([ten]))[0])
    k = calib_k(sig_eff, kmap)
    rets = [mk["RET"].get(t) for t in ts]
    if len(ts) == 3 and all(r is not None for r in rets):
        sigs = [F.vol180(r, dt) for r in rets]; corr = F.corr180(rets, dt)
        if corr is not None and not any(pd.isna(sigs)):
            mc = mc_daily([k * s for s in sigs], corr, beta, B, strikes, c, ten, n=n, seed=seed)
            return (mc, sigs[0], sigs[1], sigs[2], corr[0, 1], corr[0, 2], corr[1, 2], rkrw, k)
    mc = mc_daily([k * sig_eff] * 3, np.eye(3), beta, B, strikes, c, ten, n=n, seed=seed)   # 폴백: 단일자산
    return (mc, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, rkrw, k)
