# -*- coding: utf-8 -*-
"""raw + cache → els3_dataset (원천 재유도). exp15 로직을 정식 이식하되 '현재 정의'로 빌드:
   변동성·상관 = 180일 역사(module.features), 수익률곡선 = KRW Nelson-Siegel. MC 는 여기서 계산하지 않음(1_MC_recompute).
 산출: data/els3_dataset.parquet (보조피처 + branch[sig/rho/sig_eff/u/r/curve] + fair + recent_mktvol). item·isu_ord 포함."""
import io, sys, json, time
from datetime import date
import numpy as np
import pandas as pd
import bisect

from util import file_manager as fm
from . import features as F


def _safe(t):
    return "px_" + t.replace("^", "_").replace(".", "_") + ".parquet"


def build_source(save=True, verbose=True):
    CA, RAW = fm.CACHE, fm.RAW
    mapping = json.loads((CA / "udly_ticker_map.json").read_text(encoding="utf-8"))
    RET = {}
    for uid, t in mapping.items():
        if not t:
            continue
        f = CA / _safe(t)
        if t in RET or not f.exists():
            continue
        s = pd.read_parquet(f)["close"].dropna(); s = s[~s.index.duplicated()]
        RET[t] = np.log(s).diff()
    KRW = pd.read_parquet(CA / "krw_curve.parquet")[["call", "m3", "y10"]]

    ac = pd.read_csv(RAW / "LAKE_V2_DART_AUTO_CALL.csv", low_memory=False)
    sc = pd.read_csv(RAW / "LAKE_V2_DART_SCHD_INFO.csv", low_memory=False)
    ud = pd.read_csv(RAW / "LAKE_V2_DART_UDLY_INFO.csv", low_memory=False)
    for c in ("ISU_DT", "MAT_DT", "SUB_START_DT", "SUB_END_DT"):
        ac[c] = pd.to_datetime(ac[c], errors="coerce")
    ac["tenor"] = (ac["MAT_DT"] - ac["ISU_DT"]).dt.days / 365.25
    ac["fair"] = ac["FAIR_VALUE"] / ac["ISU_PRC_DETAIL"]
    nu = ud.groupby("ITEM_CD")["UDLY_ID"].nunique(); three = nu[nu == 3].index
    u3map = ud[ud.ITEM_CD.isin(three)].groupby("ITEM_CD")["UDLY_ID"].apply(list)
    sc1 = sc[sc.SCHD_TYPE == 1].sort_values(["ITEM_CD", "SEQ"])
    strk_by = sc1.groupby("ITEM_CD")["STRK_1"].apply(lambda s: list(s / 100))
    barr_by = sc1.groupby("ITEM_CD")["BARR_1"].min() / 100
    cand = ac[(ac.ITEM_CD.isin(three)) & (ac.OPT_TYPE == "STEP") & (ac.CUR_CD == "KRW")
              & ac.fair.between(0.7, 1.05) & ac.tenor.between(0.5, 5)]
    if verbose:
        print("candidate 3-star STEP KRW:", len(cand))

    def mom6m(rets, dt, win=126):
        vals = []
        for r in rets:
            if r is None:
                return 0.0
            w = r[r.index < dt].tail(win)
            if len(w) < 40:
                return 0.0
            vals.append(float(w.sum()))
        return float(np.mean(vals)) if vals else 0.0

    rows = []; t0 = time.time()
    for it, rw in cand.set_index("ITEM_CD").iterrows():
        try:
            ts = [mapping.get(x) for x in u3map.get(it, [])]
            if len(ts) != 3 or any(t is None for t in ts):
                continue
            rets = [RET.get(t) for t in ts]
            if any(r is None for r in rets):
                continue
            dt = rw.ISU_DT
            sigs = [F.vol180(r, dt) for r in rets]                 # 현재 정의: 180일 역사 변동성
            if any(pd.isna(sigs)):
                continue
            corr = F.corr180(rets, dt)                             # 180일 역사 상관
            if corr is None:
                continue
            strikes = strk_by.get(it)
            if strikes is None or len(strikes) < 2 or any(pd.isna(strikes)):
                continue
            B = barr_by.get(it); c = rw.ANL_RTRN / 100; ten = rw.tenor
            if any(pd.isna(x) for x in [B, c, ten]):
                continue
            beta = F.krw_beta(KRW.asof(dt).values)                 # KRW NS 곡선
            if beta is None:
                continue
            u = F.krw_curve_nodes(beta); r = float(F.zero_curve(beta, np.array([ten]))[0])
            iu = np.triu_indices(3, 1); rr = np.sort(np.clip(corr[iu], -0.999, 0.999))
            rho12, rho13, rho23 = float(rr[0]), float(rr[1]), float(rr[2])
            rho_v = float(np.mean(rr))
            ss = np.sort(sigs); sig1, sig2, sig3 = float(ss[0]), float(ss[1]), float(ss[2])
            smean = float(np.mean(sigs)); sig_eff = smean * float(np.sqrt(1.0 + max(0.0, 1.0 - rho_v)))
            Klast = float(strikes[-1]); Kfst = float(strikes[0])
            sd = (rw.SUB_END_DT - rw.SUB_START_DT).days if pd.notna(rw.SUB_END_DT) and pd.notna(rw.SUB_START_DT) else np.nan
            rec = dict(item=it, sig1=sig1, sig2=sig2, sig3=sig3, rho12=rho12, rho13=rho13, rho23=rho23,
                       sig_mean=smean, sig_max=sig3, sig_min=sig1, rho=rho_v, sig_eff=sig_eff,
                       cpn_spread=float(c - r), b_over_k=float(B / Klast) if Klast > 0 else 1.0,
                       stepdown=float(Kfst - Klast), mom6m=mom6m(rets, dt), isu_ord=int(dt.toordinal()),
                       r=r, B=float(B), Kfirst=Kfst, K=Klast, coupon=float(c), tenor=float(ten), nobs=len(strikes),
                       fair=float(rw.fair), issuer=str(rw.ISU_ORG), risk=str(rw.RISK_GRADE),
                       ptype=str(rw.PRODUCT_TYPE), rdmp=str(rw.RDMP_TYPE), imonth=str(int(dt.month)),
                       amt=float(rw.ACT_ISU_AMT) if pd.notna(rw.ACT_ISU_AMT) else np.nan,
                       sbrt=float(rw.SB_RT) if pd.notna(rw.SB_RT) else np.nan,
                       dvrt=float(rw.DV_RT) if pd.notna(rw.DV_RT) else np.nan,
                       prcp=float(rw.PRCP_GRTE_RT) if pd.notna(rw.PRCP_GRTE_RT) else np.nan,
                       kigrc=float(rw.KNCK_IN_GRC_PRD) if pd.notna(rw.KNCK_IN_GRC_PRD) else np.nan,
                       iyear=float(dt.year), subdays=float(sd))
            for j in range(10):
                rec[f"u{j}"] = float(u[j])
            rec["curve_level"] = float(u.mean()); rec["curve_slope"] = float(u[9] - u[0])
            rec["curve_curv"] = float(2 * u[4] - u[0] - u[9])
            rows.append(rec)
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("isu_ord").reset_index(drop=True)
    if verbose:
        print(f"built {len(df)} products in {time.time()-t0:.0f}s")
    # recent_mktvol: 발행 전 90일 발행분 평균 sig_mean (인과적)
    o = df["isu_ord"].tolist(); sm = df["sig_mean"].values; rmv = np.zeros(len(df))
    for i in range(len(df)):
        hi = bisect.bisect_left(o, o[i]); lo = bisect.bisect_left(o, o[i] - 90)
        rmv[i] = sm[lo:hi].mean() if hi > lo else (sm[:hi].mean() if hi > 0 else sm[i])
    df["recent_mktvol"] = rmv.astype("float32")
    if save:
        out = fm.DATA / "els3_dataset.parquet"
        df.to_parquet(out)
        if verbose:
            print("saved", out, "rows", len(df))
    return df
