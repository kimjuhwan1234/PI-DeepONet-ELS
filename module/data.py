# -*- coding: utf-8 -*-
"""데이터 스키마·생성·로딩. 정통 DeepONet 재설계:
 branch = 시장상태(바스켓 vol·corr + 수익률곡선), trunk = 계약(전체 STRK 스케줄 + 배리어 + 쿠폰 + 만기).
 PI-DeepONet만 trunk에 평가좌표(S,τ,I)를 추가로 넣고 BS-PDE 물리를 건다.
 컬럼명은 raw DART와 직접 비교되도록 raw 이름 활용. 데이터셋은 CSV.
"""
import bisect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from util import file_manager as fm

# ===== 원천 컬럼명 -> 데이터셋 컬럼명 (raw 이름/산식 활용) =====
RENAME = {
    "fair": "FAIR_VALUE/ISU_PRC_DETAIL",
    "coupon": "ANL_RTRN/100",
    "B": "BARR_1/100",
    "tenor": "TENOR",
    "isu_ord": "ISU_DT_ordinal",
    "mc": "MC",
    # pass-through
    "amt": "ACT_ISU_AMT", "sbrt": "SB_RT", "dvrt": "DV_RT", "prcp": "PRCP_GRTE_RT",
    "kigrc": "KNCK_IN_GRC_PRD", "issuer": "ISU_ORG", "risk": "RISK_GRADE",
    "ptype": "PRODUCT_TYPE", "rdmp": "RDMP_TYPE", "iyear": "ISU_DT_year",
    "imonth": "ISU_DT_month", "subdays": "SUB_END_DT-SUB_START_DT",
    "K": "STRK_1_last/100", "Kfirst": "STRK_1_first/100",
    "item": "ITEM_CD",   # 행 식별 인덱스 (훈련 미사용, 예측→상품 추적용)
}
INV = {v: k for k, v in RENAME.items()}


def _new(name):
    return RENAME.get(name, name)


# ===== 피처 그룹 =====
# tabular (ml.csv): 벤치마크 + 하이브리드 stage-2 마진모델
BASE = [_new(c) for c in ["sig1", "sig2", "sig3", "rho12", "rho13", "rho23", "sig_mean", "rho", "r", "B",
        "K", "Kfirst", "coupon", "tenor", "nobs", "cpn_spread", "b_over_k", "stepdown",
        "mom6m", "amt", "sbrt", "dvrt", "prcp", "kigrc", "iyear", "subdays"]]
REG = ["recent_margin", "recent_mktvol", "curve_level", "curve_slope", "curve_curv", "issue_intensity"]
CAT = [_new(c) for c in ["issuer", "risk", "ptype", "rdmp", "imonth"]]

# 연산자망 (deeponet.csv)
VOLCORR = ["sig1", "sig2", "sig3", "rho12", "rho13", "rho23", "sig_eff"]   # branch: 바스켓 변동성·상관
UC = [f"u{j}" for j in range(10)]                                         # branch: 수익률곡선 10노드
NSTRK = 12
STRK = [f"strk_{j}" for j in range(NSTRK)]                                # trunk: 전체 오토콜 STRK 스케줄(/100, 패딩)

# ml BASE 에도 전체 행사가 스케줄 포함 (deeponet trunk 와 동일 정보). K/Kfirst 는 파생피처(b_over_k, stepdown)용으로 유지.
BASE = BASE + STRK
CONTRACT = STRK + [_new("B"), _new("coupon"), _new("tenor")]              # trunk: 계약(STRK + 배리어 + 쿠폰 + 만기)

INDEX = _new("item")      # ITEM_CD (행 식별 인덱스, 훈련 미사용)
TARGET = _new("fair")     # FAIR_VALUE/ISU_PRC_DETAIL
ANCHOR = _new("mc")       # MC
MARGIN = "recent_margin"
ORDER = _new("isu_ord")   # ISU_DT_ordinal
SRC_ORDER = "isu_ord"
TENOR = _new("tenor")     # TENOR
BARR = _new("B")          # BARR_1/100
COUPON = _new("coupon")   # ANL_RTRN/100
RF = "r"                  # 무위험금리 (물리용)
SIGEFF = "sig_eff"
CSV_ENC = "utf-8-sig"


def featnum(feat):
    return BASE + (REG if feat == "regime" else [])


def _strike_schedule():
    """raw SCHD_INFO(SCHD_TYPE=1)에서 ITEM_CD별 전체 STRK 스케줄(/100) 리스트."""
    sc = pd.read_csv(fm.RAW / "LAKE_V2_DART_SCHD_INFO.csv", low_memory=False)
    sc1 = sc[sc["SCHD_TYPE"] == 1].sort_values(["ITEM_CD", "SEQ"])
    return sc1.groupby("ITEM_CD")["STRK_1"].apply(lambda s: [float(x) / 100 for x in s])


def _pad(strikes, k=NSTRK):
    """길이 k로 고정: 짧으면 마지막값 forward-fill, 길면 앞 k개로 절단."""
    if strikes is None or len(strikes) == 0:
        return [np.nan] * k
    s = list(strikes[:k])
    if len(s) < k:
        s = s + [s[-1]] * (k - len(s))
    return s


def _load_source():
    """원천 = data/els3_dataset.parquet (0_data build_source가 raw+cache로 생성한 branch·aux + 1_MC_recompute가 갱신한 mc·recent_margin)
    + issue_intensity + raw SCHD STRK 스케줄."""
    df = pd.read_parquet(fm.source()).sort_values(SRC_ORDER).reset_index(drop=True)
    o = df[SRC_ORDER].tolist()
    df["issue_intensity"] = [bisect.bisect_left(o, o[i]) - bisect.bisect_left(o, o[i] - 90)
                             for i in range(len(df))]
    strk_by = _strike_schedule()
    padded = [_pad(strk_by.get(str(it))) for it in df["item"].astype(str)]
    for j in range(NSTRK):
        df[f"strk_{j}"] = [p[j] for p in padded]
    return df


# ===== 0_data: 데이터셋 생성 (CSV) =====
def build_datasets():
    """원천 -> data/ml.csv, data/deeponet.csv 생성.
     ml = tabular(BASE+REG+CAT), deeponet = 연산자망 입력(곡선 u0-9 + vol·corr·sig_eff + 계약 + r). 둘 다 공통(fair·mc·recent_margin) 포함."""
    fm.ensure_dirs()
    df = _load_source()
    common = [INDEX, ORDER, TARGET, ANCHOR, MARGIN]   # INDEX(ITEM_CD) 선두 — 모든 데이터셋에 인덱스 컬럼

    def write(name, new_cols):
        new_cols = list(dict.fromkeys(new_cols))
        old_cols = [INV.get(c, c) for c in new_cols]
        out = df[old_cols].rename(columns=RENAME)
        out.to_csv(fm.dataset(name), index=False, encoding=CSV_ENC)
        return list(out.columns)

    cols_ml = write("ml", [INDEX] + BASE + REG + CAT + common)
    # DeepONet 연산자망 입력: 곡선 u0-9 + vol·corr·sig_eff + 계약(strk0-11,B,coupon,tenor) + r (stage1 앵커/stage2 D.DON 공용)
    cols_dn = write("deeponet", [INDEX] + UC + VOLCORR + CONTRACT + [RF] + common)
    return {"ml": len(df), "deeponet": len(df),
            "columns": {"ml": cols_ml, "deeponet": cols_dn}}


# ===== 1_run: 통합 로딩 =====
def load(cfg):
    ml = pd.read_csv(fm.dataset("ml"), encoding=CSV_ENC).sort_values(ORDER).reset_index(drop=True)
    don = pd.read_csv(fm.dataset("deeponet"), encoding=CSV_ENC).sort_values(ORDER).reset_index(drop=True)
    n = len(ml)
    assert len(don) == n, "데이터셋 행수 불일치"
    for c in CAT:
        ml[c] = ml[c].astype(str).astype("category")

    dev = "cuda" if (cfg.get("device") != "cpu" and torch.cuda.is_available()) else "cpu"
    D = SimpleNamespace()
    D.n = n; D.SMAX = float(cfg["data"]["smax"]); D.DEV = dev
    D.ml = ml
    D.FAIR = ml[TARGET].values.astype("float32")
    D.MC = ml[ANCHOR].values.astype("float32")
    D.rm = ml[MARGIN].values.astype("float32")
    D.ITEM = ml[INDEX].astype(str).values   # ITEM_CD 인덱스 (훈련 미사용, 예측 추적용)
    D.ORD = ml[ORDER].values.astype(float)
    # 연산자망 입력 (branch=vol·corr+곡선, trunk=계약) — deeponet.csv
    D.VC = don[VOLCORR].values.astype("float32")      # (n, 7)
    D.CURVE = don[UC].values.astype("float32")        # (n, 10)
    D.CON = don[CONTRACT].values.astype("float32")    # (n, 15) = strk12 + BARR + coupon + TENOR
    D.R = don[RF].values.astype("float32")            # r
    D.TEN = don[TENOR].values.astype("float32")       # 만기
    D.SIGEFF = don[SIGEFF].values.astype("float32")   # σ_eff
    # stage-2 잔차모델 입력 = deeponet.csv 특성 블록 [곡선|vol·corr·sig_eff|계약] (= 앵커 입력, 이론가 결정 특성만; ml aux/범주형 제외)
    D.DON = np.concatenate([D.CURVE, D.VC, D.CON], axis=1).astype("float32")   # (n, 32)
    # CONTRACT 내 인덱스 (물리/payoff용)
    D.iK = CONTRACT.index(_new("K")) if _new("K") in CONTRACT else NSTRK - 1   # 만기 행사가 = strk_{last}
    D.iKlast = NSTRK - 1
    D.iBARR = CONTRACT.index(BARR)
    D.iCOUPON = CONTRACT.index(COUPON)
    D.iTEN = CONTRACT.index(TENOR)
    D.WF = walk_forward(n, cfg["data"]["walk_forward"],
                        cfg["data"].get("val_frac", 0.0), cfg["data"].get("val_seed", 0))
    return D


def walk_forward(n, bounds, val_frac=0.0, val_seed=0):
    """확장윈도우 walk-forward. 반환: [(tr, va, te)].
     tr=학습(과거, val 제외) / va=validation(train에서 랜덤샘플, 미래참조 허용, early stopping용) / te=test(미래 OOS).
     val_frac=0 이면 va=빈배열 (기존 동작)."""
    rng = np.random.default_rng(val_seed)
    folds = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tr_all = np.arange(0, int(a * n))
        te = np.arange(int(a * n), int(b * n))
        if val_frac and len(tr_all) > 10:
            nv = max(1, int(round(val_frac * len(tr_all))))
            va = np.sort(rng.choice(tr_all, size=nv, replace=False))
            tr = np.setdiff1d(tr_all, va, assume_unique=True)
        else:
            va = np.empty(0, dtype=int); tr = tr_all
        folds.append((tr, va, te))
    return folds


def time_weights(D, tr):
    return (0.5 ** ((D.ORD[tr].max() - D.ORD[tr]) / 365.25)).astype("float32")


def to_tensor(a, dev):
    return torch.tensor(np.asarray(a, dtype="float32"), device=dev)


def zstats(a, tr):
    """train 기준 표준화 통계."""
    return a[tr].mean(0), a[tr].std(0) + 1e-8
