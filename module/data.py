# -*- coding: utf-8 -*-
"""데이터 스키마(피처 그룹), 데이터셋 생성/로딩, walk-forward 폴드, 전처리 헬퍼."""
import bisect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from util import file_manager as fm

# ===== 피처 그룹 (데이터 스키마) =====
BASE = [
    "sig1",
    "sig2",
    "sig3",
    "rho12",
    "rho13",
    "rho23",
    "sig_mean",
    "rho",
    "r",
    "B",
    "K",
    "Kfirst",
    "coupon",
    "tenor",
    "nobs",
    "cpn_spread",
    "b_over_k",
    "stepdown",
    "mom6m",
    "amt",
    "sbrt",
    "dvrt",
    "prcp",
    "kigrc",
    "iyear",
    "subdays",
]
REG = [
    "recent_margin",
    "recent_mktvol",
    "curve_level",
    "curve_slope",
    "curve_curv",
    "issue_intensity",
]
CAT = ["issuer", "risk", "ptype", "rdmp", "imonth"]
PB = ["sig_eff", "r", "B", "Kfirst", "K", "coupon", "tenor"]  # DeepONet(MLP) branch
UC = [f"u{j}" for j in range(10)]  # 수익률곡선 10노드
DPHYS = [
    "sig_eff",
    "sig1",
    "sig2",
    "sig3",
    "B",
    "K",
    "coupon",
    "rho12",
    "rho13",
    "rho23",
    "cpn_spread",
    "b_over_k",
    "stepdown",
    "mom6m",
    "recent_margin",
    "recent_mktvol",
    "curve_slope",
    "curve_curv",
]
DNPN = ["amt", "sbrt", "dvrt", "prcp", "kigrc", "iyear", "subdays"]
COND = ["sig_eff", "r", "B", "Kfirst", "K", "coupon"]  # curve PI branch 조건
TARGET = "fair"
ANCHOR = "mc"
MARGIN = "recent_margin"
ORDER = "isu_ord"


def featnum(feat):
    """tabular 수치 피처: base 또는 regime(=base+reg)."""
    return BASE + (REG if feat == "regime" else [])


def _load_source():
    """원천(피처 완성본) 로딩 + isu_ord 정렬 + issue_intensity 파생."""
    df = pd.read_parquet(fm.source()).sort_values(ORDER).reset_index(drop=True)
    o = df[ORDER].tolist()
    df["issue_intensity"] = [
        bisect.bisect_left(o, o[i]) - bisect.bisect_left(o, o[i] - 90)
        for i in range(len(df))
    ]
    return df


# ===== 0_data: 3종 데이터셋 생성 =====
def build_datasets():
    """원천 -> data/ml.parquet, data/deeponet.parquet, data/deeponet_curve.parquet 생성."""
    fm.ensure_dirs()
    df = _load_source()
    common = [ORDER, TARGET, ANCHOR, MARGIN]

    # ① ML: 직접 tabular 벤치마크 + 하이브리드 stage-2 마진모델 (base+reg+cat + fair+mc+margin)
    ml_cols = list(dict.fromkeys(BASE + REG + CAT + common))
    ml = df[ml_cols].copy()
    ml.to_parquet(fm.dataset("ml"))

    # ② DeepONet/PI-DeepONet(MLP): PB + fair + mc + margin
    dn_cols = list(dict.fromkeys(PB + common))
    df[dn_cols].to_parquet(fm.dataset("deeponet"))

    # ③ DeepONet-Curve: u0..u9 + DPHYS + DNPN + CAT + COND + tenor + fair + mc + margin
    cv_cols = list(dict.fromkeys(UC + DPHYS + DNPN + CAT + COND + ["tenor"] + common))
    df[cv_cols].to_parquet(fm.dataset("deeponet_curve"))

    return {
        "ml": len(ml),
        "deeponet": len(df),
        "deeponet_curve": len(df),
        "columns": {"ml": ml_cols, "deeponet": dn_cols, "deeponet_curve": cv_cols},
    }


# ===== 1_run: 통합 로딩 =====
def load(cfg):
    """3종 데이터셋을 row-정렬 정합으로 로딩해 통합 번들 반환.

    데이터셋은 isu_ord 정렬 동일 순서라 인덱스 정합. 모델은 필요한 배열만 사용.
    """
    ml = pd.read_parquet(fm.dataset("ml"))
    dn = pd.read_parquet(fm.dataset("deeponet"))
    cv = pd.read_parquet(fm.dataset("deeponet_curve"))
    # 정렬 정합 보장
    ml = ml.sort_values(ORDER).reset_index(drop=True)
    dn = dn.sort_values(ORDER).reset_index(drop=True)
    cv = cv.sort_values(ORDER).reset_index(drop=True)
    n = len(ml)
    assert len(dn) == n and len(cv) == n, "데이터셋 행수 불일치"

    for c in CAT:
        ml[c] = ml[c].astype(str).astype("category")
        cv[c] = cv[c].astype(str).astype("category")

    smax = float(cfg["data"]["smax"])
    dev = (
        "cuda"
        if (
            cfg.get("device", "auto") != "cpu"
            and torch.cuda.is_available()
            and cfg.get("device") != "cpu"
        )
        else "cpu"
    )
    if cfg.get("device") == "cpu":
        dev = "cpu"

    D = SimpleNamespace()
    D.n = n
    D.SMAX = smax
    D.DEV = dev
    D.ml = ml  # tabular(벤치마크+마진) df (categorical 포함)
    D.FAIR = ml[TARGET].values.astype("float32")
    D.MC = ml[ANCHOR].values.astype("float32")
    D.rm = ml[MARGIN].values.astype("float32")
    D.ORD = ml[ORDER].values.astype(float)
    # DeepONet(MLP)
    D.Mv = dn[PB].values.astype("float32")
    # DeepONet-Curve
    D.Uarr = cv[UC].values.astype("float32")
    D.Dphys = cv[DPHYS].values.astype("float32")
    D.Dnp = cv[DNPN].values.astype("float32")
    D.Cond = cv[COND].values.astype("float32")
    D.TENv = cv["tenor"].values.astype("float32")
    D.CCs = {c: cv[c].astype(str) for c in CAT}
    D.CAT = CAT
    # walk-forward 폴드
    D.WF = walk_forward(n, cfg["data"]["walk_forward"])
    return D


def walk_forward(n, bounds):
    """확장윈도우 폴드: [(train_idx, test_idx), ...]. bounds 예: [0.6,0.7,0.8,0.9,1.0]."""
    folds = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        folds.append((np.arange(0, int(a * n)), np.arange(int(a * n), int(b * n))))
    return folds


def time_weights(D, tr):
    """시간감쇠 표본가중 0.5^(dt/365.25)."""
    return (0.5 ** ((D.ORD[tr].max() - D.ORD[tr]) / 365.25)).astype("float32")


def to_tensor(a, dev):
    return torch.tensor(np.asarray(a, dtype="float32"), device=dev)


# ===== 전처리 (연산자망) =====
def curve_prep(D, tr):
    """DeepONet-Curve 입력 정규화 + 범주형 코드. 반환: Un, CATt, card."""
    um, us = D.Uarr[tr].mean(0), D.Uarr[tr].std(0) + 1e-8
    card, codes = [], []
    for cc in CAT:
        mp = {v: i + 1 for i, v in enumerate(pd.Index(D.CCs[cc].iloc[tr].unique()))}
        codes.append(D.CCs[cc].map(mp).fillna(0).astype(int).values)
        card.append(len(mp) + 1)
    Un = to_tensor((D.Uarr - um) / us, D.DEV)
    CATt = torch.tensor(np.stack(codes, 1), device=D.DEV, dtype=torch.long)
    return Un, CATt, card


def don_direct_prep(D, tr):
    """DONdirect(곡선 직접) 입력: 곡선/물리/비물리 정규화 + 범주형. 반환: Un,PHn,NPn,CATt,card."""
    um, us = D.Uarr[tr].mean(0), D.Uarr[tr].std(0) + 1e-8
    pm, ps = D.Dphys[tr].mean(0), D.Dphys[tr].std(0) + 1e-8
    npv = D.Dnp.copy()
    for j in range(npv.shape[1]):
        med = np.nanmedian(npv[tr, j])
        c = npv[:, j]
        c[np.isnan(c)] = med
        npv[:, j] = c
    npv[:, 0] = np.log1p(np.clip(npv[:, 0], 0, None))
    nm, ns = npv[tr].mean(0), npv[tr].std(0) + 1e-8
    card, codes = [], []
    for cc in CAT:
        mp = {v: i + 1 for i, v in enumerate(pd.Index(D.CCs[cc].iloc[tr].unique()))}
        codes.append(D.CCs[cc].map(mp).fillna(0).astype(int).values)
        card.append(len(mp) + 1)
    return (
        to_tensor((D.Uarr - um) / us, D.DEV),
        to_tensor((D.Dphys - pm) / ps, D.DEV),
        to_tensor((npv - nm) / ns, D.DEV),
        torch.tensor(np.stack(codes, 1), device=D.DEV, dtype=torch.long),
        card,
    )
