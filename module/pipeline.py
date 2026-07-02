# -*- coding: utf-8 -*-
"""walk-forward OOS 예측 조립: 직접(tabular/NN)·하이브리드. 반환 DataFrame."""
import numpy as np
import pandas as pd

from .tabular import fit_tab


def predict_direct_tab(D, cfg, model_key):
    """직접 tabular 회귀 (regime 피처, FAIR 타깃)."""
    rows = []
    for tr, te in D.WF:
        yp = fit_tab(
            D, cfg, model_key, tr, te, "regime", D.FAIR, tw=cfg["data"]["time_decay"]
        )
        rows.append(
            pd.DataFrame({"isu_ord": D.ORD[te], "y_true": D.FAIR[te], "y_pred": yp})
        )
    return pd.concat(rows, ignore_index=True)


def predict_direct_nn(D, cfg, train_fn):
    """직접 연산자망 (FAIR 직접학습). train_fn(D,cfg,tr,te)->(_, fair_te)."""
    rows = []
    for tr, te in D.WF:
        _, yp = train_fn(D, cfg, tr, te)
        rows.append(
            pd.DataFrame({"isu_ord": D.ORD[te], "y_true": D.FAIR[te], "y_pred": yp})
        )
    return pd.concat(rows, ignore_index=True)


def predict_hybrid(D, cfg, anchor_fn):
    """하이브리드: FAIR = MC_hat(anchor) + recent_margin + resid_hat(margin model).

    anchor_fn(D,cfg,tr,te) -> (mc_tr, mc_te). stage-2 마진모델은 config 설정 사용(base 피처).
    """
    mkey = cfg["margin"]["model"]
    feat = cfg["margin"]["feature_set"]
    resid_target = (D.FAIR - D.MC - D.rm).astype("float32")
    rows = []
    for tr, te in D.WF:
        _, mc_te = anchor_fn(D, cfg, tr, te)
        resid_pred = fit_tab(
            D, cfg, mkey, tr, te, feat, resid_target, tw=cfg["data"]["time_decay"]
        )
        y_pred = mc_te + D.rm[te] + resid_pred
        rows.append(
            pd.DataFrame(
                {
                    "isu_ord": D.ORD[te],
                    "y_true": D.FAIR[te],
                    "y_pred": y_pred,
                    "mc_true": D.MC[te],
                    "mc_pred": mc_te,
                    "resid_true": resid_target[te],
                    "resid_pred": resid_pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
