# -*- coding: utf-8 -*-
"""직접학습 tabular 벤치마크(GBM/Ridge/LightGBM/CatBoost/XGBoost, regime 피처, FAIR 직접)
 + XGBoost 하이브리드(stage-1 앵커=XGB(base)→MC, + recent_margin + XGB(base)→resid)."""
from module.pipeline import predict_direct_tab, predict_hybrid
from module.tabular import fit_tab

MODELS = {
    "bench_gbm": "gbm",
    "bench_ridge": "ridge",
    "bench_lgbm": "lgbm",
    "bench_catboost": "cat",
    "bench_xgboost": "xgb",
}


def _xgb_anchor(D, cfg, tr, te):
    """stage-1 앵커: XGBoost가 무차익가 MC를 예측 (base 피처)."""
    mc_te = fit_tab(D, cfg, "xgb", tr, te, "base", D.MC, tw=cfg["data"]["time_decay"])
    return None, mc_te


def run(D, cfg):
    out = {name: predict_direct_tab(D, cfg, key) for name, key in MODELS.items()}
    out["xgb_hybrid"] = predict_hybrid(D, cfg, _xgb_anchor)   # XGBoost 하이브리드
    return out
