# -*- coding: utf-8 -*-
"""직접학습 tabular 벤치마크: GBM/Ridge/LightGBM/CatBoost/XGBoost (regime 피처, FAIR 직접)."""
from module.pipeline import predict_direct_tab

MODELS = {
    "bench_gbm": "gbm",
    "bench_ridge": "ridge",
    "bench_lgbm": "lgbm",
    "bench_catboost": "cat",
    "bench_xgboost": "xgb",
}


def run(D, cfg):
    return {name: predict_direct_tab(D, cfg, key) for name, key in MODELS.items()}
