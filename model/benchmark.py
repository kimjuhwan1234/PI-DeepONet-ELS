# -*- coding: utf-8 -*-
"""직접학습 tabular 벤치마크(GBM/Ridge/LightGBM/CatBoost/XGBoost, regime 피처, FAIR 직접)
 + XGBoost 하이브리드(stage-1 앵커=XGB→MC, + recent_margin + XGB(base)→resid).
 stage-1 앵커 입력은 DeepONet 앵커와 동일하게 맞춤: 곡선 u0-9 + vol·corr + 전체 계약(strk0-11,B,coupon,tenor)."""
import numpy as np
from xgboost import XGBRegressor

from module.pipeline import predict_direct_tab, predict_hybrid
from module.data import time_weights

MODELS = {
    "bench_gbm": "gbm",
    "bench_ridge": "ridge",
    "bench_lgbm": "lgbm",
    "bench_catboost": "cat",
    "bench_xgboost": "xgb",
}


def _xgb_anchor(D, cfg, tr, te):
    """stage-1 앵커: XGBoost가 무차익가 MC를 예측.
    입력을 DeepONet-Curve 앵커와 동일하게 맞춤 → 순수 아키텍처 비교.
    X = [곡선 u0-9 | vol·corr(7) | 계약(strk0-11 + B + coupon + tenor)]."""
    XP = cfg["tabular"]["xgb"]
    X = np.concatenate([D.CURVE, D.VC, D.CON], axis=1)
    w = time_weights(D, tr) if cfg["data"]["time_decay"] else None
    m = XGBRegressor(n_estimators=XP["n_estimators"], learning_rate=XP["learning_rate"],
                     max_depth=XP["max_depth"], subsample=XP["subsample"],
                     colsample_bytree=XP["colsample_bytree"], n_jobs=0,
                     random_state=cfg["seed"])
    m.fit(X[tr], D.MC[tr], sample_weight=w)
    return None, m.predict(X[te]).astype("float32")


def run(D, cfg):
    out = {name: predict_direct_tab(D, cfg, key) for name, key in MODELS.items()}
    out["xgb_hybrid"] = predict_hybrid(D, cfg, _xgb_anchor)   # XGBoost 하이브리드
    return out
