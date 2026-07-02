# -*- coding: utf-8 -*-
"""예측(추론) 속도 측정 — 학습된 모델로 test 폴드를 예측하는 데 걸리는 시간만 측정.
 학습 시간은 제외(즉시 견적 지연시간 = 추론 지연). fold-0 기준, 반복 median.
 하이브리드는 최종 fair 산출 경로 전체(앵커 forward + recent_margin + 마진모델 predict)를 잰다."""
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from module.data import time_weights
from module.train import train_curve, train_margin_don
from module.tabular import fit_tab


def _time(fn, repeat=20, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure(D, cfg, repeat=20):
    """반환: index=model, cols=[n, batch_ms, per_product_us, products_per_sec, kind]."""
    tr, te = D.WF[0]
    N = len(te)
    rm = D.rm
    resid = (D.FAIR - D.MC - rm).astype("float32")
    tw = cfg["data"]["time_decay"]
    rows = []

    def add(name, kind, predict_fn):
        sec = _time(lambda: predict_fn(te), repeat=repeat)
        rows.append(dict(model=name, kind=kind, n=N, batch_ms=sec * 1e3,
                         per_product_us=sec / N * 1e6, products_per_sec=N / sec))

    # 직접 tabular 벤치마크 (regime 피처, FAIR)
    for name, key in [("bench_ridge", "ridge"), ("bench_xgboost", "xgb"),
                      ("bench_catboost", "cat"), ("bench_lgbm", "lgbm"), ("bench_gbm", "gbm")]:
        _, pr = fit_tab(D, cfg, key, tr, te, "regime", D.FAIR, tw=tw, return_predictor=True)
        add(name, "direct", pr)

    # DeepONet-Curve 직접 (FAIR)
    *_, p_dir = train_curve(D, cfg, tr, te, target=D.FAIR, return_predict=True)
    add("deeponet_direct", "direct", p_dir)

    # 공통 stage-2 마진모델 (두 하이브리드 동일)
    _, p_margin = fit_tab(D, cfg, cfg["margin"]["model"], tr, te, cfg["margin"]["feature_set"],
                          resid, tw=tw, return_predictor=True)

    # DeepONet 하이브리드: 연산자망 MC 앵커 forward + rm + 마진(XGB)
    *_, p_don_anchor = train_curve(D, cfg, tr, te, target=D.MC, return_predict=True)
    add("deeponet_hybrid", "hybrid", lambda idx: p_don_anchor(idx) + rm[idx] + p_margin(idx))

    # DeepONet 하이브리드(stage-2도 DeepONet): 같은 MC 앵커 forward + rm + 마진(DeepONet)
    *_, p_s2don = train_margin_don(D, cfg, tr, te, resid, return_predict=True)
    add("deeponet_hybrid_s2don", "hybrid", lambda idx: p_don_anchor(idx) + rm[idx] + p_s2don(idx))

    # XGB 하이브리드: matched-입력 XGB MC 앵커 + rm + 마진
    XP = cfg["tabular"]["xgb"]
    X = np.concatenate([D.CURVE, D.VC, D.CON], axis=1)
    w = time_weights(D, tr) if tw else None
    ma = XGBRegressor(n_estimators=XP["n_estimators"], learning_rate=XP["learning_rate"],
                      max_depth=XP["max_depth"], subsample=XP["subsample"],
                      colsample_bytree=XP["colsample_bytree"], n_jobs=0, random_state=cfg["seed"])
    ma.fit(X[tr], D.MC[tr], sample_weight=w)
    add("xgb_hybrid", "hybrid", lambda idx: ma.predict(X[idx]) + rm[idx] + p_margin(idx))

    return pd.DataFrame(rows).set_index("model").sort_values("per_product_us")
