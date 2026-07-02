# -*- coding: utf-8 -*-
"""tabular 회귀기 (직접 벤치마크 + 하이브리드 stage-2 마진모델 공용). config 파라미터 사용."""
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipe
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from .data import featnum, CAT, time_weights


def fit_tab(D, cfg, model, tr, te, feat, target, tw=True, return_predictor=False):
    """train 에만 fit, 시간가중. target: 전체 길이 배열, feat: 'base'|'regime'.
    return_predictor=True 이면 (preds, predictor) 반환 — predictor(idx)는 D.ml 행 idx의 예측(추론 속도 측정용)."""
    num = featnum(feat)
    X = D.ml[num + CAT]
    Xtr = X.iloc[tr]
    ytr = target[tr]
    w = time_weights(D, tr) if tw else None
    P = cfg["tabular"]

    if model == "ridge":
        ct = ColumnTransformer(
            [
                (
                    "num",
                    SkPipe(
                        [
                            ("i", SimpleImputer(strategy="median")),
                            ("s", StandardScaler()),
                        ]
                    ),
                    num,
                ),
                ("c", OneHotEncoder(handle_unknown="ignore"), CAT),
            ]
        ).fit(Xtr)
        m = Ridge(alpha=P["ridge"]["alpha"]).fit(
            ct.transform(Xtr), ytr, sample_weight=w
        )
        predictor = lambda idx: m.predict(ct.transform(X.iloc[idx]))
    elif model == "gbm":
        m = HistGradientBoostingRegressor(
            categorical_features=CAT, random_state=0, **P["gbm"]
        )
        m.fit(Xtr, ytr, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    elif model == "lgbm":
        m = lgb.LGBMRegressor(random_state=0, verbose=-1, **P["lgbm"])
        m.fit(Xtr, ytr, categorical_feature=CAT, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    elif model == "cat":
        m = CatBoostRegressor(random_seed=0, verbose=0, cat_features=CAT, **P["cat"])
        m.fit(Xtr, ytr, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    elif model == "xgb":
        m = xgb.XGBRegressor(
            tree_method="hist", enable_categorical=True, random_state=0, **P["xgb"]
        )
        m.fit(Xtr, ytr, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    else:
        raise ValueError(model)
    preds = predictor(te)
    return (preds, predictor) if return_predictor else preds
