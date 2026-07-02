# -*- coding: utf-8 -*-
"""평가 지표: R2/MAE/RMSE/Spearman + 하이브리드 stage별 R2."""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def stage_r2(df):
    """하이브리드 예측 DataFrame(mc_true/mc_pred/resid_true/resid_pred/y_true/y_pred)에서 stage별 R2."""
    need = {"mc_true", "mc_pred", "resid_true", "resid_pred", "y_true", "y_pred"}
    if not need.issubset(df.columns):
        return None
    return {
        "stage1_MC_R2": r2_score(df["mc_true"], df["mc_pred"]),
        "stage2_resid_R2": r2_score(df["resid_true"], df["resid_pred"]),
        "final_FAIR_R2": r2_score(df["y_true"], df["y_pred"]),
    }
