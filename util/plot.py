# -*- coding: utf-8 -*-
"""그래프 (전부 영문 텍스트). 각 함수는 파일로 저장하고 fig 를 반환."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

plt.rcParams["axes.unicode_minus"] = False


def model_color(name):
    n = name.lower()
    if "pi_deeponet" in n or "pi-deeponet" in n:
        return "#4a8"  # green = PI-DeepONet
    if "curve" in n:
        return "#a5a"  # purple = curve
    if "deeponet" in n:
        return "#59c"  # blue = DeepONet(MLP)
    return "#e77"  # red = tabular benchmark


def metric_bars(metrics_df, path=None, title="Benchmark - walk-forward OOS"):
    """metrics_df: index=model, columns=[R2,MAE,RMSE,Spearman]."""
    specs = [
        ("R2", "test R2 (higher=better)", False),
        ("MAE", "test MAE (lower=better)", True),
        ("RMSE", "test RMSE (lower=better)", True),
        ("Spearman", "Spearman rho (higher=better)", False),
    ]
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    for a, (col, tt, asc) in zip(ax.ravel(), specs):
        d = metrics_df.sort_values(col, ascending=asc)
        names = list(d.index)
        a.barh(
            names[::-1],
            d[col].values[::-1],
            color=[model_color(x) for x in names][::-1],
        )
        a.axvline(0, color="k", lw=0.5)
        a.set_title(tt)
        a.tick_params(labelsize=7)
        for i, v in enumerate(d[col].values[::-1]):
            a.text(v, i, f"{v:.3f}", va="center", fontsize=6.5)
    fig.suptitle(
        title + "  (green=PI-DeepONet, blue=DeepONet, purple=Curve, red=benchmark)",
        fontsize=10,
    )
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig


def scatter_grid(preds, order=None, path=None, ncol=4):
    """preds: dict[name -> (y_true, y_pred)]."""
    names = order or list(preds.keys())
    nrow = int(np.ceil(len(names) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 4.6, nrow * 4.6))
    axl = np.atleast_1d(ax).ravel()
    for a, nm in zip(axl, names):
        yt, yp = preds[nm]
        a.scatter(yt, yp, s=5, alpha=0.3, color=model_color(nm))
        lim = [float(np.min(yt)), float(np.max(yt))]
        a.plot(lim, lim, "r--", lw=1)
        a.set_title(
            f"{nm}\nR2={r2_score(yt, yp):+.3f}, rho={spearmanr(yt, yp).correlation:+.3f}",
            fontsize=8,
        )
        a.set_xlabel("actual FAIR")
        a.set_ylabel("pred")
    for a in axl[len(names) :]:
        a.axis("off")
    fig.suptitle("Predicted vs actual FAIR_VALUE - walk-forward OOS")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig


def speed_bar(speed_df, path=None):
    """speed_df: index=model, cols include per_product_us, products_per_sec. 추론 지연시간(로그축)."""
    d = speed_df.sort_values("per_product_us")
    names = list(d.index)
    vals = d["per_product_us"].values
    fig, ax = plt.subplots(figsize=(11, max(3.2, 0.5 * len(names) + 1.4)))
    ax.barh(names[::-1], vals[::-1], color=[model_color(x) for x in names][::-1])
    ax.set_xscale("log")
    ax.set_xlabel("inference latency per product (microseconds, log scale)  -  lower = faster")
    ax.set_title("Prediction speed - walk-forward fold-0 test (train time excluded)")
    ax.tick_params(labelsize=8)
    for i, (v, pps) in enumerate(zip(vals[::-1], d["products_per_sec"].values[::-1])):
        ax.text(v, i, f"  {v:.1f}us  ({pps:,.0f}/s)", va="center", fontsize=7.5)
    ax.margins(x=0.18)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig


def stage_r2_bar(stage_df, path=None):
    """stage_df: index=model, columns=[stage1_MC_R2, stage2_resid_R2, final_FAIR_R2]."""
    labels = list(stage_df.index)
    x = np.arange(3)
    w = 0.8 / max(len(labels), 1)
    palette = ["#4a8", "#59c", "#a5a", "#d46a1f", "#e8923d", "#888"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for j, nm in enumerate(labels):
        ax.bar(
            x + (j - (len(labels) - 1) / 2) * w,
            stage_df.loc[nm].values,
            w,
            label=nm,
            color=palette[j % len(palette)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Stage-1 (MC reproduce)", "Stage-2 (resid margin)", "Final (FAIR)"]
    )
    ax.set_ylabel("R2")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Per-stage R2 (walk-forward OOS)")
    for j, nm in enumerate(labels):
        for i, v in enumerate(stage_df.loc[nm].values):
            ax.text(
                x[i] + (j - (len(labels) - 1) / 2) * w,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=6.5,
                rotation=90,
            )
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig
