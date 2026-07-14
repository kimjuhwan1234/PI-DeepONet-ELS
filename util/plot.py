# -*- coding: utf-8 -*-
"""그래프 (전부 영문 텍스트). 학술/PPT 용: dpi 400·투명배경·16:9 슬라이드 크기.
 각 함수는 파일로 저장하고 fig 를 반환.

 *** 기본값(디폴트) ***: 이 모듈을 import 하는 순간 matplotlib 전역 rcParams 가 학술/PPT 사양으로 설정된다.
 즉 `from util import plot` 후에는 어떤 `fig.savefig(path)` 든 별도 인자 없이 자동으로
 **dpi 400 · 투명배경 · bbox tight** 로 저장되고, 새 figure 기본 크기도 PPT 콘텐츠 밴드(13.333 x 5.0in).
 (개별 인자를 주면 그것이 우선.)"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

SLIDE = (13.333, 7.5)      # PowerPoint 16:9 widescreen (inches)
SLIDE_W = 13.333           # 슬라이드 폭 (측면 여백 포함 full)
BAND_H = 5.2               # 제목바 + 하단 footnote 를 뺀 콘텐츠 영역 높이 (그림이 밴드에 딱 맞게)
DPI = 400

# ===== 학술 스타일 + PPT 저장 기본값 (import 시 전역 적용) =====
plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "figure.figsize": (SLIDE_W, 5.0),   # 기본 figure 크기 = PPT 콘텐츠 밴드 (미지정 시)
    # --- 저장 기본값: dpi 400 · 투명 · tight (savefig 인자 없이도 PPT-ready) ---
    "savefig.dpi": DPI,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "legend.frameon": False,
})


def _save(fig, path, dpi=DPI):
    """투명배경·고해상도 저장 (PPT 오버레이용)."""
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)


def model_color(name):
    n = name.lower()
    if "mape" in n:
        return "#1f6f9c"   # deep blue = DeepONet (MAPE loss)
    if "deeponet" in n:
        return "#5aa9dd"   # blue = DeepONet hybrid
    return "#e07a5f"       # terracotta = tabular ML benchmark (직접/XGB 하이브리드)


def _sort_best_first(df, col, mode):
    """mode: 'higher'(클수록 좋음) | 'lower'(작을수록) | 'abs'(0에 가까울수록)."""
    if mode == "higher":
        return df.sort_values(col, ascending=False)
    if mode == "lower":
        return df.sort_values(col, ascending=True)
    return df.reindex(df[col].abs().sort_values().index)   # abs


def metric_bars(metrics_df, path=None, title="Walk-forward OOS benchmark"):
    """metrics_df: index=model, columns⊇[R2,MAPE%,MAE,RMSE,Bias%,Spearman]. 2x3 학술 패널."""
    specs = [
        ("R2", "R²  (higher better)", "higher", "{:.3f}"),
        ("MAPE%", "MAPE %  (lower better)", "lower", "{:.2f}"),
        ("Spearman", "Spearman ρ  (higher better)", "higher", "{:.3f}"),
        ("MAE", "MAE  (lower better)", "lower", "{:.4f}"),
        ("RMSE", "RMSE  (lower better)", "lower", "{:.4f}"),
        ("Bias%", "Bias %  (→ 0 better)", "abs", "{:+.2f}"),
    ]
    specs = [s for s in specs if s[0] in metrics_df.columns]
    fig, ax = plt.subplots(2, 3, figsize=(SLIDE_W, BAND_H))
    for a, (col, tt, mode, fmt) in zip(ax.ravel(), specs):
        d = _sort_best_first(metrics_df, col, mode)
        names = list(d.index); vals = d[col].values
        y = np.arange(len(names))
        a.barh(y, vals, color=[model_color(x) for x in names], edgecolor="white", linewidth=0.5)
        a.set_yticks(y); a.set_yticklabels(names)
        a.invert_yaxis()                       # best 맨 위
        a.axvline(0, color="#444", lw=0.6)
        a.set_title(tt)
        a.grid(axis="x", color="#cccccc", lw=0.6, alpha=0.6)
        a.tick_params(labelsize=8.5)
        xr = (vals.max() - min(vals.min(), 0)) or 1
        for i, v in enumerate(vals):
            a.text(v + (0.01 * xr if v >= 0 else -0.01 * xr), i, fmt.format(v),
                   va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
        a.margins(x=0.16)
    for a in ax.ravel()[len(specs):]:
        a.axis("off")
    fig.suptitle(title + "   (blue=DeepONet hybrid · orange=ML benchmark)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)
    return fig


def scatter_grid(preds, order=None, path=None, ncol=6):
    """preds: dict[name -> (y_true, y_pred)]. 예측 vs 실제 산점도 그리드 (콘텐츠 밴드에 맞게 2행 6열)."""
    names = order or list(preds.keys())
    nrow = int(np.ceil(len(names) / ncol))
    h = min(BAND_H, (SLIDE_W / ncol) * nrow * 1.22)
    fig, ax = plt.subplots(nrow, ncol, figsize=(SLIDE_W, h))
    axl = np.atleast_1d(ax).ravel()
    for a, nm in zip(axl, names):
        yt, yp = preds[nm]
        a.scatter(yt, yp, s=4, alpha=0.28, color=model_color(nm), edgecolors="none")
        lim = [float(np.min(yt)), float(np.max(yt))]
        a.plot(lim, lim, "--", color="#c0392b", lw=1.0)
        a.set_title(f"{nm}\nR²={r2_score(yt, yp):+.3f} ρ={spearmanr(yt, yp).correlation:+.3f}",
                    fontsize=7.5)
        a.set_xlabel("actual", fontsize=8); a.set_ylabel("predicted", fontsize=8)
        a.grid(color="#dddddd", lw=0.5, alpha=0.6)
        a.tick_params(labelsize=7)
    for a in axl[len(names):]:
        a.axis("off")
    fig.suptitle("Predicted vs actual (target) — walk-forward OOS", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, path)
    return fig


def speed_bar(speed_df, path=None):
    """speed_df: index=model, cols include per_product_us, products_per_sec. 추론 지연(로그축)."""
    d = _sort_best_first(speed_df, "per_product_us", "lower")
    names = list(d.index); vals = d["per_product_us"].values
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(SLIDE_W, min(BAND_H, 0.5 * len(names) + 1.5)))
    ax.barh(y, vals, color=[model_color(x) for x in names], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("inference latency per product (µs, log scale) — lower = faster")
    ax.set_title("Prediction speed — walk-forward fold-0 test (training time excluded)")
    ax.grid(axis="x", which="both", color="#cccccc", lw=0.6, alpha=0.5)
    for i, (v, pps) in enumerate(zip(vals, d["products_per_sec"].values)):
        ax.text(v * 1.08, i, f"{v:.1f} µs  ({pps:,.0f}/s)", va="center", fontsize=8.5)
    ax.margins(x=0.22)
    fig.tight_layout()
    _save(fig, path)
    return fig


# 모델 구분용 정성 팔레트 (grouped bar 에서 시리즈 식별)
_QUAL = ["#1f6f9c", "#5aa9dd", "#2a9d8f", "#e07a5f", "#9b5de5", "#e9c46a", "#c0637a", "#7f8c8d"]


def stage_r2_bar(stage_df, path=None):
    """stage_df: index=model, columns=[stage1_MC_R2, stage2_resid_R2, final_MC_R2] (이론가 2단계)."""
    labels = list(stage_df.index)
    x = np.arange(3)
    w = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(SLIDE_W, BAND_H))
    for j, nm in enumerate(labels):
        off = (j - (len(labels) - 1) / 2) * w
        ax.bar(x + off, stage_df.loc[nm].values, w, label=nm,
               color=_QUAL[j % len(_QUAL)], edgecolor="white", linewidth=0.4)
        for i, v in enumerate(stage_df.loc[nm].values):
            ax.text(x[i] + off, v + (0.012 if v >= 0 else -0.012), f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["Stage-1\n(anchor MC_hat)", "Stage-2\n(residual MC−MC_hat)", "Final\n(theoretical MC)"])
    ax.set_ylabel("R²")
    ax.axhline(0, color="#444", lw=0.6)
    ax.grid(axis="y", color="#cccccc", lw=0.6, alpha=0.5)
    ax.set_title("Per-stage R²  (walk-forward OOS)")
    ax.legend(fontsize=9.5, loc="center left", bbox_to_anchor=(1.01, 0.5))   # 우측 범례 = 제목과 겹침 방지
    fig.tight_layout()
    _save(fig, path)
    return fig
