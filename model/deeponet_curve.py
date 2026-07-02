# -*- coding: utf-8 -*-
"""DeepONet-Curve (1D-CNN 수익률곡선 branch + FiLM): 직접(FAIR) + 하이브리드(MC앵커+마진+잔차)."""
from module.train import train_don_direct, train_curve_anchor
from module.pipeline import predict_direct_nn, predict_hybrid


def _anchor(D, cfg, tr, te):
    return train_curve_anchor(D, cfg, tr, te, use_physics=False)


def run(D, cfg):
    return {
        "deeponet_curve_direct": predict_direct_nn(D, cfg, train_don_direct),
        "deeponet_curve_hybrid": predict_hybrid(D, cfg, _anchor),
    }
