# -*- coding: utf-8 -*-
"""DeepONet (MLP branch, 물리손실 없음): 직접(FAIR) + 하이브리드(MC앵커+마진+잔차)."""
from module.train import train_operator_mlp
from module.pipeline import predict_direct_nn, predict_hybrid


def _anchor(D, cfg, tr, te):
    return train_operator_mlp(D, cfg, tr, te, target=D.MC, use_physics=False)


def run(D, cfg):
    """반환: {예측명: DataFrame}."""
    return {
        "deeponet_direct": predict_direct_nn(
            D,
            cfg,
            lambda D, cfg, tr, te: train_operator_mlp(
                D, cfg, tr, te, target=D.FAIR, use_physics=False
            ),
        ),
        "deeponet_hybrid": predict_hybrid(D, cfg, _anchor),
    }
