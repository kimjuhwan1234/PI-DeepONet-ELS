# -*- coding: utf-8 -*-
"""DeepONet-Curve (정통 재설계, 데이터기반): branch=1D-CNN(수익률곡선)+바스켓 vol·corr, trunk=계약.
 직접(FAIR 회귀) + 하이브리드(MC 앵커 + recent_margin + 잔차) 두 모드."""
from module.train import train_curve
from module.pipeline import predict_direct_nn, predict_hybrid


def _anchor(D, cfg, tr, te):
    return train_curve(D, cfg, tr, te, target=D.MC)


def run(D, cfg):
    return {
        "deeponet_direct": predict_direct_nn(
            D, cfg, lambda D, cfg, tr, te: train_curve(D, cfg, tr, te, target=D.FAIR)),
        "deeponet_hybrid": predict_hybrid(D, cfg, _anchor),
    }
