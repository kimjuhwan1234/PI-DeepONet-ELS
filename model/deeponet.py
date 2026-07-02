# -*- coding: utf-8 -*-
"""DeepONet-Curve (정통 재설계, 데이터기반): branch=1D-CNN(수익률곡선)+바스켓 vol·corr, trunk=계약.
 직접(FAIR 회귀) + 하이브리드(MC 앵커 + recent_margin + 잔차) 두 모드.
 하이브리드 stage-2는 XGB(deeponet_hybrid) 또는 DeepONet(deeponet_hybrid_s2don) 두 종류."""
from module.train import train_curve, train_margin_don
from module.pipeline import predict_direct_nn, predict_hybrid


def _anchor(D, cfg, tr, te):
    return train_curve(D, cfg, tr, te, target=D.MC)


def run(D, cfg):
    return {
        "deeponet_direct": predict_direct_nn(
            D, cfg, lambda D, cfg, tr, te: train_curve(D, cfg, tr, te, target=D.FAIR)),
        "deeponet_hybrid": predict_hybrid(D, cfg, _anchor),
        # stage-1 동일(DeepONet MC 앵커), stage-2만 DeepONet(branch=리스크+금리, trunk=나머지)
        "deeponet_hybrid_s2don": predict_hybrid(D, cfg, _anchor, resid_fn=train_margin_don),
    }
