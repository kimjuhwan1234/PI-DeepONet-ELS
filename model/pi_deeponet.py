# -*- coding: utf-8 -*-
"""PI-DeepONet (MLP branch + BS-PDE 물리손실, autograd): 하이브리드(MC앵커+마진+잔차).

물리(무차익 조건)는 마진 포함 FAIR가 아니라 무차익가 MC를 기술하므로 하이브리드만 둔다.
"""
from module.train import train_operator_mlp
from module.pipeline import predict_hybrid


def _anchor(D, cfg, tr, te):
    return train_operator_mlp(D, cfg, tr, te, target=D.MC, use_physics=True)


def run(D, cfg):
    return {"pi_deeponet_hybrid": predict_hybrid(D, cfg, _anchor)}
