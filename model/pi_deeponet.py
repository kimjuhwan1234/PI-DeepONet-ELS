# -*- coding: utf-8 -*-
"""PI-DeepONet (정통 재설계): branch=바스켓 vol·corr+수익률곡선, trunk=계약+평가좌표(S,τ,I), BS-PDE 물리.
 물리가 무차익가 MC를 규제하므로 MC 앵커 → 하이브리드로만 사용."""
from module.train import train_pi
from module.pipeline import predict_hybrid


def _anchor(D, cfg, tr, te):
    return train_pi(D, cfg, tr, te)


def run(D, cfg):
    return {"pi_deeponet_hybrid": predict_hybrid(D, cfg, _anchor)}
