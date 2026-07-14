# -*- coding: utf-8 -*-
"""정통 DeepONet 재설계.
 CurveOperatorV2 : branch=CNN(곡선)+vol·corr, trunk=MLP(계약). 데이터기반. stage-1 앵커.
 MarginOperator  : stage-2 잔차 DeepONet(내적). branch=시장상태, trunk=계약 → 내적으로 가격."""
import torch
import torch.nn as nn


def mlp(d, p=128):
    return nn.Sequential(nn.Linear(d, p), nn.Tanh(), nn.Linear(p, p), nn.Tanh(), nn.Linear(p, p))


class MarginOperator(nn.Module):
    """stage-2 잔차 DeepONet: branch=MLP(vol·corr·sig_eff), trunk=MLP(곡선+계약). 내적으로 잔차 회귀."""
    def __init__(self, nb, nt, P):
        super().__init__()
        self.b = mlp(nb, P)          # branch: 리스크(vol·corr) + 금리
        self.t = mlp(nt, P)          # trunk: 나머지(계약+발행+범주형 one-hot)
        self.b0 = nn.Parameter(torch.zeros(1))

    def V(self, bx, tx):
        return (self.b(bx) * self.t(tx)).sum(-1) + self.b0


class CurveOperatorV2(nn.Module):
    """DeepONet-Curve: branch=1D-CNN(곡선)+vol·corr 융합, trunk=계약. 스팟 없음."""
    def __init__(self, nvc, ncon, P):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.branch = nn.Sequential(nn.Linear(32 + nvc, 128), nn.Tanh(), nn.Linear(128, P), nn.Tanh(), nn.Linear(P, P))
        self.t = mlp(ncon, P)        # trunk: 계약
        self.b0 = nn.Parameter(torch.zeros(1))

    def V(self, curve, vc, con):
        hc = self.cnn(curve.unsqueeze(1)).squeeze(-1)          # (n, 32)
        branch = self.branch(torch.cat([hc, vc], -1))          # (n, P)  시장상태(곡선+vol·corr)
        return (branch * self.t(con)).sum(-1) + self.b0
