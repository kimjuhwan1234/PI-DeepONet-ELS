# -*- coding: utf-8 -*-
"""정통 DeepONet 재설계.
 PIOperator      : branch=MLP(vol·corr+곡선), trunk=MLP(계약+평가좌표 S,τ,I). BS-PDE 물리.
 CurveOperatorV2 : branch=CNN(곡선)+vol·corr, trunk=MLP(계약). 데이터기반(물리 없음).
 두 모델 모두 branch=시장상태, trunk=계약 → 내적으로 가격."""
import torch
import torch.nn as nn


def mlp(d, p=128):
    return nn.Sequential(nn.Linear(d, p), nn.Tanh(), nn.Linear(p, p), nn.Tanh(), nn.Linear(p, p))


class PIOperator(nn.Module):
    """PI-DeepONet: branch=vol·corr+곡선(정규화), trunk=계약(정규화)+[S/SMAX, τ/tenor, I]."""
    def __init__(self, nb, ncon, P, smax):
        super().__init__()
        self.b = mlp(nb, P)          # branch: 시장상태(vol·corr + 곡선)
        self.t = mlp(ncon + 3, P)    # trunk: 계약 + 평가좌표(S,τ,I)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.smax = smax

    def V(self, bx, cx, S, tau, ten, I):
        coord = torch.stack([S / self.smax, tau / ten, I], -1)
        trunk = self.t(torch.cat([cx, coord], -1))
        return (self.b(bx) * trunk).sum(-1) + self.b0


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
