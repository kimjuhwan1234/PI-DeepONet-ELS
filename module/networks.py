# -*- coding: utf-8 -*-
"""신경망 아키텍처: MLP 헬퍼, OperatorMLP(DeepONet MLP), CurveBranch(CNN+FiLM),
DONdirect(곡선 직접), CurveOperator(곡선 연산자·trunk S,τ,I)."""
import torch
import torch.nn as nn

from .data import COND


def mlp(d, p=128):
    return nn.Sequential(
        nn.Linear(d, p), nn.Tanh(), nn.Linear(p, p), nn.Tanh(), nn.Linear(p, p)
    )


class OperatorMLP(nn.Module):
    """DeepONet(MLP): branch=params(7), trunk=(S/SMAX, τ/tenor, I). 내적."""

    def __init__(self, lo, hi, smax, width=128, dev="cpu"):
        super().__init__()
        self.b = mlp(7, width)
        self.t = mlp(3, width)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.lo = torch.tensor(lo, dtype=torch.float32, device=dev)
        self.hi = torch.tensor(hi - lo, dtype=torch.float32, device=dev) + 1e-9
        self.smax = smax

    def V(self, m, S, tau, I):
        ten = m[:, 6]
        trunk = self.t(torch.stack([S / self.smax, tau / ten, I], -1))
        return (self.b((m - self.lo) / self.hi) * trunk).sum(-1) + self.b0


class CurveBranch(nn.Module):
    """1D-CNN(수익률곡선) + 범주형 임베딩 + FiLM 융합 → P차원 임베딩."""

    def __init__(self, card, ncond, P):
        super().__init__()
        self.P = P
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cfc = nn.Linear(32, P)
        self.emb = (
            nn.ModuleList(
                [nn.Embedding(c, 8 if i == 0 else 4) for i, c in enumerate(card)]
            )
            if card
            else None
        )
        din = ncond + (sum(8 if i == 0 else 4 for i in range(len(card))) if card else 0)
        self.els = nn.Sequential(nn.Linear(din, 128), nn.Tanh(), nn.Linear(128, P))
        self.film = nn.Linear(P, 2 * P)
        self.out = nn.Sequential(nn.Linear(2 * P, P), nn.Tanh(), nn.Linear(P, P))

    def forward(self, u, cond, cats=None):
        hc = self.cfc(self.cnn(u.unsqueeze(1)).squeeze(-1))
        parts = [cond] + (
            [e(cats[:, i]) for i, e in enumerate(self.emb)]
            if self.emb is not None
            else []
        )
        el = self.els(torch.cat(parts, -1))
        gb = self.film(el)
        g, b = gb[:, : self.P], gb[:, self.P :]
        return self.out(torch.cat([g * hc + b, el], -1))


class DONdirect(nn.Module):
    """DeepONet-Curve 직접학습(FAIR): branch=CurveBranch(DPHYS+DNPN), trunk=(S,τ)."""

    def __init__(self, card, ncond, P, smax):
        super().__init__()
        self.br = CurveBranch(card, ncond, P)
        self.tr = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, P), nn.Tanh(), nn.Linear(P, P)
        )
        self.b0 = nn.Parameter(torch.zeros(1))
        self.smax = smax

    def V(self, u, ph, npn, cats, S, tau, ten):
        cond = torch.cat([ph, npn], -1)
        return (
            self.br(u, cond, cats)
            * self.tr(torch.stack([S / self.smax, tau / ten], -1))
        ).sum(-1) + self.b0


class CurveOperator(nn.Module):
    """DeepONet-Curve 연산자: branch=CurveBranch(COND), trunk=(S,τ,I). 하이브리드 MC앵커/물리."""

    def __init__(self, card, cm, cs, P, smax, dev="cpu"):
        super().__init__()
        self.br = CurveBranch(card, len(COND), P)
        self.tr = mlp(3, P)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.cm = torch.tensor(cm, dtype=torch.float32, device=dev)
        self.cs = torch.tensor(cs, dtype=torch.float32, device=dev)
        self.smax = smax

    def V(self, u, cond, cats, S, tau, ten, I):
        cn = (cond - self.cm) / self.cs
        trunk = self.tr(torch.stack([S / self.smax, tau / ten, I], -1))
        return (self.br(u, cn, cats) * trunk).sum(-1) + self.b0
