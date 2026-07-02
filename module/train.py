# -*- coding: utf-8 -*-
"""연산자망 학습.
 train_pi    : PI-DeepONet (branch=vol·corr+곡선, trunk=계약+S,τ,I) + BS-PDE 물리. MC 앵커.
 train_curve : DeepONet-Curve (CNN 곡선+vol·corr, trunk=계약). 데이터기반. target=MC(하이브리드) 또는 FAIR(직접).
"""
import numpy as np
import torch

from .data import to_tensor, zstats, VOLCORR, UC, CONTRACT
from .networks import PIOperator, CurveOperatorV2


def _rf(k, a, b, dev):
    return torch.rand(k, device=dev) * (b - a) + a


def _opt(net, cfg):
    return torch.optim.Adam(net.parameters(), cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])


def _branch_mat(D):
    """branch 원본 행렬 [vol·corr | 곡선]."""
    return np.concatenate([D.VC, D.CURVE], 1)


# ===== PI-DeepONet (물리, MC 앵커) =====
def train_pi(D, cfg, tr, te):
    dev = D.DEV; B = cfg["train"]["batch"]; NIT = cfg["train"]["nit"]; P = cfg["networks"]["P"]
    wt = cfg["physics"]["terminal_weight"]; wp = cfg["physics"]["pde_weight"]
    torch.manual_seed(cfg["seed"])
    Bmat = _branch_mat(D); Con = D.CON
    bm, bs = zstats(Bmat, tr); cm, cs = zstats(Con, tr)
    Bn = to_tensor((Bmat - bm) / bs, dev)          # 정규화 branch
    Cn = to_tensor((Con - cm) / cs, dev)           # 정규화 contract
    Conr = to_tensor(Con, dev)                     # 원본 contract (payoff/PDE용)
    ten = to_tensor(D.TEN, dev); sig = to_tensor(D.SIGEFF, dev); r = to_tensor(D.R, dev)
    iK, iC, iT, iBar = D.iKlast, D.iCOUPON, D.iTEN, D.iBARR
    net = PIOperator(Bn.shape[1], Cn.shape[1], P, D.SMAX).to(dev); opt = _opt(net, cfg)
    ym, ysd = float(D.MC[tr].mean()), float(D.MC[tr].std()); Y = to_tensor((D.MC - ym) / ysd, dev)
    ntr = len(tr); trt = torch.tensor(tr, device=dev)

    def payoff(cr, S, I):   # worst-of 만기: K=만기행사가(strk_last), coupon, ten, barrier(KI)
        K = cr[:, iK]; cpn = cr[:, iC]; tn = cr[:, iT]
        return torch.where(S >= K, 1 + cpn * tn, torch.where(I < 0.5, torch.ones_like(S), S))

    for _ in range(NIT):
        bb = trt[torch.randint(0, ntr, (B,), device=dev)]
        bx, cx, cr, tn = Bn[bb], Cn[bb], Conr[bb], ten[bb]
        # 데이터손실: 발행시점 (S=1, τ=tenor, I=0)
        l = ((net.V(bx, cx, torch.ones(B, device=dev), tn, tn, torch.zeros(B, device=dev)) - Y[bb]) ** 2).mean()
        # 만기조건
        S = _rf(B, 0, D.SMAX, dev)
        for Iv in (0.0, 1.0):
            Ii = torch.full((B,), Iv, device=dev)
            l = l + wt * ((net.V(bx, cx, S, torch.zeros(B, device=dev), tn, Ii) - (payoff(cr, S, Ii) - ym) / ysd) ** 2).mean()
        # BS-PDE 잔차 (σ=sig_eff, r): trunk 좌표 S,τ에 대해 미분
        Sr = _rf(B, 0, D.SMAX, dev).clone().requires_grad_(True)
        tr_ = (_rf(B, 0, 1, dev) * tn).clone().requires_grad_(True)
        Ii = (torch.rand(B, device=dev) > 0.5).float()
        V = net.V(bx, cx, Sr, tr_, tn, Ii)
        Vt, = torch.autograd.grad(V.sum(), tr_, create_graph=True)
        Vs, = torch.autograd.grad(V.sum(), Sr, create_graph=True)
        Vss, = torch.autograd.grad(Vs.sum(), Sr, create_graph=True)
        res = -Vt + 0.5 * sig[bb] ** 2 * Sr ** 2 * Vss + r[bb] * Sr * Vs - r[bb] * V
        l = l + wp * (res ** 2).mean()
        opt.zero_grad(); l.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        def p(idx):
            i = torch.tensor(idx, device=dev)
            return (net.V(Bn[i], Cn[i], torch.ones(len(idx), device=dev), ten[i], ten[i],
                          torch.zeros(len(idx), device=dev)) * ysd + ym).cpu().numpy()
        return p(tr), p(te)


# ===== DeepONet-Curve (데이터기반) =====
def train_curve(D, cfg, tr, te, target):
    """target=D.MC → 하이브리드 앵커 / target=D.FAIR → 직접."""
    dev = D.DEV; B = cfg["train"]["batch"]; NIT = cfg["train"]["nit"]; P = cfg["networks"]["P"]
    torch.manual_seed(cfg["seed"])
    um, us = zstats(D.CURVE, tr); vm, vs = zstats(D.VC, tr); cm, cs = zstats(D.CON, tr)
    Un = to_tensor((D.CURVE - um) / us, dev)
    Vn = to_tensor((D.VC - vm) / vs, dev)
    Cn = to_tensor((D.CON - cm) / cs, dev)
    net = CurveOperatorV2(Vn.shape[1], Cn.shape[1], P).to(dev); opt = _opt(net, cfg)
    ym, ysd = float(target[tr].mean()), float(target[tr].std() + 1e-8); Y = to_tensor((target - ym) / ysd, dev)
    ntr = len(tr); trt = torch.tensor(tr, device=dev)
    for _ in range(NIT):
        bb = trt[torch.randint(0, ntr, (B,), device=dev)]
        pr = net.V(Un[bb], Vn[bb], Cn[bb])
        opt.zero_grad(); ((pr - Y[bb]) ** 2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        def p(idx):
            i = torch.tensor(idx, device=dev)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()
        return p(tr), p(te)
