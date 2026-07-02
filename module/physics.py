# -*- coding: utf-8 -*-
"""물리: 만기 payoff, BS-PDE 잔차(autograd). + RBF 미분행렬 도구(보존; 기본 파이프라인 미사용).

BS-PDE (시간=잔존만기 τ):  -Vτ + ½σ²S²V_SS + rSV_S − rV = 0
worst-of autocall+KI 만기조건: S≥K → 1+coupon·tenor, 아니면 (KI면 S, 아니면 1)
"""
import numpy as np
import torch

from .data import to_tensor


# ===== MLP (OperatorMLP) =====
def terminal_payoff_mlp(m, S, I):
    # m: [sig_eff,r,B,Kfirst,K,coupon,tenor] → K=col4, coupon=col5, tenor=col6
    return torch.where(
        S >= m[:, 4], 1 + m[:, 5] * m[:, 6], torch.where(I < 0.5, torch.ones_like(S), S)
    )


def bs_residual_mlp(net, m, S, tau, I):
    S = S.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)
    V = net.V(m, S, tau, I)
    (Vt,) = torch.autograd.grad(V.sum(), tau, create_graph=True)
    (Vs,) = torch.autograd.grad(V.sum(), S, create_graph=True)
    (Vss,) = torch.autograd.grad(Vs.sum(), S, create_graph=True)
    return -Vt + 0.5 * m[:, 0] ** 2 * S**2 * Vss + m[:, 1] * S * Vs - m[:, 1] * V


# ===== Curve (CurveOperator) =====
def terminal_payoff_curve(cond, S, I, ten):
    # cond: [sig_eff,r,B,Kfirst,K,coupon] → K=col4, coupon=col5
    return torch.where(
        S >= cond[:, 4],
        1 + cond[:, 5] * ten,
        torch.where(I < 0.5, torch.ones_like(S), S),
    )


def bs_residual_curve(net, u, cond, cats, S, tau, ten, I):
    S = S.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)
    V = net.V(u, cond, cats, S, tau, ten, I)
    (Vt,) = torch.autograd.grad(V.sum(), tau, create_graph=True)
    (Vs,) = torch.autograd.grad(V.sum(), S, create_graph=True)
    (Vss,) = torch.autograd.grad(Vs.sum(), S, create_graph=True)
    return (
        -Vt + 0.5 * cond[:, 0] ** 2 * S**2 * Vss + cond[:, 1] * S * Vs - cond[:, 1] * V
    )


# ===== RBF 미분행렬 도구 (보존; exp2_rbf_vs_autograd / rbf_pde_loss 방식) =====
def build_rbf(cfg, dev):
    """정규화 (τ~,S~) 격자에서 inverse-quadratic 커널 미분행렬 precompute."""
    p = cfg["physics"]["rbf"]
    Nt, Ns, s = p["Nt"], p["Ns"], p["s"]
    smax = float(cfg["data"]["smax"])
    tt = np.linspace(0, 1, Nt)
    ss = np.linspace(0, 1, Ns)
    TG, SG = np.meshgrid(tt, ss)
    Xn = np.stack([TG.ravel(), SG.ravel()], 1).astype("float64")
    diff = Xn[:, None, :] - Xn[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    Phi = 1.0 / (1.0 + s**2 * r2)
    Pinv = np.linalg.pinv(Phi, rcond=1e-6)
    Dc = -2.0 * s**2 / (1.0 + s**2 * r2) ** 2
    G0 = (Dc * diff[:, :, 0]) @ Pinv
    G1 = (Dc * diff[:, :, 1]) @ Pinv
    DSS = G1 @ G1
    Sn = SG.ravel()
    tn = TG.ravel()
    interior = np.where((tn > 1e-9) & (Sn > 1e-9) & (Sn < 1 - 1e-9))[0]
    return dict(
        Sg=to_tensor(Sn, dev),
        tg=to_tensor(tn, dev),
        G0=to_tensor(G0.T.copy(), dev),
        G1=to_tensor(G1.T.copy(), dev),
        DSS=to_tensor(DSS.T.copy(), dev),
        Sphys=to_tensor(Sn * smax, dev),
        interior=torch.tensor(interior, device=dev),
        M=len(Sn),
        nel=p["nel"],
        smax=smax,
    )


def rbf_grid_input(RBF, I):
    n, M = I.shape[0], RBF["M"]
    return torch.stack(
        [
            RBF["Sg"].unsqueeze(0).expand(n, M),
            RBF["tg"].unsqueeze(0).expand(n, M),
            I.unsqueeze(1).expand(n, M),
        ],
        -1,
    )


def rbf_residual(RBF, U, sig, r, ten):
    smax = RBF["smax"]
    Dt = (U @ RBF["G0"]) / ten[:, None]
    DS = (U @ RBF["G1"]) / smax
    DSS = (U @ RBF["DSS"]) / smax**2
    res = (
        -Dt
        + 0.5 * sig[:, None] ** 2 * (RBF["Sphys"] ** 2)[None, :] * DSS
        + r[:, None] * RBF["Sphys"][None, :] * DS
        - r[:, None] * U
    )
    return (res[:, RBF["interior"]] ** 2).mean()
