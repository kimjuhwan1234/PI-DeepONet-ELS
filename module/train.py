# -*- coding: utf-8 -*-
"""연산자망 학습 루프. 결과 재현을 위해 검증된 로직을 그대로 이식 (동일 seed/NIT/draw 순서)."""
import torch

from .data import to_tensor, curve_prep, don_direct_prep, DPHYS, DNPN
from .networks import OperatorMLP, DONdirect, CurveOperator
from .physics import (
    terminal_payoff_mlp,
    bs_residual_mlp,
    terminal_payoff_curve,
    bs_residual_curve,
)


def _rf(k, a, b, dev):
    return torch.rand(k, device=dev) * (b - a) + a


def _opt(net, cfg):
    return torch.optim.Adam(
        net.parameters(), cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )


# ===== DeepONet(MLP) / PI-DeepONet(MLP) : OperatorMLP =====
def train_operator_mlp(D, cfg, tr, te, target, use_physics):
    """target=D.MC → 하이브리드 MC앵커 / target=D.FAIR → 직접. use_physics=True → PI-DeepONet."""
    dev = D.DEV
    B = cfg["train"]["batch"]
    NIT = cfg["train"]["nit"]
    wt = cfg["physics"]["terminal_weight"]
    wp = cfg["physics"]["pde_weight"]
    torch.manual_seed(cfg["seed"])
    lo = D.Mv[tr].min(0)
    hi = D.Mv[tr].max(0)
    net = OperatorMLP(lo, hi, D.SMAX, cfg["networks"]["mlp_width"], dev).to(dev)
    opt = _opt(net, cfg)
    Mt = to_tensor(D.Mv, dev)
    ym, ysd = float(target[tr].mean()), float(target[tr].std())
    yt = (to_tensor(target[tr], dev) - ym) / ysd
    ntr = len(tr)
    Mtr = Mt[tr]
    for _ in range(NIT):
        b = torch.randint(0, ntr, (B,), device=dev)
        m = Mtr[b]
        l = (
            (
                net.V(m, torch.ones(B, device=dev), m[:, 6], torch.zeros(B, device=dev))
                - yt[b]
            )
            ** 2
        ).mean()
        if use_physics:
            S = _rf(B, 0, D.SMAX, dev)
            for Iv in (0.0, 1.0):
                Ii = torch.full((B,), Iv, device=dev)
                l = (
                    l
                    + wt
                    * (
                        (
                            net.V(m, S, torch.zeros(B, device=dev), Ii)
                            - (terminal_payoff_mlp(m, S, Ii) - ym) / ysd
                        )
                        ** 2
                    ).mean()
                )
            Ii = (torch.rand(B, device=dev) > 0.5).float()
            l = (
                l
                + wp
                * (
                    bs_residual_mlp(
                        net, m, _rf(B, 0, D.SMAX, dev), _rf(B, 0, 1, dev) * m[:, 6], Ii
                    )
                    ** 2
                ).mean()
            )
        opt.zero_grad()
        l.backward()
        opt.step()
    net.eval()
    with torch.no_grad():

        def p(idx):
            i = torch.tensor(idx, device=dev) if not torch.is_tensor(idx) else idx
            return (
                (
                    net.V(
                        Mt[i],
                        torch.ones(len(idx), device=dev),
                        Mt[i, 6],
                        torch.zeros(len(idx), device=dev),
                    )
                    * ysd
                    + ym
                )
                .cpu()
                .numpy()
            )

        return p(tr), p(te)


# ===== DeepONet-Curve 직접(FAIR) : DONdirect =====
def train_don_direct(D, cfg, tr, te):
    dev = D.DEV
    B = cfg["train"]["batch"]
    NIT = cfg["train"]["nit"]
    P = cfg["networks"]["P"]
    torch.manual_seed(cfg["seed"])
    Un, PHn, NPn, CATt, card = don_direct_prep(D, tr)
    TENt = to_tensor(D.TENv, dev)
    net = DONdirect(card, len(DPHYS) + len(DNPN), P, D.SMAX).to(dev)
    opt = _opt(net, cfg)
    ym, ysd = float(D.FAIR[tr].mean()), float(D.FAIR[tr].std() + 1e-8)
    ntr = len(tr)
    trt = torch.tensor(tr, device=dev)
    tet = torch.tensor(te, device=dev)
    Y = to_tensor(D.FAIR, dev)
    for _ in range(NIT):
        bb = trt[torch.randint(0, ntr, (B,), device=dev)]
        pr = net.V(
            Un[bb],
            PHn[bb],
            NPn[bb],
            CATt[bb],
            torch.ones(B, device=dev),
            TENt[bb],
            TENt[bb],
        )
        opt.zero_grad()
        (((pr - ((Y[bb]) - ym) / ysd)) ** 2).mean().backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        fair_te = (
            (
                net.V(
                    Un[tet],
                    PHn[tet],
                    NPn[tet],
                    CATt[tet],
                    torch.ones(len(te), device=dev),
                    TENt[tet],
                    TENt[tet],
                )
                * ysd
                + ym
            )
            .cpu()
            .numpy()
        )
    return None, fair_te


# ===== DeepONet-Curve 하이브리드 MC앵커 : CurveOperator (use_physics=False=DeepONet, True=PI) =====
def train_curve_anchor(D, cfg, tr, te, use_physics=False):
    dev = D.DEV
    B = cfg["train"]["batch"]
    NIT = cfg["train"]["nit"]
    P = cfg["networks"]["P"]
    wt = cfg["physics"]["terminal_weight"]
    wp = cfg["physics"]["pde_weight"]
    torch.manual_seed(cfg["seed"])
    Un, CATt, card = curve_prep(D, tr)
    Ct = to_tensor(D.Cond, dev)
    TENt = to_tensor(D.TENv, dev)
    cm, cs = D.Cond[tr].mean(0), D.Cond[tr].std(0) + 1e-8
    net = CurveOperator(card, cm, cs, P, D.SMAX, dev).to(dev)
    opt = _opt(net, cfg)
    ym, ysd = float(D.MC[tr].mean()), float(D.MC[tr].std())
    ntr = len(tr)
    trt = torch.tensor(tr, device=dev)
    tet = torch.tensor(te, device=dev)
    Y = to_tensor(D.MC, dev)
    for _ in range(NIT):
        bb = trt[torch.randint(0, ntr, (B,), device=dev)]
        u = Un[bb]
        cond = Ct[bb]
        cats = CATt[bb]
        ten = TENt[bb]
        l = (
            (
                net.V(
                    u,
                    cond,
                    cats,
                    torch.ones(B, device=dev),
                    ten,
                    ten,
                    torch.zeros(B, device=dev),
                )
                - ((Y[bb]) - ym) / ysd
            )
            ** 2
        ).mean()
        if use_physics:
            S = _rf(B, 0, D.SMAX, dev)
            for Iv in (0.0, 1.0):
                Ii = torch.full((B,), Iv, device=dev)
                tp = terminal_payoff_curve(cond, S, Ii, ten)
                l = (
                    l
                    + wt
                    * (
                        (
                            net.V(u, cond, cats, S, torch.zeros(B, device=dev), ten, Ii)
                            - (tp - ym) / ysd
                        )
                        ** 2
                    ).mean()
                )
            Ii = (torch.rand(B, device=dev) > 0.5).float()
            res = bs_residual_curve(
                net,
                u,
                cond,
                cats,
                _rf(B, 0, D.SMAX, dev),
                _rf(B, 0, 1, dev) * ten,
                ten,
                Ii,
            )
            l = l + wp * (res**2).mean()
        opt.zero_grad()
        l.backward()
        opt.step()
    net.eval()
    with torch.no_grad():

        def p(idx):
            i = torch.tensor(idx, device=dev) if not torch.is_tensor(idx) else idx
            return (
                (
                    net.V(
                        Un[i],
                        Ct[i],
                        CATt[i],
                        torch.ones(len(idx), device=dev),
                        TENt[i],
                        TENt[i],
                        torch.zeros(len(idx), device=dev),
                    )
                    * ysd
                    + ym
                )
                .cpu()
                .numpy()
            )

        return p(tr), p(te)
