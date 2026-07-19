import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from model.deeponet_ci import MarginOperatorNLL, _nll_loss, LAM, LV_MAX, Z


def test_marginoperator_nll_forward_shapes():
    net = MarginOperatorNLL(nb=7, nt=25, P=16)
    bx = torch.randn(5, 7); tx = torch.randn(5, 25)
    mu, lv = net(bx, tx)
    assert mu.shape == (5,)
    assert lv.shape == (5,)


def test_nll_loss_finite_and_clamped():
    y = torch.tensor([0.0, 1.0, -1.0])
    mu = torch.tensor([0.0, 1.0, -1.0])
    lv = torch.tensor([0.0, 0.0, 0.0])
    assert torch.isfinite(_nll_loss(mu, lv, y))
    # 극단 logσ² 도 clamp 로 유한 유지
    lv_ext = torch.tensor([-1e9, 1e9, 0.0])
    assert torch.isfinite(_nll_loss(mu, lv_ext, y))


def test_nll_loss_penalizes_bad_fit():
    far = _nll_loss(torch.tensor([5.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    near = _nll_loss(torch.tensor([0.1]), torch.tensor([0.0]), torch.tensor([0.0]))
    assert far > near


def test_constants():
    assert LAM == 1e-6 and LV_MAX == 10.0 and Z == 2.0
