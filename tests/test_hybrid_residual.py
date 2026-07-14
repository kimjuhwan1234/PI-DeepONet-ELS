import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from module.pipeline import hybrid_residual_target


def test_residual_is_fair_minus_mchat_minus_rm():
    fair = np.array([0.90, 1.00, 0.85], dtype="float32")
    mc_hat = np.array([0.98, 1.02, 0.95], dtype="float32")
    rm = np.array([-0.05, -0.01, -0.08], dtype="float32")
    out = hybrid_residual_target(fair, mc_hat, rm)
    np.testing.assert_allclose(out, fair - mc_hat - rm, rtol=0, atol=1e-7)
    assert out.dtype == np.float32


from types import SimpleNamespace
import pandas as pd
from module.pipeline import predict_hybrid


def _fake_D():
    n = 6
    return SimpleNamespace(
        n=n,
        WF=[(np.array([0, 1, 2]), np.array([3]), np.array([4, 5]))],
        FAIR=np.array([0.90, 0.95, 0.85, 0.92, 1.00, 0.80], dtype="float32"),
        MC=np.array([0.98, 0.99, 0.95, 0.97, 1.03, 0.93], dtype="float32"),
        rm=np.array([-0.05, -0.04, -0.06, -0.05, -0.02, -0.07], dtype="float32"),
        ITEM=np.array([f"IT{i}" for i in range(n)]),
        ORD=np.arange(n, dtype=float),
    )


def test_predict_hybrid_rebases_on_mc_hat_not_true_mc():
    D = _fake_D()
    mc_hat = np.array([0.90, 0.91, 0.92, 0.93, 0.94, 0.95], dtype="float32")

    def anchor_fn(D, cfg, tr, va, te, save_path=None):
        return lambda idx: mc_hat[np.asarray(idx)]

    def resid_fn(D, cfg, tr, va, te, target, save_path=None):
        expected_tr = (D.FAIR[tr] - mc_hat[tr] - D.rm[tr]).astype("float32")
        np.testing.assert_allclose(target[tr], expected_tr, atol=1e-6)
        expected_va = (D.FAIR[va] - mc_hat[va] - D.rm[va]).astype("float32")
        np.testing.assert_allclose(target[va], expected_va, atol=1e-6)
        return None, np.array([0.001, 0.002], dtype="float32")

    df = predict_hybrid(D, {"margin": {"model": "xgb", "feature_set": "base"},
                            "data": {"time_decay": True}}, anchor_fn, resid_fn=resid_fn, name=None)
    te = np.array([4, 5])
    np.testing.assert_allclose(df["mc_pred"].values, mc_hat[te], atol=1e-6)
    np.testing.assert_allclose(df["resid_true"].values,
                               D.FAIR[te] - mc_hat[te] - D.rm[te], atol=1e-6)
    np.testing.assert_allclose(df["y_pred"].values,
                               mc_hat[te] + D.rm[te] + np.array([0.001, 0.002]), atol=1e-6)
