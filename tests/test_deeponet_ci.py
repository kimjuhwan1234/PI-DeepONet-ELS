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


from types import SimpleNamespace
from model.deeponet_ci import _predict_hybrid_ci


def _fake_D():
    n = 6
    return SimpleNamespace(
        n=n,
        WF=[(np.array([0, 1, 2]), np.array([3]), np.array([4, 5]))],
        MC=np.array([0.98, 0.99, 0.95, 0.97, 1.03, 0.93], dtype="float32"),
        rm=np.zeros(n, dtype="float32"),
        ITEM=np.array([f"IT{i}" for i in range(n)]),
        ORD=np.arange(n, dtype=float),
    )


def test_predict_hybrid_ci_columns_and_interval():
    D = _fake_D()
    mc_hat = np.array([0.90, 0.91, 0.92, 0.93, 0.94, 0.95], dtype="float32")

    def anchor_fn(D, cfg, tr, va, te, save_path=None):
        return lambda idx: mc_hat[np.asarray(idx)]

    def resid_ci_fn(D, cfg, tr, va, te, target, save_path=None):
        # 잔차타깃 = MC − MC_hat − 0 이 train/val 에 올바로 전달됐는지 확인
        exp_tr = (D.MC[tr] - mc_hat[tr]).astype("float32")
        np.testing.assert_allclose(target[tr], exp_tr, atol=1e-6)
        return (np.array([0.01, 0.02], dtype="float32"),
                np.array([0.03, 0.05], dtype="float32"))

    cfg = {"data": {"time_decay": False}}
    df = _predict_hybrid_ci(D, cfg, anchor_fn, resid_ci_fn, name=None,
                            target=D.MC, use_margin=False)

    cols = {"ITEM_CD", "isu_ord", "y_true", "y_pred", "mc_true", "mc_pred",
            "resid_true", "resid_pred", "resid_std", "y_std", "y_lo", "y_hi"}
    assert cols.issubset(df.columns)
    te = np.array([4, 5])
    np.testing.assert_allclose(df["mc_pred"].values, mc_hat[te], atol=1e-6)
    np.testing.assert_allclose(df["y_pred"].values, mc_hat[te] + np.array([0.01, 0.02]), atol=1e-6)
    assert (df["y_lo"].values <= df["y_pred"].values).all()
    assert (df["y_pred"].values <= df["y_hi"].values).all()
    assert (df["y_std"].values > 0).all()
    assert (df["resid_std"].values > 0).all()
    np.testing.assert_allclose((df["y_hi"] - df["y_lo"]).values,
                               2 * Z * np.array([0.03, 0.05]), atol=1e-6)


from model import registry


def _fake_full_D(n=24):
    rng = np.random.RandomState(0)
    tr = np.arange(0, 16); va = np.arange(16, 20); te = np.arange(20, 24)
    return SimpleNamespace(
        n=n,
        WF=[(tr, va, te)],
        CURVE=rng.rand(n, 10).astype("float32"),
        VC=rng.rand(n, 7).astype("float32"),
        CON=rng.rand(n, 15).astype("float32"),
        MC=(0.9 + 0.1 * rng.rand(n)).astype("float32"),
        rm=np.zeros(n, dtype="float32"),
        ITEM=np.array([f"IT{i}" for i in range(n)]),
        ORD=np.arange(n, dtype=float),
        DEV=torch.device("cpu"),
    )


def _smoke_cfg():
    return {
        "seed": 0,
        "train": {"nit": 30, "batch": 8, "lr": 1e-3, "weight_decay": 1e-5,
                  "es_every": 100, "es_patience": 6, "es_rounds": 50},
        "networks": {"P": 16},
        "data": {"time_decay": False},
    }


def test_run_smoke_end_to_end(tmp_path, monkeypatch):
    import model.deeponet_ci as m
    monkeypatch.setattr(m.fm, "RESULT", tmp_path)   # result/ 오염 방지
    D = _fake_full_D()
    out = m.run(D, _smoke_cfg())
    assert set(out.keys()) == {"deeponet_hybrid_ci"}
    df = out["deeponet_hybrid_ci"]
    cols = {"ITEM_CD", "isu_ord", "y_true", "y_pred", "mc_true", "mc_pred",
            "resid_true", "resid_pred", "resid_std", "y_std", "y_lo", "y_hi"}
    assert cols.issubset(df.columns)
    assert len(df) == 4                              # test 폴드 크기
    assert np.isfinite(df["y_pred"].values).all()
    assert (df["y_std"].values > 0).all()
    assert (df["y_lo"].values <= df["y_pred"].values).all()
    assert (df["y_pred"].values <= df["y_hi"].values).all()


def test_nll_resid_returns_positive_std(tmp_path, monkeypatch):
    import model.deeponet_ci as m
    D = _fake_full_D()
    tr, va, te = D.WF[0]
    target = (D.MC - D.MC.mean()).astype("float32")   # 임의 잔차 타깃
    mean_te, std_te = m.nll_resid(D, _smoke_cfg(), tr, va, te, target)
    assert mean_te.shape == te.shape and std_te.shape == te.shape
    assert np.isfinite(mean_te).all()
    assert (std_te > 0).all()


def test_registry_discovers_deeponet_ci():
    names = [n for n, _ in registry.discover()]
    assert "deeponet_ci" in names
