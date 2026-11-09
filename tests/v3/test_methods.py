import numpy as np
import pandas as pd
import pytest
import torch

from cvd70k.data.pipeline import clean, feature_groups
from cvd70k.v2.engine import Stopper
from cvd70k.v2.models import MODULES, MultiView
from cvd70k.v2.selector import (
    calibrated_mask,
    matched_random,
    multi_mmd,
    relative_geometry,
    snapshot_frozen,
    verify_frozen,
)
from cvd70k.v3.adaptation import Radial, anchors, drift, policy, scores, source_penalty
from cvd70k.v3.design import CFG, balanced, candidates, isolated, label_cost
from cvd70k.v3.engine import loss_value, state_hash, train
from cvd70k.v3.features import Representation
from cvd70k.v3.models import build, total


@pytest.fixture
def fixture():
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame(
        dict(
            id=np.arange(n),
            age=rng.integers(15000, 24000, n),
            gender=rng.integers(1, 3, n),
            height=rng.normal(165, 10, n),
            weight=rng.normal(70, 10, n),
            ap_hi=rng.normal(130, 15, n),
            ap_lo=rng.normal(80, 8, n),
            cholesterol=rng.integers(1, 4, n),
            gluc=rng.integers(1, 4, n),
            smoke=rng.integers(0, 2, n),
            alco=rng.integers(0, 2, n),
            active=rng.integers(0, 2, n),
            cardio=np.tile([0, 1], 40),
        )
    )
    df.loc[0, "ap_hi"] = 500
    df, _ = clean(df)
    df["group"] = feature_groups(df)
    return df


def test_features_fit_boundary(fixture):
    for block in ["base", "missing", "spline", "nonlinear", "shuffled"]:
        prep = Representation(block).fit(fixture.iloc[:40])
        before = prep.scaler.mean_.copy()
        z = prep.transform(fixture.iloc[40:])
        assert np.isfinite(z).all()
        assert sum(prep.dims) == z.shape[1]
        assert np.array_equal(before, prep.scaler.mean_)
        with pytest.raises(ValueError):
            prep.fit(fixture, role="evaluation")


def test_shapes_and_capacity(fixture):
    prep = Representation().fit(fixture)
    for cfg in candidates().values():
        if cfg["family"] != "neural" or cfg["architecture"] == "ft":
            continue
        torch.manual_seed(1)
        m = build(prep, cfg)
        assert m(torch.tensor(prep.transform(fixture))).shape == (80,)
    assert total(MultiView([2, 2, 3, 6, 3], CFG)) == 7985
    assert total(MultiView([4, 4, 6, 12, 6], CFG)) == 8241


def test_l2sp_and_anchors():
    m = MultiView([2, 2, 3, 6, 3], CFG)
    ref = anchors(m)
    before = state_hash(m)
    assert source_penalty(m, ref, {n: 1 for n in MODULES}) == 0
    with torch.no_grad():
        next(m.encoders[0].parameters()).add_(0.2)
    expected = next(m.encoders[0].parameters()).numel() * 0.04
    assert float(source_penalty(m, ref, {n: 1 for n in MODULES}).detach()) == pytest.approx(
        expected, rel=1e-5
    )
    assert state_hash(m) != before
    assert not torch.equal(next(m.encoders[0].parameters()), ref["encoders.0.0.weight"])


def test_radial_updates_bounds_identifiability():
    torch.manual_seed(2)
    source = MultiView([2, 2, 3, 6, 3], CFG)
    before = state_hash(source)
    caps = {m: 0.5 for m in MODULES}
    caps[MODULES[0]] = 0
    radial = Radial(source, caps)
    x = torch.randn(12, 16)
    assert torch.equal(radial(x), source(x))
    opt = torch.optim.SGD([p for p in radial.parameters() if p.requires_grad], lr=0.1)
    loss = radial(x).square().mean() + radial.regularizer()
    loss.backward()
    opt.step()
    radial.project()
    assert state_hash(source) == before
    effective = radial.materialize()
    changes = drift(source, effective)
    assert changes[MODULES[0]]["absolute"] == 0
    assert any(v["absolute"] > 0 for v in changes.values())
    for m, g in radial.gates().items():
        assert 0 <= g["effective_gate"] <= caps[m] + 1e-6
        assert changes[m]["absolute"] == pytest.approx(
            g["effective_gate"] * radial.radii[m], abs=2e-6
        )
    assert torch.allclose(radial(x), effective(x), atol=1e-6)


def test_policy_invalid_geometry_and_frozen():
    ev = {
        m: dict(geometry_status="invalid", domain_status="valid", domain_p=0.01, U=None, R=None)
        for m in MODULES
    }
    mask, _ = calibrated_mask(ev)
    assert not any(mask.values())
    source = MultiView([2, 2, 3, 6, 3], CFG)
    model, mask, _, _ = policy(source, "logical", ev, 1)
    frozen = snapshot_frozen(model, mask)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 33
    verify_frozen(model, frozen)
    assert all(v == 0 for v in scores(ev)[0].values())
    random, info = matched_random(source, mask, 1)
    assert info["parameter_error"] == 0
    assert not any(random.values())
    ev[MODULES[0]].update(geometry_status="valid", U=0.7, R=0.9)
    assert calibrated_mask(ev)[0][MODULES[0]]


def test_losses_and_stopping():
    z = torch.tensor([-0.2, 0.4], requires_grad=True)
    y = torch.tensor([0.0, 1.0])
    for loss in ["bce", "focal", "rank"]:
        value = loss_value(z, y, {**CFG, "loss": loss, "rank_weight": 0.1})
        assert torch.isfinite(value)
        value.backward(retain_graph=True)
    m = torch.nn.Linear(2, 1)
    s = Stopper("ap", 2, True)
    s.update(dict(epoch=0, validation_ap=0.8), m)
    s.update(dict(epoch=1, validation_ap=0.7), m)
    assert s.update(dict(epoch=2, validation_ap=0.6), m)
    assert s.epoch == 0
    s = Stopper("ap", 1, False)
    s.update(dict(epoch=0, validation_ap=0.99), m)
    assert s.state is None
    s.update(dict(epoch=1, validation_ap=0.5), m)
    assert s.epoch == 1


def test_roles_and_cost(fixture):
    a, b = fixture.iloc[:40], fixture.iloc[40:]
    isolated({"a": a, "b": b})
    with pytest.raises(ValueError):
        isolated({"a": a, "b": a})
    costs = label_cost(fit=a, stop=a.iloc[:10], select=b)
    assert costs["total_distinct"] == 80 and costs["reused_label_count"] == 10
    assert balanced(fixture, 10, 4).cardio.sum() == 10
    with pytest.raises(ValueError):
        balanced(fixture, 100, 4)


def test_geometry_properties():
    rng = np.random.default_rng(2)
    y = np.tile([0, 1], 30)
    x = rng.normal(size=(60, 3)) + y[:, None]
    assert abs(multi_mmd(x, x)) < 1e-10
    r = relative_geometry(x, y, x, y, bootstraps=30)
    assert r["R"] == pytest.approx(1)
    assert relative_geometry(x[:4], y[:4], x, y)["geometry_status"] != "valid"


def test_reproducible_training(fixture):
    prep = Representation("base").fit(fixture.iloc[:40])
    cfg = {**CFG, "epochs": 2, "patience": 2}
    results = []
    for _ in range(2):
        torch.manual_seed(1)
        m = MultiView(prep.dims, cfg)
        model, _ = train(
            m,
            prep.transform(fixture.iloc[:40]),
            fixture.iloc[:40].cardio.to_numpy(),
            prep.transform(fixture.iloc[40:]),
            fixture.iloc[40:].cardio.to_numpy(),
            cfg,
            2,
            "cpu",
        )
        results.append(state_hash(model))
    assert results[0] == results[1]
