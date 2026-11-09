import numpy as np
import pytest
import torch

from cvd70k.v2.config import candidates
from cvd70k.v2.models import MODULES, MultiView, count
from cvd70k.v2.selector import (
    calibrated_mask,
    external_gate,
    matched_random,
    multi_mmd,
    permutation_mmd,
    relative_geometry,
    set_mask,
    snapshot_frozen,
    verify_frozen,
)


def example():
    rng = np.random.default_rng(4)
    y = np.repeat([0, 1], 50)
    x = rng.normal(scale=0.4, size=(100, 4)) + y[:, None] * 2
    return x, y


def test_mmd_properties_and_permutation():
    x, _ = example()
    assert multi_mmd(x, x) == pytest.approx(0, abs=1e-12)
    assert multi_mmd(x, x + 4) > multi_mmd(x, x)
    result = permutation_mmd(x, x + 4, permutations=99)
    assert 0.01 <= result["domain_p"] <= 0.05
    assert result == permutation_mmd(x, x + 4, permutations=99)
    with pytest.raises(ValueError):
        permutation_mmd(x, np.zeros((5, 3)))


def test_signed_geometry_reversal_and_invalidity():
    x, y = example()
    reverse = -x
    geometry = relative_geometry(x, y, reverse, y, seed=2)
    assert geometry["geometry_status"] == "valid" and geometry["R"] < 0 and geometry["U"] < 0
    invalid = relative_geometry(np.zeros_like(x), y, x, y)
    assert invalid["geometry_status"] != "valid"
    too_few = relative_geometry(x, y, x[:4], y[:4])
    assert too_few["geometry_status"] != "valid"


def test_exact_logical_policy_and_abstention():
    e = {"geometry_status": "valid", "R": 0.7, "U": 0.9, "domain_status": "valid", "domain_p": 0.01}
    mask, _ = calibrated_mask({"a": e})
    assert mask["a"]
    e = {**e, "domain_p": 0.5}
    assert not calibrated_mask({"a": e})[0]["a"]
    e = {**e, "U": 0.7}
    assert calibrated_mask({"a": e})[0]["a"]
    e = {**e, "geometry_status": "weak_source_geometry"}
    mask, why = calibrated_mask({"a": e})
    assert not mask["a"] and why["a"].startswith("ABSTAIN")


def test_frozen_parameters_and_matched_mask():
    model = MultiView([2, 2, 3, 6, 3], candidates("multiview")[0])
    mask = {n: False for n in MODULES}
    random, meta = matched_random(model, mask, 8)
    assert random == mask and meta["candidate_count"] == 1 and meta["parameter_error"] == 0
    set_mask(model, mask)
    assert count(model) == 33
    before = snapshot_frozen(model, mask)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    loss = model(torch.randn(8, 16)).sum()
    loss.backward()
    opt.step()
    assert verify_frozen(model, before)
    with torch.no_grad():
        next(model.encoders[0].parameters()).add_(1)
    with pytest.raises(RuntimeError):
        verify_frozen(model, before)


def test_external_cross_cohort_gate():
    gate = external_gate({"source_identity": True, "target_identity": True})
    assert gate["status"] == "EXTERNAL_CROSS_COHORT_TRANSFER_NOT_EXECUTED"
    assert "endpoint_crosswalk" in gate["missing"] and "positive_counts" in gate["missing"]
