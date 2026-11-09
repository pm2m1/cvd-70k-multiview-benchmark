import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, confusion_matrix

from cvd70k.evaluation.metrics import calibration_bins, metrics, paired_bootstrap, select_threshold
from cvd70k.models.networks import FiveViewMLP, FlatMLP, parameter_count
from cvd70k.training.engine import EarlyStopping, make_loss, train_nn
from cvd70k.utils.io import read_json, write_json


@pytest.mark.parametrize("model,count", [(FiveViewMLP(), 7985), (FlatMLP(), 8225)])
def test_forward_count_gradient(model, count):
    x = torch.randn(7, 16)
    output = model(x)
    assert output.shape == (7,)
    assert parameter_count(model) == count
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_weighted_loss():
    loss, weight = make_loss([0, 0, 0, 1], "weighted", "cpu")
    assert weight == 3
    value = loss(torch.zeros(4), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert value.item() == pytest.approx(np.log(2) * 1.5)


def test_early_stopping_restoration():
    model = FlatMLP()
    stop = EarlyStopping(patience=2)
    assert not stop.update(0.6, model, 0)  # epoch-zero eligible when caller supplies it
    saved = stop.state["net.0.0.weight"].clone()
    with torch.no_grad():
        next(model.parameters()).add_(10)
    assert not stop.update(0.7, model, 1)
    assert stop.update(0.8, model, 2)
    assert stop.epoch == 0
    torch.testing.assert_close(stop.state["net.0.0.weight"], saved)
    with pytest.raises(FloatingPointError):
        stop.update(float("nan"), model, 3)


def test_tiny_real_training_determinism():
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 16)).astype("float32")
    y = (x[:, 0] > 0).astype("int64")
    cfg = dict(
        embedding=4,
        fusion=[8, 4],
        flat_hidden=[8, 4],
        lr=0.001,
        weight_decay=0.0001,
        patience=2,
        epochs=3,
        batch_size=20,
    )
    a, info = train_nn(x, y, x, y, "multiview", [2, 2, 3, 6, 3], cfg, 4)
    b, _ = train_nn(x, y, x, y, "multiview", [2, 2, 3, 6, 3], cfg, 4)
    assert info["curve"][0]["epoch"] == 0 and info["best_epoch"] >= 1
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_metrics_confusion_consistency():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.4, 0.8])
    m = metrics(y, p)
    assert [m[k] for k in ["tn", "fp", "fn", "tp"]] == confusion_matrix(
        y, p >= 0.5
    ).ravel().tolist()
    assert sum(m[k] for k in ["tn", "fp", "fn", "tp"]) == len(y)
    assert m["recall"] == 0.5 and m["balanced_accuracy"] == 0.5
    assert m["ap"] == pytest.approx(average_precision_score(y, p))
    assert m["brier"] == pytest.approx(np.mean((y - p) ** 2))
    assert metrics([0, 1], [0, 1])["log_loss"] < 1e-6


def test_threshold_role_and_known_optimum():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])
    threshold = select_threshold(y, p)
    assert 0.2 < threshold <= 0.3
    assert metrics(y, p, threshold)["balanced_accuracy"] == 1
    with pytest.raises(ValueError):
        select_threshold(y, p, role="test")


def test_calibration_bins_include_one():
    bins = calibration_bins(np.array([0, 1, 1]), np.array([0.0, 0.5, 1.0]))
    assert sum(b["n"] for b in bins) == 3
    assert bins[0]["n"] == 1 and bins[-1]["n"] == 1


def test_paired_cluster_bootstrap_ties():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.2, 0.6, 0.6])
    result = paired_bootstrap(y, {"mlp": [p], "multiview": [p]}, np.zeros(4), "mlp", 10)
    expected = average_precision_score(y, p)
    np.testing.assert_allclose(result["ap_ci"]["mlp"], [expected, expected])
    np.testing.assert_allclose(result["ap_difference_vs_mlp"]["multiview"]["ci95"], [0, 0])


@pytest.mark.parametrize("p", [[np.nan, 0.2], [-0.1, 0.1], [1.1, 0.2]])
def test_bad_probabilities(p):
    with pytest.raises(ValueError):
        metrics([0, 1], p)


def test_config_strict_json(tmp_path):
    cfg = read_json("configs/benchmark.json")
    assert len(cfg["seeds"]) == 5 and len(cfg["search"]["mlp"]) == 2
    path = tmp_path / "config.json"
    write_json(path, cfg)
    assert read_json(path) == cfg
    with pytest.raises(ValueError):
        write_json(path, {"x": float("nan")})
