import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from cvd70k.data.pipeline import clean, feature_groups
from cvd70k.v2.config import candidates
from cvd70k.v2.data import Features, folds
from cvd70k.v2.engine import Stopper, train_network
from cvd70k.v2.models import MLP, MultiView, count
from cvd70k.v2.postprocess import Calibrator, convex_weights, operating_points


def test_group_cv_and_features(frame):
    df, _ = clean(frame)
    df["group"] = feature_groups(frame)
    cv = folds(df)
    assert set(cv) == {0, 1, 2}
    for k in range(3):
        assert not set(df.loc[cv == k, "group"]) & set(df.loc[cv != k, "group"])
    prep = Features("both").fit(df.iloc[:60])
    old = prep.scaler.mean_.copy()
    x = prep.transform(df.iloc[60:])
    assert np.isfinite(x).all() and x.shape[1] > 16
    np.testing.assert_array_equal(old, prep.scaler.mean_)
    with pytest.raises(ValueError):
        Features().fit(df, role="outer")
    raw = Features("derived")
    a = raw.raw(df)
    j = raw.names.index("pulse_pressure")
    np.testing.assert_allclose(a[:, j], df.ap_hi - df.ap_lo)


def test_shapes_and_counts():
    cfg = candidates("multiview")[0]
    mv = MultiView([2, 2, 3, 6, 3], cfg)
    flat = MLP(16, candidates("mlp")[0])
    assert count(mv) == 7985 and count(flat) == 8225
    z, features = mv(torch.randn(4, 16), True)
    assert z.shape == (4,) and len(features) == 7


def test_ap_stop_and_epoch_zero():
    model = MLP(16, candidates("mlp")[0])
    stop = Stopper("ap", 2, zero=True)
    stop.update({"epoch": 0, "validation_ap": 0.8}, model)
    stop.update({"epoch": 1, "validation_ap": 0.7}, model)
    assert stop.update({"epoch": 2, "validation_ap": 0.79}, model)
    assert stop.epoch == 0
    scratch = Stopper("ap", 2, zero=False)
    scratch.update({"epoch": 0, "validation_ap": 0.99}, model)
    scratch.update({"epoch": 1, "validation_ap": 0.6}, model)
    scratch.update({"epoch": 2, "validation_ap": 0.7}, model)
    assert scratch.epoch == 2


def test_calibration_threshold_ensemble_roles():
    y = np.tile([0, 1], 40)
    p = np.linspace(0.01, 0.99, 80)
    for kind in ["none", "platt", "temperature", "isotonic"]:
        cal = Calibrator(kind).fit(y, p)
        q = cal.transform(p)
        assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
        if kind in ["platt", "temperature"]:
            assert average_precision_score(y, q) == pytest.approx(average_precision_score(y, p))
    with pytest.raises(ValueError):
        Calibrator("platt").fit(y, p, role="outer")
    with pytest.raises(ValueError):
        convex_weights(y, {"a": p, "b": p}, role="outer")
    names, w = convex_weights(y, {"a": p, "b": p[::-1]})
    assert len(names) == 2 and w.sum() == pytest.approx(1)
    assert operating_points(y, p)["fixed_050"] == 0.5


def test_training_reproducible_cpu():
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 16)).astype("float32")
    y = (x[:, 0] > 0).astype(int)
    cfg = candidates("multiview")[0]
    cfg = {**cfg, "epochs": 2, "batch": 20}

    def fit():
        torch.manual_seed(101)
        m = MultiView([2, 2, 3, 6, 3], cfg)
        return train_network(m, x, y, x, y, cfg, 101)[0]

    a, b = fit(), fit()
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_official_ft_tokens_and_shape(frame):
    pytest.importorskip("rtdl_revisiting_models")
    from cvd70k.v2.models import FT

    df, _ = clean(frame)
    prep = Features().fit(df)
    model = FT(prep, candidates("ft_transformer")[0])
    output = model(torch.tensor(prep.transform(df.iloc[:4])))
    assert output.shape == (4,) and torch.isfinite(output).all()


def test_weighted_rank_metrics_and_bootstrap():
    from sklearn.metrics import roc_auc_score

    from cvd70k.v2.statistics import paired_interval, ranked_cache, weighted_scores

    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.2, 0.4, 0.8, 0.8, 0.9])
    ap, auc = weighted_scores(ranked_cache(y, p), np.ones(len(y)))
    assert ap == pytest.approx(average_precision_score(y, p))
    assert auc == pytest.approx(roc_auc_score(y, p))
    result = paired_interval(y, [p], [p], np.arange(len(y)), replicates=20)
    assert result["ap_difference"] == 0 and result["auroc_difference"] == 0
    np.testing.assert_allclose(result["ap_ci95"], [0, 0])


def test_oof_incomplete_rejected():
    with pytest.raises(ValueError):
        convex_weights([0, 1], {"a": np.array([0.2, np.nan]), "b": np.array([0.3, 0.8])})
