import numpy as np

from cvd70k.experiments.runner import xgb_process


def test_isolated_xgboost_with_torch_loaded(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 16)).astype("float32")
    y = (x[:, 0] > 0).astype(int)
    model = str(tmp_path / "model.joblib")
    result = xgb_process(
        dict(
            operation="fit",
            x=x,
            y=y,
            xv=x,
            yv=y,
            params=dict(n_estimators=4, n_jobs=1, max_depth=2, early_stopping_rounds=2),
            model_path=model,
        )
    )
    p = xgb_process(dict(operation="predict", x=x, model_path=model))
    np.testing.assert_array_equal(result["probability"], p)
    assert np.isfinite(p).all()
