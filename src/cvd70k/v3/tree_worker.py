"""Native tree libraries run without loading PyTorch/OpenMP into their process."""

import sys
import time

import joblib
from sklearn.metrics import average_precision_score


def fit(request):
    family = request["family"]
    cfg = request["config"]
    start = time.perf_counter()
    x, y, xs, ys = request["x"], request["y"], request["xs"], request["ys"]
    if family == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=cfg["rounds"],
            max_depth=cfg["depth"],
            learning_rate=cfg["tree_lr"],
            min_child_weight=cfg["child"] / 4,
            reg_lambda=cfg["l2"],
            subsample=0.85,
            colsample_bytree=0.9,
            n_jobs=4,
            random_state=request["seed"],
            early_stopping_rounds=50,
            eval_metric="logloss",
            tree_method="hist",
        )
        model.fit(x, y, eval_set=[(xs, ys)], verbose=False)
        best = model.best_iteration + 1
    elif family == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=cfg["rounds"],
            max_depth=cfg["depth"],
            num_leaves=2 ** cfg["depth"],
            min_child_samples=cfg["child"],
            learning_rate=cfg["tree_lr"],
            reg_lambda=cfg["l2"],
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            n_jobs=4,
            verbosity=-1,
            random_state=request["seed"],
        )

        def ap(y, p):
            return "sklearn_ap", float(average_precision_score(y, p)), True

        model.fit(
            x,
            y,
            eval_set=[(xs, ys)],
            eval_metric=ap,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best = model.best_iteration_
    elif family == "catboost":
        from catboost import CatBoostClassifier

        # Schema-defined ordinal codes are restored before this worker; native categories.
        x = request.get("cat_x", x)
        xs = request.get("cat_xs", xs)
        model = CatBoostClassifier(
            iterations=cfg["rounds"],
            depth=cfg["depth"],
            learning_rate=cfg["tree_lr"],
            l2_leaf_reg=cfg["l2"],
            thread_count=4,
            random_seed=request["seed"],
            loss_function="Logloss",
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            x,
            y,
            cat_features=request.get("cat_features", []),
            eval_set=(xs, ys),
            early_stopping_rounds=50,
            verbose=False,
        )
        best = model.best_iteration_ + 1
    else:
        raise ValueError(family)
    joblib.dump({"model": model}, request["native_path"])
    return dict(native_path=request["native_path"]), dict(
        best_epoch=int(best),
        fit_seconds=time.perf_counter() - start,
        parameter_count=None,
        total_parameters=None,
        training_device="cpu",
        stopping_objective="sklearn_ap_and_logloss" if family == "lightgbm" else "logloss",
    )


if __name__ == "__main__":
    req, resp = sys.argv[1:]
    request = joblib.load(req)
    if request.get("action") == "predict":
        model = joblib.load(request["checkpoint"])["model"]
        out = model.predict_proba(request["x"])[:, 1]
    else:
        out = fit(request)
    joblib.dump(out, resp)
