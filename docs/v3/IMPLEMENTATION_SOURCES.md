# Verified implementation sources

Official documentation checked during V3 implementation:

- [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier): installed CatBoost 1.2.10, CPU native categorical handling, validation Logloss stopping. The two candidates vary depth, learning rate and L2 regularization.
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/stable/Python-API.html): installed LightGBM 4.7.0, CPU histogram boosting, leaf/min-child and regularization controls. Custom sklearn AP and default Logloss both participate in validation stopping; reported AP is sklearn non-interpolated AP. Deprecated eval_set currently works and its warning is retained locally.
- [Official FT-Transformer implementation](https://github.com/yandex-research/rtdl-revisiting-models): installed rtdl-revisiting-models 0.0.2. V3 uses the existing documented V2 wrapper with native categorical token reconstruction from train-fitted scaling, not an unrelated transformer presented under this name.
- [L2-SP source-preserving transfer regularization, Li et al., ICML 2018](https://proceedings.mlr.press/v80/li18a.html): V3 uses squared distance to the pretrained source backbone rather than distance to zero. This paper does not establish effectiveness for cardiovascular tabular data; V3 tests that separately.
- [Kaggle official dataset metadata](https://www.kaggle.com/api/v1/datasets/view/sulianova/cardiovascular-disease-dataset): version 1, licence field **Unknown** at the recorded check. Attribution: Svetlana Ulianova. No unrestricted participant-data or trained-weight redistribution permission is inferred.

Exact package versions and dependency hashes are locked in `configs/v3/uv.lock`; actual successful/failed execution records take precedence over documentation expectations. No TabPFN or TabNet V3 fits were registered; their V2 statuses remain unchanged.
