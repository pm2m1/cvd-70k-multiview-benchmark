"""Pickle-safe scikit-learn AP metric for the established TabNet implementation."""

from pytorch_tabnet.metrics import Metric
from sklearn.metrics import average_precision_score


class AveragePrecision(Metric):
    def __init__(self):
        self._name = "ap"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return average_precision_score(y_true, y_score[:, 1])
