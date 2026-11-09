"""Train-only representations preserving the five explicit clinical views."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

from cvd70k.v2.data import Features


class Representation:
    def __init__(self, block="missing", scale="standard"):
        self.block = block
        self.scale = scale
        self.base = Features("base" if block == "base" else "missing")
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = (
            QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=73)
            if scale == "quantile"
            else StandardScaler()
        )
        self.spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
        self.age_imputer = SimpleImputer(strategy="median")

    def raw(self, df):
        a = self.base.raw(df)
        dims = self.base.dims
        pieces = []
        names = []
        outdims = []
        offset = 0
        for i, d in enumerate(dims):
            v = a[:, offset : offset + d].copy()
            nn = self.base.names[offset : offset + d].copy()
            offset += d
            if self.block == "shuffled":
                # Deterministic independent permutations per call; no label input.
                # Prevents learning quality flags while retaining their marginal rates.
                for j, name in enumerate(nn):
                    if name.endswith("_missing"):
                        v[:, j] = np.random.default_rng(930 + j).permutation(v[:, j])
            extra = {}
            if self.block == "nonlinear":
                extra = (
                    {
                        "sbp_squared": (df.ap_hi / 100) ** 2,
                        "dbp_squared": (df.ap_lo / 100) ** 2,
                        "sbp_dbp": df.ap_hi * df.ap_lo / 10000,
                        "age_pressure_squared": df.age_years * (df.ap_hi / 100) ** 2,
                    }
                    if i == 1
                    else {
                        "bmi_sbp": df.bmi * df.ap_hi / 100,
                        "weight_height": df.weight * df.height / 10000,
                    }
                    if i == 2
                    else {}
                )
            for name, x in extra.items():
                v = np.column_stack([v, np.asarray(x)])
                nn.append(name)
            if self.block == "spline" and i == 0:
                spline = self.spline.transform(self.age_imputer.transform(df[["age_years"]]))
                v = np.column_stack([v, spline])
                nn += [f"age_spline_{j}" for j in range(spline.shape[1])]
            pieces.append(v)
            names += nn
            outdims.append(v.shape[1])
        self.names = names
        self.dims = outdims
        return np.column_stack(pieces)

    def fit(self, df, role="train"):
        if role != "train":
            raise ValueError("Only training roles may fit representations")
        self.fit_rows = tuple(df.index)
        if self.block == "spline":
            self.spline.fit(self.age_imputer.fit_transform(df[["age_years"]]))
        self.scaler.fit(self.imputer.fit_transform(self.raw(df)))
        return self

    def transform(self, df):
        return self.scaler.transform(self.imputer.transform(self.raw(df))).astype("float32")
