import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def frame():
    n = 100
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "age": np.arange(n) + 18000,
            "gender": rng.integers(1, 3, n),
            "height": np.full(n, 170),
            "weight": np.full(n, 70.0),
            "ap_hi": np.full(n, 120),
            "ap_lo": np.full(n, 80),
            "cholesterol": rng.integers(1, 4, n),
            "gluc": rng.integers(1, 4, n),
            "smoke": rng.integers(0, 2, n),
            "alco": rng.integers(0, 2, n),
            "active": rng.integers(0, 2, n),
            "cardio": np.tile([0, 1], n // 2),
        }
    )
