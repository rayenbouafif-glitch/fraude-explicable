"""Tests : absence de fuite dans les variables, séparation chronologique, coût, API."""
import numpy as np
import pandas as pd

from src.data import generate_synthetic
from src.features import build_features
from src.pipeline import chronological_split
from src.evaluate import optimal_threshold, business_cost


def _small():
    return generate_synthetic(n_cards=15, days=60, seed=1)


def test_no_future_leakage():
    """Les variables d'une transaction ne changent pas si on supprime toutes les transactions futures."""
    df = _small()
    full = build_features(df)
    cut = df.ts.iloc[len(df) // 2]
    past = df[df.ts <= cut]
    part = build_features(past)
    pd.testing.assert_frame_equal(full.loc[past.index], part, check_exact=False, rtol=1e-5)


def test_chronological_split():
    df = _small()
    tr, va, te = chronological_split(df)
    assert df.ts.iloc[tr].max() <= df.ts.iloc[va].min() <= df.ts.iloc[te].min()
    assert abs(len(tr) / len(df) - 0.6) < 0.01


def test_optimal_threshold_is_minimum():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    s = rng.random(2000) + y * 0.5
    amt = rng.uniform(1, 500, 2000)
    thr, cost, _ = optimal_threshold(y, s, amt, 5.0)
    for t in np.quantile(s, np.linspace(0, 1, 50)):
        assert cost <= business_cost(y, amt, s >= t, 5.0)[0] + 1e-6


def test_api():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        r = c.post("/predict", json={"transaction": {"cc_num": 1, "trans_date_trans_time": "2020-06-01 23:30:00",
                                                     "amt": 900, "category": "shopping_net"}})
        assert r.status_code == 200 and 0 <= r.json()["score"] <= 1
