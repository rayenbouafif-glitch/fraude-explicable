"""Module 2 - Variables temporelles et comportementales SANS fuite de données.

Toutes les variables comportementales d'une transaction n'utilisent que les transactions
STRICTEMENT ANTÉRIEURES de la même carte (fenêtres glissantes closed='left').
La même fonction sert à l'entraînement et à l'API : aucune divergence train/production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import CATEGORIES

BASE_FEATURES = [
    "amt", "log_amt", "hour_sin", "hour_cos", "dow", "is_night", "age", "gender_M",
    "log_city_pop", "distance_km", "n_tx_1h", "n_tx_24h", "amt_ratio_30d", "hours_since_last",
]
CAT_FEATURES = [f"cat_{c}" for c in CATEGORIES]
FEATURES = BASE_FEATURES + CAT_FEATURES

FEATURE_LABELS = {
    "amt": "Montant", "log_amt": "Montant (log)", "hour_sin": "Heure (sin)", "hour_cos": "Heure (cos)",
    "dow": "Jour de la semaine", "is_night": "Transaction nocturne", "age": "Âge du porteur",
    "gender_M": "Sexe (M)", "log_city_pop": "Population de la ville (log)",
    "distance_km": "Distance client-marchand (km)", "n_tx_1h": "Nb transactions dernière heure",
    "n_tx_24h": "Nb transactions dernières 24 h", "amt_ratio_30d": "Montant / moyenne 30 j",
    "hours_since_last": "Heures depuis la transaction précédente",
    **{f"cat_{c}": f"Catégorie {c}" for c in CATEGORIES},
}

MAX_GAP_H = 720.0  # 30 jours : valeur par défaut si aucune transaction antérieure


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un DataFrame de variables aligné sur l'index de df."""
    d = df[["ts", "cc_num", "amt", "category", "gender", "lat", "long", "city_pop",
            "dob", "merch_lat", "merch_long"]].copy()
    d["_pos"] = np.arange(len(d))
    d = d.sort_values(["cc_num", "ts", "_pos"], kind="mergesort")

    out = pd.DataFrame(index=d.index)
    hour = d["ts"].dt.hour + d["ts"].dt.minute / 60
    out["amt"] = d["amt"].astype(float)
    out["log_amt"] = np.log1p(out["amt"])
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow"] = d["ts"].dt.dayofweek
    out["is_night"] = ((d["ts"].dt.hour >= 22) | (d["ts"].dt.hour < 4)).astype(int)
    out["age"] = (d["ts"] - d["dob"]).dt.days / 365.25
    out["gender_M"] = (d["gender"] == "M").astype(int)
    out["log_city_pop"] = np.log1p(d["city_pop"].astype(float))
    out["distance_km"] = _haversine(d["lat"], d["long"], d["merch_lat"], d["merch_long"])

    # --- Variables comportementales (passé strict de la carte) ---
    g = d.set_index("ts").groupby("cc_num", sort=False)["amt"]
    out["n_tx_1h"] = g.rolling("1h", closed="left").count().fillna(0).to_numpy()
    out["n_tx_24h"] = g.rolling("24h", closed="left").count().fillna(0).to_numpy()
    mean30 = g.rolling("30D", closed="left").mean().to_numpy()
    out["amt_ratio_30d"] = np.where(np.isnan(mean30) | (mean30 <= 0), 1.0, out["amt"].to_numpy() / mean30)
    gap = d.groupby("cc_num", sort=False)["ts"].diff().dt.total_seconds() / 3600
    out["hours_since_last"] = gap.fillna(MAX_GAP_H).clip(upper=MAX_GAP_H)

    for c in CATEGORIES:
        out[f"cat_{c}"] = (d["category"] == c).astype(int)

    return out.loc[df.index, FEATURES].astype("float32")
