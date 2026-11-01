"""Module 1 - Chargement des données (Sparkov) : téléchargement Kaggle, import de fichier ou jeu synthétique."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import config

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel",
]

REQUIRED = ["trans_date_trans_time", "cc_num", "category", "amt", "gender", "lat", "long",
            "city_pop", "dob", "merch_lat", "merch_long", "is_fraud"]


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes (format Sparkov attendu) : {missing}")
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["cc_num"] = df["cc_num"].astype("int64")
    df["is_fraud"] = df["is_fraud"].astype(int)
    if "trans_num" not in df.columns:
        df["trans_num"] = [f"T{i:09d}" for i in range(len(df))]
    if "merchant" not in df.columns:
        df["merchant"] = "inconnu"
    df = df.sort_values(["ts", "cc_num"], kind="mergesort").reset_index(drop=True)
    return df


def sample_cards(df: pd.DataFrame, fraction: float, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Échantillonne des CARTES (pas des lignes) pour garder des historiques complets."""
    if fraction >= 1.0:
        return df
    cards = df["cc_num"].unique()
    rng = np.random.default_rng(seed)
    keep = rng.choice(cards, size=max(1, int(len(cards) * fraction)), replace=False)
    return df[df["cc_num"].isin(keep)].reset_index(drop=True)


def load_csv(paths: list[str | Path] | list) -> pd.DataFrame:
    """Import de un ou plusieurs CSV (ex. fraudTrain.csv + fraudTest.csv), chemins ou fichiers ouverts."""
    frames = [pd.read_csv(p) for p in paths]
    return _standardize(pd.concat(frames, ignore_index=True))


def download_kaggle() -> pd.DataFrame:
    """Téléchargement automatique via kagglehub (nécessite un compte Kaggle configuré)."""
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError("Installez kagglehub (pip install kagglehub) ou importez les CSV manuellement.") from e
    folder = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    files = sorted(folder.glob("fraud*.csv"))
    if not files:
        raise RuntimeError(f"Aucun fichier fraud*.csv trouvé dans {folder}")
    return load_csv(files)


def find_local_csv() -> list[Path]:
    return sorted(config.DATA_DIR.glob("fraud*.csv"))


# ---------------------------------------------------------------------------
# Jeu synthétique au format Sparkov (démo hors-ligne et tests)
# ---------------------------------------------------------------------------
_CAT_AMT = {  # (moyenne log, écart-type log) des montants légitimes par catégorie
    "entertainment": (3.6, 0.8), "food_dining": (3.4, 0.7), "gas_transport": (4.0, 0.4),
    "grocery_net": (3.8, 0.5), "grocery_pos": (4.5, 0.6), "health_fitness": (3.6, 0.7),
    "home": (3.8, 0.8), "kids_pets": (3.7, 0.8), "misc_net": (3.5, 1.0), "misc_pos": (3.6, 0.9),
    "personal_care": (3.3, 0.8), "shopping_net": (3.9, 1.0), "shopping_pos": (3.9, 0.9),
    "travel": (3.0, 1.3),
}
_FRAUD_CATS = ["shopping_net", "grocery_pos", "misc_net", "shopping_pos", "gas_transport", "misc_pos"]
_FRAUD_P = [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]


def generate_synthetic(n_cards: int = 300, days: int = 540, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Simule des transactions de carte au format Sparkov avec des épisodes de fraude réalistes :
    rafales nocturnes, montants élevés (ou petits montants de test), catégories en ligne."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-01-01")
    rows = []
    hours_p = np.array([1, .5, .4, .3, .3, .5, 1, 2, 3, 4, 4.5, 5, 6, 6, 5.5, 5, 5, 5.5, 6, 5.5, 4, 3, 2, 1.5])
    hours_p = hours_p / hours_p.sum()
    for c in range(n_cards):
        cc = 4_000_000_000_000 + c * 7919
        lat, lon = rng.uniform(30, 47), rng.uniform(-120, -75)
        city_pop = int(np.exp(rng.uniform(5, 14)))
        dob = start - pd.Timedelta(days=int(rng.uniform(18, 85) * 365))
        gender = rng.choice(["M", "F"])
        rate = rng.gamma(2.0, 0.85)                     # transactions/jour
        n = rng.poisson(rate * days)
        prefs = rng.dirichlet(np.ones(len(CATEGORIES)) * 0.8)
        d = rng.integers(0, days, n)
        h = rng.choice(24, n, p=hours_p)
        ts = start + pd.to_timedelta(d, "D") + pd.to_timedelta(h, "h") + pd.to_timedelta(rng.integers(0, 3600, n), "s")
        cats = rng.choice(CATEGORIES, n, p=prefs)
        amt = np.array([np.exp(rng.normal(*_CAT_AMT[k])) for k in cats])
        rows.append(pd.DataFrame({
            "trans_date_trans_time": ts, "cc_num": cc, "category": cats, "amt": amt.round(2),
            "gender": gender, "lat": lat, "long": lon, "city_pop": city_pop, "dob": dob,
            "merch_lat": lat + rng.normal(0, 0.6, n), "merch_long": lon + rng.normal(0, 0.6, n),
            "is_fraud": 0,
        }))
        # Épisode de fraude (environ 1 carte sur 2)
        if rng.random() < 0.5:
            k = rng.integers(4, 14)
            t0 = start + pd.Timedelta(days=int(rng.integers(30, days - 3)))
            offs = np.sort(rng.uniform(0, rng.uniform(4, 60), k))           # rafale sur quelques heures/jours
            night = rng.choice([22, 23, 0, 1, 2, 3], k)
            fts = t0 + pd.to_timedelta(offs, "h") + pd.to_timedelta(night, "h")
            fc = rng.choice(_FRAUD_CATS, k, p=_FRAUD_P)
            big = rng.random(k) < 0.8
            famt = np.where(big, np.exp(rng.normal(6.2, 0.6, k)), rng.uniform(1, 20, k))
            rows.append(pd.DataFrame({
                "trans_date_trans_time": fts, "cc_num": cc, "category": fc, "amt": famt.round(2),
                "gender": gender, "lat": lat, "long": lon, "city_pop": city_pop, "dob": dob,
                "merch_lat": lat + rng.normal(0, 1.0, k), "merch_long": lon + rng.normal(0, 1.0, k),
                "is_fraud": 1,
            }))
    df = pd.concat(rows, ignore_index=True)
    df["trans_date_trans_time"] = df["trans_date_trans_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["dob"] = df["dob"].dt.strftime("%Y-%m-%d")
    df["merchant"] = "fraud_" + df["category"]
    df["trans_num"] = [f"S{i:09d}" for i in range(len(df))]
    return _standardize(df)


def load_dataset(source: str = "auto", paths=None, card_fraction: float = config.CARD_FRACTION):
    """source : 'auto' | 'csv' | 'kaggle' | 'synthetic'. Retourne (df, description_source)."""
    if source == "csv":
        df, desc = load_csv(paths), "CSV importé(s)"
    elif source == "kaggle":
        df, desc = download_kaggle(), "Kaggle - Sparkov (kartik2112/fraud-detection)"
    elif source == "synthetic":
        df, desc = generate_synthetic(), "Synthétique au format Sparkov (démo)"
    else:  # auto
        local = find_local_csv()
        if local:
            df, desc = load_csv(local), f"CSV locaux : {', '.join(p.name for p in local)}"
        else:
            df, desc = generate_synthetic(), "Synthétique au format Sparkov (aucun CSV dans data/)"
    return sample_cards(df, card_fraction), desc
