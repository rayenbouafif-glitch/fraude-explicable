"""Module 5 - Explicabilité : SHAP global (beeswarm) et local (waterfall + codes motifs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.features import FEATURE_LABELS

# Codes motifs lisibles par un analyste
REASON_CODES = {
    "amt": "R01_MONTANT", "log_amt": "R01_MONTANT", "amt_ratio_30d": "R02_ECART_PROFIL",
    "n_tx_1h": "R03_VELOCITE_1H", "n_tx_24h": "R04_VELOCITE_24H", "hours_since_last": "R05_DELAI_PRECEDENTE",
    "hour_sin": "R06_HEURE", "hour_cos": "R06_HEURE", "is_night": "R06_HEURE", "dow": "R07_JOUR",
    "distance_km": "R08_DISTANCE", "age": "R09_PROFIL_PORTEUR", "gender_M": "R09_PROFIL_PORTEUR",
    "log_city_pop": "R10_LOCALISATION",
}


def get_explainer(model):
    return shap.TreeExplainer(model)


def shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # anciennes versions : [classe0, classe1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def base_value(explainer) -> float:
    ev = np.atleast_1d(explainer.expected_value)
    return float(ev[-1])


def _describe(feat: str, val: float) -> str:
    if feat in ("amt", "log_amt"):
        v = np.expm1(val) if feat == "log_amt" else val
        return f"Montant de {v:,.2f} $"
    if feat == "amt_ratio_30d":
        return f"Montant = {val:.1f}x la moyenne des 30 derniers jours"
    if feat == "n_tx_1h":
        return f"{int(val)} transaction(s) dans l'heure précédente"
    if feat == "n_tx_24h":
        return f"{int(val)} transaction(s) dans les 24 h précédentes"
    if feat == "hours_since_last":
        return f"{val:.1f} h depuis la transaction précédente"
    if feat in ("hour_sin", "hour_cos", "is_night"):
        return "Heure de la transaction (horaire inhabituel)"
    if feat == "distance_km":
        return f"Marchand à {val:,.0f} km du domicile"
    if feat == "age":
        return f"Âge du porteur : {val:.0f} ans"
    if feat.startswith("cat_"):
        cat = feat[4:]
        return f"Catégorie « {cat} »" if val == 1 else f"Catégorie différente de « {cat} »"
    return f"{FEATURE_LABELS.get(feat, feat)} = {val:.3g}"


def reason_codes(x_row: pd.Series, sv_row: np.ndarray, k: int = 5) -> list[dict]:
    """Les k motifs qui ont le plus POUSSÉ vers l'alerte (contributions SHAP positives).
    Les variables d'une même famille (ex. amt + log_amt, hour_sin + hour_cos) sont regroupées sous un code."""
    groups: dict[str, dict] = {}
    for i in np.argsort(-sv_row):
        f, v = x_row.index[i], float(sv_row[i])
        code = REASON_CODES.get(f, "R11_CATEGORIE" if f.startswith("cat_") else "R99_AUTRE")
        if f.startswith("cat_"):
            code = f"R11_CAT_{f[4:].upper()}"
        g = groups.setdefault(code, {"code": code, "variable": f, "libellé": FEATURE_LABELS.get(f, f),
                                     "valeur": float(x_row.iloc[i]),
                                     "explication": _describe(f, float(x_row.iloc[i])),
                                     "contribution_shap": 0.0})
        g["contribution_shap"] += v
    out = sorted((g for g in groups.values() if g["contribution_shap"] > 0),
                 key=lambda g: -g["contribution_shap"])[:k]
    for g in out:
        g["contribution_shap"] = round(g["contribution_shap"], 4)
    return out


def explanation_object(explainer, X_row: pd.DataFrame) -> shap.Explanation:
    sv = shap_values(explainer, X_row)[0]
    return shap.Explanation(values=sv, base_values=base_value(explainer), data=X_row.iloc[0].values,
                            feature_names=[FEATURE_LABELS.get(c, c) for c in X_row.columns])
