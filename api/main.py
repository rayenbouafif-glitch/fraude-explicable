"""Module 6 - API FastAPI de scoring et d'explication des transactions.

Lancement :  uvicorn api.main:app --reload
Interface d'analyse : http://127.0.0.1:8000/
Documentation technique : http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.data import CATEGORIES  # noqa: E402
from src.features import build_features, FEATURE_LABELS  # noqa: E402
from src.explain import get_explainer, shap_values, base_value, reason_codes  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="API Détection de fraude explicable",
              description="Score de risque, décision d'alerte au seuil coût-optimal et codes motifs SHAP.",
              version="1.1")

STATE: dict = {}


def _load():
    path = config.ART_DIR / "model_bundle.joblib"
    if not path.exists():
        raise RuntimeError("Aucun modèle entraîné : lancez d'abord `python train.py`.")
    b = joblib.load(path)
    STATE.update(bundle=b, explainer=get_explainer(b["model"]),
                 history=joblib.load(config.ART_DIR / "card_history.joblib"))


class Transaction(BaseModel):
    cc_num: int = Field(..., description="Numéro de carte (identifiant)")
    trans_date_trans_time: str = Field(..., examples=["2020-12-31 23:15:00"])
    amt: float = Field(..., gt=0, examples=[950.0])
    category: Literal[tuple(CATEGORIES)] = Field(..., examples=["shopping_net"])
    gender: Literal["M", "F"] = "F"
    lat: float = 40.0
    long: float = -100.0
    city_pop: int = 5000
    dob: str = Field("1980-01-01", examples=["1980-01-01"])
    merch_lat: float = 40.5
    merch_long: float = -100.5
    trans_num: str | None = None


class ScoreRequest(BaseModel):
    transaction: Transaction
    update_history: bool = Field(False, description="Ajouter la transaction à l'historique de la carte")
    n_reasons: int = Field(5, ge=1, le=10)


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., max_length=1000)
    n_reasons: int = Field(5, ge=1, le=10)


def _row(t: Transaction) -> pd.DataFrame:
    try:
        ts, dob = pd.to_datetime(t.trans_date_trans_time), pd.to_datetime(t.dob)
    except (ValueError, TypeError):
        raise HTTPException(422, "Date invalide : format attendu AAAA-MM-JJ HH:MM:SS")
    return pd.DataFrame([{**t.model_dump(exclude={"trans_num", "trans_date_trans_time"}), "ts": ts, "dob": dob}])


def _features_for(t: Transaction) -> pd.DataFrame:
    """Recalcule les variables avec EXACTEMENT la fonction d'entraînement, à partir de l'historique de la carte."""
    hist = STATE["history"]
    row = _row(t)
    past = hist[(hist.cc_num == t.cc_num) & (hist.ts < row.ts.iloc[0])]
    past = past[past.ts >= row.ts.iloc[0] - pd.Timedelta(days=30)]
    frame = pd.concat([past, row], ignore_index=True)
    return build_features(frame).iloc[[-1]]


def _score(t: Transaction, n_reasons: int) -> dict:
    b = STATE["bundle"]
    X = _features_for(t)
    p = float(b["model"].predict_proba(X)[:, 1][0])
    alert = p >= b["threshold"]
    sv = shap_values(STATE["explainer"], X)[0]
    return {
        "trans_num": t.trans_num, "score": round(p, 6), "seuil": b["threshold"], "alerte": bool(alert),
        "niveau_risque": "ÉLEVÉ" if alert else ("MOYEN" if p >= b["threshold"] / 2 else "FAIBLE"),
        "codes_motifs": reason_codes(X.iloc[0], sv, n_reasons),
        "variables": {k: round(float(v), 4) for k, v in X.iloc[0].items()},
    }


def _require_model():
    if "bundle" not in STATE:
        raise HTTPException(503, "Modèle non chargé : lancez `python train.py` puis redémarrez l'API")


@app.on_event("startup")
def startup():
    try:
        _load()
    except RuntimeError as e:
        print(f"[API] {e}")


@app.get("/", include_in_schema=False)
def interface():
    """Interface web d'analyse (console analyste)."""
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "modele_charge": "bundle" in STATE}


@app.get("/model/info")
def model_info():
    _require_model()
    b = STATE["bundle"]
    return {k: b[k] for k in ("model_name", "strategy", "threshold", "review_cost", "budget",
                              "trained_at", "source", "features")}


@app.post("/predict")
def predict(req: ScoreRequest):
    _require_model()
    out = _score(req.transaction, req.n_reasons)
    if req.update_history:
        new = _row(req.transaction)
        STATE["history"] = pd.concat([STATE["history"], new[STATE["history"].columns]], ignore_index=True)
    return out


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    _require_model()
    res = [_score(t, req.n_reasons) for t in req.transactions]
    return {"n": len(res), "n_alertes": sum(r["alerte"] for r in res), "resultats": res}


@app.post("/explain")
def explain(req: ScoreRequest):
    """Contributions SHAP complètes (log-odds) pour un graphique waterfall côté client."""
    _require_model()
    X = _features_for(req.transaction)
    sv = shap_values(STATE["explainer"], X)[0]
    contrib = sorted(({"variable": f, "libellé": FEATURE_LABELS[f], "valeur": float(X.iloc[0][f]),
                       "shap": float(v)} for f, v in zip(X.columns, sv)), key=lambda d: -abs(d["shap"]))
    return {"valeur_de_base": base_value(STATE["explainer"]),
            "somme_log_odds": base_value(STATE["explainer"]) + float(np.sum(sv)), "contributions": contrib}
