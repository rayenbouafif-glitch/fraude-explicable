"""Module 3 - Modèles supervisés et non supervisés, stratégies contre le déséquilibre."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

RS = config.RANDOM_STATE


# ---------------------------------------------------------------------------
# Supervisé
# ---------------------------------------------------------------------------
def make_lgbm(scale_pos_weight: float = 1.0):
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=RS, n_jobs=-1, verbose=-1,
    )


def undersample(X, y, ratio: int = 10, seed: int = RS):
    """Sous-échantillonnage aléatoire des légitimes : ratio légitimes pour 1 fraude (train uniquement)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep_neg = rng.choice(neg, size=min(len(neg), ratio * len(pos)), replace=False)
    idx = np.sort(np.concatenate([pos, keep_neg]))
    return X.iloc[idx], y[idx]


IMBALANCE_STRATEGIES = {
    "Aucune": "none",
    "Pondération des classes": "weight",
    "Sous-échantillonnage 1:10": "under",
}


def fit_lgbm_strategy(strategy: str, X, y):
    if strategy == "weight":
        spw = (y == 0).sum() / max(1, (y == 1).sum())
        return make_lgbm(scale_pos_weight=spw).fit(X, y)
    if strategy == "under":
        Xu, yu = undersample(X, y)
        return make_lgbm().fit(Xu, yu)
    return make_lgbm().fit(X, y)


def fit_logreg(X, y):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)).fit(X, y)


# ---------------------------------------------------------------------------
# Non supervisé (les étiquettes ne sont PAS utilisées pour l'apprentissage)
# ---------------------------------------------------------------------------
class UnsupervisedScorer:
    """Normalise les variables puis renvoie un score d'anomalie (plus grand = plus suspect)."""

    def __init__(self, kind: str):
        self.kind = kind
        self.scaler = StandardScaler()

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        rng = np.random.default_rng(RS)
        if self.kind == "iforest":
            self.model = IsolationForest(n_estimators=300, max_samples=4096, contamination="auto",
                                         random_state=RS, n_jobs=-1).fit(Xs)
        elif self.kind == "lof":
            idx = rng.choice(len(Xs), size=min(len(Xs), config.LOF_SAMPLE), replace=False)
            self.model = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(Xs[idx])
        elif self.kind == "autoencoder":
            idx = rng.choice(len(Xs), size=min(len(Xs), config.AE_SAMPLE), replace=False)
            self.model = MLPRegressor(hidden_layer_sizes=(16, 6, 16), activation="relu", max_iter=60,
                                      early_stopping=True, random_state=RS).fit(Xs[idx], Xs[idx])
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        if self.kind == "autoencoder":
            return np.mean((self.model.predict(Xs) - Xs) ** 2, axis=1)
        return -self.model.score_samples(Xs)


UNSUPERVISED = {"Isolation Forest": "iforest", "LOF": "lof", "Autoencodeur": "autoencoder"}
