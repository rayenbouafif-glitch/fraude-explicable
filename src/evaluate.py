"""Module 4 - Métriques adaptées aux événements rares, budget d'alertes et seuil par coût métier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             fbeta_score, matthews_corrcoef, confusion_matrix)


def precision_recall_at_k(y, scores, k: int):
    k = int(max(1, min(k, len(y))))
    top = np.argsort(-scores, kind="mergesort")[:k]
    tp = y[top].sum()
    return tp / k, tp / max(1, y.sum())


def business_cost(y, amt, alerts, review_cost: float):
    """Coût = montants des fraudes manquées + coût d'analyse des alertes."""
    missed = amt[(y == 1) & (~alerts)].sum()
    return float(missed + review_cost * alerts.sum()), float(missed), float(review_cost * alerts.sum())


def optimal_threshold(y, scores, amt, review_cost: float):
    """Balaye tous les seuils possibles (tri décroissant + cumsum) et retourne celui qui minimise le coût."""
    order = np.argsort(-scores, kind="mergesort")
    s, yy, aa = scores[order], y[order], amt[order]
    fraud_amt = np.where(yy == 1, aa, 0.0)
    total_fraud = fraud_amt.sum()
    n_alerts = np.arange(0, len(s) + 1)
    caught = np.concatenate([[0.0], np.cumsum(fraud_amt)])
    cost = (total_fraud - caught) + review_cost * n_alerts
    # seuil valide uniquement entre deux scores distincts
    valid = np.ones_like(cost, dtype=bool)
    valid[1:-1] = s[:-1] != s[1:]
    cost_v = np.where(valid, cost, np.inf)
    n = int(np.argmin(cost_v))
    thr = np.inf if n == 0 else float(s[n - 1])
    curve_idx = np.unique(np.linspace(0, len(s), 400).astype(int))
    curve = {"n_alerts": n_alerts[curve_idx].tolist(), "cost": cost[curve_idx].tolist(),
             "threshold": [float(s[i - 1]) if i > 0 else float(s[0]) + 1e-9 for i in curve_idx]}
    return thr, float(cost[n]), curve


def metrics_at(y, scores, amt, thr, k, review_cost):
    alerts = scores >= thr
    p_k, r_k = precision_recall_at_k(y, scores, k)
    cost, missed, review = business_cost(y, amt, alerts, review_cost)
    tn, fp, fn, tp = confusion_matrix(y, alerts.astype(int), labels=[0, 1]).ravel()
    return {
        "PR-AUC": float(average_precision_score(y, scores)),
        "ROC-AUC": float(roc_auc_score(y, scores)),
        "Précision@k": float(p_k), "Rappel@k": float(r_k),
        "F2": float(fbeta_score(y, alerts, beta=2, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, alerts)) if alerts.any() and (~alerts).any() else 0.0,
        "Alertes": int(alerts.sum()), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Précision": float(tp / max(1, tp + fp)), "Rappel": float(tp / max(1, tp + fn)),
        "Coût total": cost, "Fraudes manquées ($)": missed, "Coût analyse ($)": review,
    }


def pr_curve_points(y, scores, n=300):
    p, r, _ = precision_recall_curve(y, scores)
    idx = np.unique(np.linspace(0, len(p) - 1, n).astype(int))
    return {"precision": p[idx].tolist(), "recall": r[idx].tolist()}
