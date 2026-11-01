"""Pipeline complet : données -> variables -> séparation chronologique -> modèles -> seuil -> SHAP -> artefacts."""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")
import math
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score

import config
from src import data as D
from src.features import build_features, FEATURES, FEATURE_LABELS
from src.models import (IMBALANCE_STRATEGIES, UNSUPERVISED, UnsupervisedScorer,
                        fit_lgbm_strategy, fit_logreg)
from src.evaluate import (metrics_at, optimal_threshold, pr_curve_points, precision_recall_at_k)
from src.explain import get_explainer, shap_values

RAW_KEEP = ["ts", "cc_num", "amt", "category", "merchant", "trans_num", "is_fraud"]


def _log(cb, msg):
    print(msg, flush=True)
    if cb:
        cb(msg)


def chronological_split(df: pd.DataFrame, fractions=config.SPLIT):
    """df trié par date : 60/20/20 dans l'ordre du temps (aucune transaction future dans l'entraînement)."""
    n = len(df)
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def eda_summary(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(config.RANDOM_STATE)
    legit = df.loc[df.is_fraud == 0, "amt"].to_numpy()
    month = df.groupby(df.ts.dt.to_period("M"))["is_fraud"].agg(["mean", "sum", "count"])
    month.index = month.index.astype(str)
    return {
        "n_rows": int(len(df)), "n_cards": int(df.cc_num.nunique()), "n_fraud": int(df.is_fraud.sum()),
        "fraud_rate": float(df.is_fraud.mean()),
        "period": (str(df.ts.min()), str(df.ts.max())),
        "by_month": month.reset_index(names="mois"),
        "by_hour": df.groupby(df.ts.dt.hour)["is_fraud"].agg(["mean", "count"]).reset_index(names="heure"),
        "by_dow": df.groupby(df.ts.dt.dayofweek)["is_fraud"].agg(["mean", "count"]).reset_index(names="jour"),
        "by_category": df.groupby("category")["is_fraud"].agg(["mean", "sum", "count"])
                         .sort_values("mean", ascending=False).reset_index(),
        "amt_fraud": df.loc[df.is_fraud == 1, "amt"].to_numpy(),
        "amt_legit": rng.choice(legit, size=min(len(legit), 30_000), replace=False),
        "amt_stats": df.groupby("is_fraud")["amt"].describe().reset_index(),
    }


def run(source="auto", paths=None, card_fraction=config.CARD_FRACTION, review_cost=config.REVIEW_COST,
        progress=None) -> dict:
    t0 = time.time()
    _log(progress, "1/7 Chargement des données…")
    df, source_desc = D.load_dataset(source, paths, card_fraction)
    _log(progress, f"    {len(df):,} transactions, {df.cc_num.nunique()} cartes, taux de fraude {df.is_fraud.mean():.3%}")
    eda = eda_summary(df)

    _log(progress, "2/7 Construction des variables (passé strict de chaque carte)…")
    X = build_features(df)
    y = df["is_fraud"].to_numpy()
    amt = df["amt"].to_numpy()

    _log(progress, "3/7 Séparation chronologique 60/20/20…")
    tr, va, te = chronological_split(df)
    split_info = {name: {"n": int(len(ix)), "début": str(df.ts.iloc[ix[0]]), "fin": str(df.ts.iloc[ix[-1]]),
                         "fraudes": int(y[ix].sum()), "taux": float(y[ix].mean())}
                  for name, ix in [("Entraînement", tr), ("Validation", va), ("Test", te)]}
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    # --- Budget d'alertes k justifié ---
    def days(ix):
        return max(1.0, (df.ts.iloc[ix[-1]] - df.ts.iloc[ix[0]]).total_seconds() / 86400)
    daily_volume = len(te) / days(te)
    exp_fraud_day = ytr.mean() * daily_volume
    k_per_day = max(1, math.ceil(config.BUDGET_FACTOR * exp_fraud_day))
    k_val, k_test = int(k_per_day * days(va)), int(k_per_day * days(te))
    budget = {"k_par_jour": k_per_day, "k_validation": k_val, "k_test": k_test,
              "volume_jour": daily_volume, "fraudes_attendues_jour": exp_fraud_day,
              "justification": (f"Volume moyen ≈ {daily_volume:,.0f} transactions/jour × taux de fraude "
                                f"d'entraînement {ytr.mean():.3%} ≈ {exp_fraud_day:.1f} fraudes/jour. "
                                f"On dimensionne l'équipe pour examiner {config.BUDGET_FACTOR:g}× ce volume, "
                                f"soit k = {k_per_day} alertes/jour (≈ {k_test:,} sur la période de test).")}

    # --- Déséquilibre : comparaison en validation (LightGBM) ---
    _log(progress, "4/7 Comparaison des stratégies de déséquilibre (validation)…")
    imb_rows, imb_models = [], {}
    for label, strat in IMBALANCE_STRATEGIES.items():
        m = fit_lgbm_strategy(strat, Xtr, ytr)
        s = m.predict_proba(Xva)[:, 1]
        p_k, r_k = precision_recall_at_k(yva, s, k_val)
        _, cost, _ = optimal_threshold(yva, s, amt[va], review_cost)
        imb_rows.append({"Stratégie": label, "PR-AUC": average_precision_score(yva, s),
                         f"Précision@{k_val}": p_k, f"Rappel@{k_val}": r_k, "Coût minimal ($)": cost})
        imb_models[label] = m
    # Critère de choix = coût métier minimal en validation (objectif final), PR-AUC en départage
    imb = pd.DataFrame(imb_rows).sort_values(["Coût minimal ($)", "PR-AUC"], ascending=[True, False])
    best_strat = imb.iloc[0]["Stratégie"]
    _log(progress, f"    Stratégie retenue : {best_strat}")

    # --- Tous les modèles ---
    _log(progress, "5/7 Entraînement : supervisé + non supervisé…")
    models = {f"LightGBM ({best_strat})": imb_models[best_strat],
              "Régression logistique pondérée": fit_logreg(Xtr, ytr)}
    for label, kind in UNSUPERVISED.items():
        _log(progress, f"    {label}…")
        models[label] = UnsupervisedScorer(kind).fit(Xtr)

    def score(m, Xp):
        return m.predict_proba(Xp)[:, 1] if hasattr(m, "predict_proba") else m.score(Xp)

    _log(progress, "6/7 Évaluation, seuils par coût métier…")
    val_frame = df.iloc[va][RAW_KEEP].reset_index(drop=True)
    test_frame = df.iloc[te][RAW_KEEP].reset_index(drop=True)
    rows, pr_curves, thresholds, cost_curves = [], {}, {}, {}
    for name, m in models.items():
        sv_, st_ = score(m, Xva), score(m, Xte)
        val_frame[name], test_frame[name] = sv_, st_
        thr, _, curve = optimal_threshold(yva, sv_, amt[va], review_cost)
        thresholds[name], cost_curves[name] = thr, curve
        kind = "Non supervisé" if name in UNSUPERVISED else "Supervisé"
        for split_name, yy, ss, aa, kk in [("Validation", yva, sv_, amt[va], k_val), ("Test", yte, st_, amt[te], k_test)]:
            r = metrics_at(yy, ss, aa, thr, kk, review_cost)
            rows.append({"Modèle": name, "Type": kind, "Jeu": split_name, "Seuil": thr, **r})
        pr_curves[name] = pr_curve_points(yte, st_)
    comparison = pd.DataFrame(rows)

    main_name = f"LightGBM ({best_strat})"
    main_model = models[main_name]

    _log(progress, "7/7 Explicabilité SHAP…")
    explainer = get_explainer(main_model)
    rng = np.random.default_rng(config.RANDOM_STATE)
    fraud_idx = np.where(yte == 1)[0]
    other = rng.choice(len(te), size=min(len(te), config.SHAP_SAMPLE), replace=False)
    samp = np.unique(np.concatenate([other, rng.choice(fraud_idx, size=min(len(fraud_idx), 500), replace=False)]))
    Xs = Xte.iloc[samp]
    sv = shap_values(explainer, Xs)
    shap_imp = (pd.DataFrame({"variable": FEATURES, "libellé": [FEATURE_LABELS[f] for f in FEATURES],
                              "mean_abs_shap": np.abs(sv).mean(0)})
                .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    plt.figure()
    shap.summary_plot(sv, Xs.rename(columns=FEATURE_LABELS), max_display=15, show=False, plot_size=(9, 6.5))
    plt.title("SHAP – variables les plus influentes (beeswarm, jeu de test)")
    plt.tight_layout()
    plt.savefig(config.ART_DIR / "shap_beeswarm.png", dpi=130)
    plt.close("all")

    # --- Historique des cartes (30 derniers jours) pour l'API ---
    last = df.ts.max()
    hist = df[df.ts >= last - pd.Timedelta(days=30)][
        ["ts", "cc_num", "amt", "category", "gender", "lat", "long", "city_pop", "dob", "merch_lat", "merch_long"]]

    bundle = {"model": main_model, "model_name": main_name, "strategy": best_strat, "features": FEATURES,
              "threshold": thresholds[main_name], "review_cost": review_cost, "budget": budget,
              "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"), "source": source_desc}
    joblib.dump(bundle, config.ART_DIR / "model_bundle.joblib", compress=3)
    joblib.dump(hist.reset_index(drop=True), config.ART_DIR / "card_history.joblib", compress=3)
    results = {
        "source": source_desc, "eda": eda, "split": split_info, "budget": budget, "imbalance": imb,
        "best_strategy": best_strat, "main_model": main_name, "comparison": comparison,
        "pr_curves": pr_curves, "thresholds": thresholds, "cost_curves": cost_curves,
        "review_cost": review_cost, "shap_importance": shap_imp,
        "val_frame": val_frame, "test_frame": test_frame, "X_test": Xte.reset_index(drop=True),
        "duration_s": time.time() - t0,
    }
    joblib.dump(results, config.ART_DIR / "results.joblib", compress=3)
    summary = comparison[comparison.Jeu == "Test"][["Modèle", "PR-AUC", "ROC-AUC", "Précision@k", "Rappel@k",
                                                     "F2", "MCC", "Coût total"]]
    (config.ART_DIR / "metrics_test.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    _log(progress, f"Terminé en {time.time() - t0:.0f} s.")
    return results
