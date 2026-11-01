"""Application web interactive - Détection de fraude explicable.

Lancement :  streamlit run app.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

import config
from src.data import CATEGORIES
from src.evaluate import precision_recall_at_k
from src.explain import get_explainer, shap_values, reason_codes, explanation_object
from src.features import FEATURE_LABELS
from src.pipeline import run

st.set_page_config(page_title="Détection de fraude explicable", page_icon="🛡️", layout="wide")
PURPLE = "#8b5cf6"
TEMPLATE = "plotly_dark"
JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

st.markdown("""<style>
div[data-testid="stMetric"]{background:rgba(139,92,246,.08);border:1px solid rgba(139,92,246,.35);
border-radius:12px;padding:10px 14px}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chargement des artefacts
# ---------------------------------------------------------------------------
def _mtime():
    p = config.ART_DIR / "results.joblib"
    return p.stat().st_mtime if p.exists() else 0


@st.cache_resource(show_spinner="Chargement des résultats…")
def load_artifacts(_mt):
    res = joblib.load(config.ART_DIR / "results.joblib")
    bundle = joblib.load(config.ART_DIR / "model_bundle.joblib")
    return res, bundle, get_explainer(bundle["model"])


# ---------------------------------------------------------------------------
# Barre latérale : (ré)entraînement
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Fraude explicable")
    st.caption("Data Mining · ENSI · Projet Challenge 2 — jeu Sparkov (carte bancaire)")
    with st.expander("⚙️ Charger les données & entraîner", expanded=_mtime() == 0):
        src = st.radio("Source", ["CSV du dossier data/ (ou synthétique)", "Importer des CSV",
                                  "Téléchargement Kaggle", "Synthétique (démo)"])
        files = None
        if src == "Importer des CSV":
            files = st.file_uploader("fraudTrain.csv / fraudTest.csv", type="csv", accept_multiple_files=True)
        frac = st.slider("Fraction des cartes conservées", 0.05, 1.0, config.CARD_FRACTION, 0.05,
                         help="Échantillonnage par carte pour accélérer sur un PC modeste.")
        rc = st.number_input("Coût d'analyse d'une alerte ($)", 0.0, 500.0, config.REVIEW_COST, 0.5)
        if st.button("🚀 Lancer le pipeline", type="primary", use_container_width=True):
            source = {"Importer des CSV": "csv", "Téléchargement Kaggle": "kaggle",
                      "Synthétique (démo)": "synthetic"}.get(src, "auto")
            if source == "csv" and not files:
                st.error("Ajoutez au moins un fichier CSV.")
            else:
                with st.status("Pipeline en cours…", expanded=True) as status:
                    try:
                        run(source, files, frac, rc, progress=status.write)
                        status.update(label="Entraînement terminé ✅", state="complete")
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as e:  # message clair pour l'utilisateur
                        status.update(label="Erreur", state="error")
                        st.exception(e)

if _mtime() == 0:
    st.info("Aucun modèle entraîné. Utilisez le panneau de gauche ou lancez `python train.py`.")
    st.stop()

res, bundle, explainer = load_artifacts(_mtime())
MAIN = res["main_model"]
with st.sidebar:
    st.divider()
    st.markdown(f"**Source :** {res['source']}")
    st.markdown(f"**Modèle déployé :** {MAIN}")
    st.markdown(f"**Seuil coût-optimal :** `{bundle['threshold']:.4f}`")
    st.caption(f"Entraîné le {bundle['trained_at']} en {res['duration_s']:.0f} s")

tabs = st.tabs(["📊 1 · Données", "🧱 2 · Préparation", "🤖 3 · Modèles", "🎚️ 4 · Seuil & coût",
                "🔍 5 · Explicabilité", "🔌 6 · API"])

# ---------------------------------------------------------------------------
# 1. Données
# ---------------------------------------------------------------------------
with tabs[0]:
    e = res["eda"]
    c = st.columns(5)
    c[0].metric("Transactions", f"{e['n_rows']:,}")
    c[1].metric("Cartes", f"{e['n_cards']:,}")
    c[2].metric("Fraudes", f"{e['n_fraud']:,}")
    c[3].metric("Taux de fraude", f"{e['fraud_rate']:.3%}")
    c[4].metric("Période", f"{e['period'][0][:10]} → {e['period'][1][:10]}")

    l, r = st.columns(2)
    amt = pd.concat([pd.DataFrame({"Montant ($)": e["amt_legit"], "Classe": "Légitime"}),
                     pd.DataFrame({"Montant ($)": e["amt_fraud"], "Classe": "Fraude"})])
    fig = px.histogram(amt, x="Montant ($)", color="Classe", log_x=False, histnorm="probability density",
                       barmode="overlay", nbins=120, template=TEMPLATE, opacity=0.65,
                       color_discrete_map={"Légitime": "#64748b", "Fraude": PURPLE},
                       title="Distribution des montants par classe (densité)")
    fig.update_xaxes(range=[0, np.quantile(e["amt_fraud"], 0.99) if len(e["amt_fraud"]) else 1500])
    l.plotly_chart(fig, use_container_width=True)
    m = e["by_month"]
    fig = go.Figure()
    fig.add_bar(x=m["mois"], y=m["count"], name="Transactions", marker_color="#334155", yaxis="y2")
    fig.add_scatter(x=m["mois"], y=m["mean"] * 100, name="Taux de fraude (%)", line=dict(color=PURPLE, width=3))
    fig.update_layout(template=TEMPLATE, title="Évolution du taux de fraude dans le temps",
                      yaxis=dict(title="Taux de fraude (%)"),
                      yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=-0.2))
    r.plotly_chart(fig, use_container_width=True)

    l, r = st.columns(2)
    h = e["by_hour"]
    l.plotly_chart(px.bar(h, x="heure", y=h["mean"] * 100, template=TEMPLATE, color_discrete_sequence=[PURPLE],
                          labels={"y": "Taux de fraude (%)", "heure": "Heure"}, title="Taux de fraude par heure"),
                   use_container_width=True)
    cat = e["by_category"]
    r.plotly_chart(px.bar(cat, y="category", x=cat["mean"] * 100, orientation="h", template=TEMPLATE,
                          color_discrete_sequence=[PURPLE], title="Taux de fraude par catégorie",
                          labels={"x": "Taux de fraude (%)", "category": ""}).update_yaxes(autorange="reversed"),
                   use_container_width=True)
    d = e["by_dow"].assign(jour=lambda x: x["jour"].map(dict(enumerate(JOURS))))
    st.plotly_chart(px.bar(d, x="jour", y=d["mean"] * 100, template=TEMPLATE, color_discrete_sequence=["#a78bfa"],
                           labels={"y": "Taux de fraude (%)", "jour": ""}, title="Taux de fraude par jour",
                           height=300), use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Préparation
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Variables construites (sans fuite de données)")
    st.markdown("""
| Famille | Variables | Construction |
|---|---|---|
| Temporelles | `hour_sin`, `hour_cos`, `dow`, `is_night` | Codage cyclique de l'heure (23 h et 0 h sont proches), jour de la semaine |
| Vélocité | `n_tx_1h`, `n_tx_24h` | Nombre de transactions **antérieures** de la même carte sur 1 h et 24 h |
| Écart au profil | `amt_ratio_30d` | Montant / moyenne des montants de la carte sur les 30 **derniers** jours |
| Récence | `hours_since_last` | Temps écoulé depuis la transaction précédente de la carte |
| Géographie | `distance_km` | Distance haversine domicile ↔ marchand |
| Porteur / contexte | `age`, `gender_M`, `log_city_pop`, `amt`, `log_amt`, `cat_*` | Âge au moment de la transaction, one-hot de la catégorie |
""")
    st.success("Anti-fuite : fenêtres glissantes `closed='left'` → seule l'information disponible **avant** la "
               "transaction est utilisée. La même fonction calcule les variables à l'entraînement et dans l'API. "
               "Un test automatique vérifie que supprimer les transactions futures ne change aucune variable.")

    st.subheader("Séparation chronologique 60 / 20 / 20")
    sp = pd.DataFrame(res["split"]).T.reset_index(names="Jeu")
    sp["taux"] = (sp["taux"].astype(float) * 100).round(3).astype(str) + " %"
    st.dataframe(sp, use_container_width=True, hide_index=True)
    fig = go.Figure()
    for (name, s), col in zip(res["split"].items(), ["#6d28d9", "#a78bfa", "#ddd6fe"]):
        fig.add_trace(go.Bar(y=["Période"], x=[(pd.Timestamp(s["fin"]) - pd.Timestamp(s["début"])).total_seconds() * 1000],
                             base=[s["début"]], orientation="h", name=name, marker_color=col))
    fig.update_layout(template=TEMPLATE, barmode="stack", height=180, margin=dict(t=10, b=10),
                      xaxis_type="date", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stratégie contre le déséquilibre (appliquée au seul entraînement, comparée en validation)")
    st.dataframe(res["imbalance"].style.format(precision=4).highlight_min(subset=["Coût minimal ($)"],
                 color="#4c1d95"), use_container_width=True, hide_index=True)
    st.markdown(f"➡️ **Retenue : {res['best_strategy']}** — coût métier minimal en validation "
                "(objectif final de l'outil), la PR-AUC servant de critère secondaire.")
    st.subheader("Budget d'alertes k")
    st.info(res["budget"]["justification"])

# ---------------------------------------------------------------------------
# 3. Modèles
# ---------------------------------------------------------------------------
with tabs[2]:
    jeu = st.radio("Jeu", ["Test", "Validation"], horizontal=True)
    cmp_ = res["comparison"]
    t = cmp_[cmp_.Jeu == jeu].drop(columns=["Jeu"]).sort_values("PR-AUC", ascending=False)
    cols = ["Modèle", "Type", "PR-AUC", "ROC-AUC", "Précision@k", "Rappel@k", "F2", "MCC", "Alertes",
            "Précision", "Rappel", "Coût total"]
    k = res["budget"]["k_test" if jeu == "Test" else "k_validation"]
    st.caption(f"k = {k:,} alertes · seuils choisis sur la validation en minimisant le coût métier, "
               "puis appliqués tels quels au test.")
    st.dataframe(t[cols].style.format({c: "{:.4f}" for c in cols[2:9] if c != "Alertes"} |
                                      {"Précision": "{:.3f}", "Rappel": "{:.3f}", "Coût total": "{:,.0f} $"})
                 .background_gradient(subset=["PR-AUC"], cmap="Purples"),
                 use_container_width=True, hide_index=True)

    fig = go.Figure()
    palette = [PURPLE, "#22d3ee", "#f59e0b", "#f43f5e", "#10b981"]
    for (name, cur), col in zip(res["pr_curves"].items(), palette):
        ap = cmp_[(cmp_.Modèle == name) & (cmp_.Jeu == "Test")]["PR-AUC"].iloc[0]
        fig.add_scatter(x=cur["recall"], y=cur["precision"], name=f"{name} (AP={ap:.3f})", line=dict(color=col))
    base = res["split"]["Test"]["taux"]
    fig.add_hline(y=base, line_dash="dot", annotation_text=f"Hasard ({base:.2%})")
    fig.update_layout(template=TEMPLATE, title="Courbes précision-rappel superposées (test)",
                      xaxis_title="Rappel", yaxis_title="Précision", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Lecture :** le supervisé exploite les étiquettes et domine nettement ; les méthodes non "
                "supervisées détectent des transactions *atypiques*, ce qui n'est pas toujours synonyme de fraude, "
                "mais restent utiles pour des schémas de fraude nouveaux, sans étiquettes.")

# ---------------------------------------------------------------------------
# 4. Seuil & coût (interactif)
# ---------------------------------------------------------------------------
with tabs[3]:
    a, b_, c_ = st.columns([2, 1, 1])
    model_name = a.selectbox("Modèle", list(res["pr_curves"].keys()), index=0)
    jeu4 = b_.radio("Jeu", ["Test", "Validation"], horizontal=True, key="jeu4")
    review_cost = c_.number_input("Coût d'analyse / alerte ($)", 0.0, 500.0, float(res["review_cost"]), 0.5)
    fr = res["test_frame" if jeu4 == "Test" else "val_frame"]
    s, y, amt = fr[model_name].to_numpy(), fr["is_fraud"].to_numpy(), fr["amt"].to_numpy()
    n_days = max(1.0, (fr.ts.max() - fr.ts.min()).total_seconds() / 86400)

    thr_opt = res["thresholds"][model_name]
    q = 1 - np.geomspace(0.5, 1e-5, 300)
    cands = np.unique(np.concatenate([np.quantile(s, q), [thr_opt]]))
    l, r = st.columns(2)
    i_thr = l.select_slider("Seuil d'alerte", options=list(range(len(cands))),
                            value=int(np.searchsorted(cands, thr_opt)),
                            format_func=lambda i: f"{cands[int(i)]:.4g}", key=f"thr_{model_name}_{jeu4}")
    thr = float(cands[int(i_thr)])
    kpd = r.slider("Budget : alertes analysables par jour", 1, max(20, res["budget"]["k_par_jour"] * 6),
                   res["budget"]["k_par_jour"])
    k = int(kpd * n_days)

    alerts = s >= thr
    tp = int((alerts & (y == 1)).sum()); fp = int((alerts & (y == 0)).sum()); fn = int((~alerts & (y == 1)).sum())
    missed = amt[(y == 1) & ~alerts].sum()
    cost = missed + review_cost * alerts.sum()
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    f2 = 5 * prec * rec / max(1e-12, 4 * prec + rec)
    p_k, r_k = precision_recall_at_k(y, s, k)
    m = st.columns(6)
    m[0].metric("Alertes", f"{alerts.sum():,}", f"{alerts.sum() / n_days:.1f}/jour", delta_color="off")
    m[1].metric("Précision", f"{prec:.1%}")
    m[2].metric("Rappel", f"{rec:.1%}")
    m[3].metric("F2", f"{f2:.3f}")
    m[4].metric("Coût total", f"{cost:,.0f} $", f"manquées {missed:,.0f} $", delta_color="off")
    m[5].metric(f"P@k / R@k (k={k:,})", f"{p_k:.1%} / {r_k:.1%}")
    if alerts.sum() / n_days > kpd:
        st.warning(f"⚠️ Ce seuil génère {alerts.sum() / n_days:.1f} alertes/jour, au-delà de la capacité "
                   f"de l'équipe ({kpd}/jour).")

    # Courbe de coût en fonction du nombre d'alertes (calcul vectorisé)
    order = np.argsort(-s, kind="mergesort")
    fa = np.where(y[order] == 1, amt[order], 0.0)
    n_al = np.arange(len(s) + 1)
    costs = (fa.sum() - np.concatenate([[0], np.cumsum(fa)])) + review_cost * n_al
    lim = int(min(len(s), max(alerts.sum(), k) * 4 + 50))
    fig = go.Figure()
    fig.add_scatter(x=n_al[:lim], y=costs[:lim], line=dict(color=PURPLE, width=3), name="Coût total")
    fig.add_vline(x=int(alerts.sum()), line_color="#f59e0b", annotation_text="Seuil courant")
    fig.add_vline(x=int(np.argmin(costs)), line_dash="dot", line_color="#10b981", annotation_text="Optimum")
    fig.add_vline(x=k, line_dash="dash", line_color="#94a3b8", annotation_text="Budget k")
    fig.update_layout(template=TEMPLATE, title="Coût métier (fraudes manquées + analyse) selon le nombre d'alertes",
                      xaxis_title="Nombre d'alertes (transactions les plus suspectes)", yaxis_title="Coût ($)",
                      height=420)
    st.plotly_chart(fig, use_container_width=True)

    l, r = st.columns([1, 2])
    cm = pd.DataFrame([[int((~alerts & (y == 0)).sum()), fp], [fn, tp]],
                      index=["Réel : légitime", "Réel : fraude"], columns=["Prédit : légitime", "Prédit : alerte"])
    l.markdown("**Matrice de confusion**")
    l.dataframe(cm, use_container_width=True)
    r.markdown("**File d'alertes (triée par score)**")
    q_ = fr[alerts].sort_values(model_name, ascending=False)
    r.dataframe(q_[["ts", "trans_num", "cc_num", "category", "amt", model_name, "is_fraud"]]
                .rename(columns={model_name: "score", "is_fraud": "fraude réelle"}).head(500),
                use_container_width=True, hide_index=True, height=300)

# ---------------------------------------------------------------------------
# 5. Explicabilité
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Explication globale — {MAIN}")
    l, r = st.columns([3, 2])
    img = config.ART_DIR / "shap_beeswarm.png"
    if img.exists():
        l.image(str(img), use_container_width=True)
    imp = res["shap_importance"].head(15)
    r.plotly_chart(px.bar(imp, x="mean_abs_shap", y="libellé", orientation="h", template=TEMPLATE,
                          color_discrete_sequence=[PURPLE], title="Importance moyenne |SHAP|",
                          labels={"mean_abs_shap": "|SHAP| moyen", "libellé": ""}, height=520)
                   .update_yaxes(autorange="reversed"), use_container_width=True)

    st.divider()
    st.subheader("Explication d'une alerte")
    tf, Xt = res["test_frame"], res["X_test"]
    thr_main = bundle["threshold"]
    kind = st.radio("Transactions", ["Alertes vraies (fraudes détectées)", "Fausses alertes",
                                     "Fraudes manquées"], horizontal=True)
    alerted = tf[MAIN] >= thr_main
    mask = {"Alertes vraies (fraudes détectées)": alerted & (tf.is_fraud == 1),
            "Fausses alertes": alerted & (tf.is_fraud == 0),
            "Fraudes manquées": ~alerted & (tf.is_fraud == 1)}[kind]
    pool = tf[mask].sort_values(MAIN, ascending=False).head(300)
    if pool.empty:
        st.info("Aucune transaction dans cette catégorie.")
    else:
        idx = st.selectbox("Transaction", pool.index,
                           format_func=lambda i: f"{tf.at[i, 'trans_num']} · {tf.at[i, 'ts']:%Y-%m-%d %H:%M} · "
                                                 f"{tf.at[i, 'category']} · {tf.at[i, 'amt']:,.2f} $ · "
                                                 f"score {tf.at[i, MAIN]:.3f}")
        x_row = Xt.loc[[idx]]
        sv = shap_values(explainer, x_row)[0]
        l, r = st.columns([3, 2])
        with l:
            plt.style.use("default")
            shap.plots.waterfall(explanation_object(explainer, x_row), max_display=12, show=False)
            fig = plt.gcf()
            fig.set_size_inches(8, 5.5)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.caption("Valeurs SHAP en log-odds : chaque barre pousse le score vers la fraude (rouge) "
                       "ou vers le légitime (bleu).")
        with r:
            row = tf.loc[idx]
            st.markdown(f"**Score : {row[MAIN]:.4f}** (seuil {thr_main:.4f}) · "
                        f"fraude réelle : {'oui' if row.is_fraud else 'non'}")
            st.markdown("**Codes motifs (3 à 5 principaux contributeurs)**")
            codes = reason_codes(x_row.iloc[0], sv, 5)
            for c in codes:
                st.markdown(f"- `{c['code']}` — {c['explication']} *(+{c['contribution_shap']:.2f})*")
            if not codes:
                st.caption("Aucune variable ne pousse vers la fraude.")
            st.markdown("**Contexte comportemental**")
            st.dataframe(pd.DataFrame({"Variable": [FEATURE_LABELS[f] for f in
                                                    ["n_tx_1h", "n_tx_24h", "amt_ratio_30d", "hours_since_last",
                                                     "distance_km"]],
                                       "Valeur": x_row.iloc[0][["n_tx_1h", "n_tx_24h", "amt_ratio_30d",
                                                                "hours_since_last", "distance_km"]].round(2).values}),
                         hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 6. API
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("API FastAPI")
    st.markdown("""
Lancer : `uvicorn api.main:app --reload` puis ouvrir **http://127.0.0.1:8000/docs**

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | État du service |
| GET | `/model/info` | Modèle, stratégie, seuil, budget |
| POST | `/predict` | Score + alerte + codes motifs d'une transaction |
| POST | `/predict/batch` | Scoring d'un lot |
| POST | `/explain` | Contributions SHAP complètes |

L'API recalcule les variables comportementales à partir de l'historique des 30 derniers jours de la carte,
avec **la même fonction** que l'entraînement.
""")
    st.markdown("**Tester ici (même code que l'API)**")
    with st.form("api_test"):
        c = st.columns(4)
        cc = c[0].number_input("cc_num", value=int(res["test_frame"].cc_num.iloc[0]), step=1, format="%d")
        dt = c[1].text_input("Date/heure", "2020-12-31 23:30:00")
        amount = c[2].number_input("Montant ($)", 0.01, 50_000.0, 950.0)
        category = c[3].selectbox("Catégorie", CATEGORIES, index=CATEGORIES.index("shopping_net"))
        go_ = st.form_submit_button("Scorer", type="primary")
    if go_:
        from api import main as api_main
        if "bundle" not in api_main.STATE:
            api_main._load()
        h = api_main.STATE["history"]
        prof = h[h.cc_num == cc].iloc[-1] if (h.cc_num == cc).any() else None
        extra = {} if prof is None else {"gender": prof.gender, "lat": prof.lat, "long": prof.long,
                                         "city_pop": int(prof.city_pop), "dob": str(prof.dob.date()),
                                         "merch_lat": prof.lat + 0.3, "merch_long": prof.long + 0.3}
        t = api_main.Transaction(cc_num=cc, trans_date_trans_time=dt, amt=amount, category=category, **extra)
        out = api_main._score(t, 5)
        (st.error if out["alerte"] else st.success)(
            f"Score {out['score']:.4f} — {'ALERTE' if out['alerte'] else 'pas d’alerte'} (risque {out['niveau_risque']})")
        st.json({k: v for k, v in out.items() if k != "variables"})
    st.code("""curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "transaction": {"cc_num": 4000000000000000, "trans_date_trans_time": "2020-12-31 23:30:00",
                  "amt": 950, "category": "shopping_net"}}'""", language="bash")
