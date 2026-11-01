# 🛡️ Application web interactive de détection de fraude explicable

**Projet Challenge 2 – Data Mining · ENSI II3-IA / II3-IF · 2026-2027**
Jeu de données : **n°1 – Credit Card Transactions Fraud Detection (simulateur Sparkov)**, cible `is_fraud`.

## Démarrage rapide (Windows)

```bat
1_installer.bat        :: installe les dépendances
2_entrainer.bat        :: entraîne tous les modèles (≈ 2 min sur le jeu de démo)
3_lancer_app.bat       :: ouvre l'application  -> http://localhost:8501
4_lancer_api.bat       :: lance l'API          -> http://127.0.0.1:8000/docs
```
Équivalent en ligne de commande : `pip install -r requirements.txt`, `python train.py`,
`streamlit run app.py`, `uvicorn api.main:app --reload`, tests : `python -m pytest -q tests`.

## Données
Trois façons de charger les données (aussi disponibles dans la barre latérale de l'application) :
1. **Import de fichiers** : placer `fraudTrain.csv` et `fraudTest.csv` (Kaggle : *kartik2112/fraud-detection*) dans `data/`, puis `python train.py`.
2. **Téléchargement automatique** : `python train.py --source kaggle` (via `kagglehub`, compte Kaggle requis).
3. **Démo hors ligne** : `python train.py --source synthetic` – jeu simulé au même format Sparkov.

Le jeu complet compte ≈ 1,85 M transactions ; sur un PC modeste : `python train.py --card-fraction 0.3`
(échantillonnage par **carte**, pour garder des historiques complets).

## Correspondance avec le cahier des charges

| Module | Réalisation | Fichier |
|---|---|---|
| 1 – Données | Chargement (Kaggle / import / synthétique), taux de fraude, montants par classe, évolution mensuelle, fraude par heure / catégorie / jour | `src/data.py`, onglet 1 |
| 2 – Préparation | Heure en codage cyclique, jour ; vélocité 1 h / 24 h, montant / moyenne 30 j, temps depuis la transaction précédente, distance ; **séparation chronologique 60/20/20** ; déséquilibre traité sur l'entraînement seul (aucune / pondération / sous-échantillonnage) et choisi en validation | `src/features.py`, `src/pipeline.py`, onglet 2 |
| 3 – Modélisation | Supervisé : LightGBM, régression logistique pondérée. Non supervisé : **Isolation Forest**, **LOF**, autoencodeur. Tableau comparatif | `src/models.py`, onglet 3 |
| 4 – Évaluation & seuil | PR-AUC, ROC-AUC, précision@k, rappel@k (k justifié), F2 et MCC au seuil ; courbes PR superposées ; seuil minimisant *montants des fraudes manquées + coût d'analyse* sur la validation puis confirmé sur le test ; curseurs seuil et budget en direct | `src/evaluate.py`, onglets 3-4 |
| 5 – Explainable AI | SHAP beeswarm global ; waterfall + **codes motifs** (3 à 5) par alerte | `src/explain.py`, onglet 5 |
| 6 – API | FastAPI : `/health`, `/model/info`, `/predict`, `/predict/batch`, `/explain` | `api/main.py`, onglet 6 |

## Choix méthodologiques clés
- **Pas de fuite de données** : les variables comportementales n'utilisent que les transactions *strictement antérieures* de la carte (fenêtres `closed='left'`). Un test automatique vérifie que supprimer le futur ne modifie aucune variable. La même fonction sert à l'entraînement et à l'API.
- **Validation chronologique** : on entraîne sur le passé, on règle le seuil sur la période suivante, on confirme sur la plus récente.
- **Budget k** : k/jour = 2 × (volume journalier × taux de fraude) → l'équipe peut examiner deux fois plus de dossiers qu'il n'y a de fraudes attendues. Modifiable dans l'application.
- **Coût métier** : coût = Σ montants des fraudes manquées + 5 $ × nombre d'alertes (coût d'analyse paramétrable).

## Structure
```
app.py              application Streamlit (6 onglets)
train.py            entraînement en ligne de commande
config.py           paramètres (coût d'analyse, budget, échantillonnage)
src/                data, features, models, evaluate, explain, pipeline
api/main.py         API FastAPI
tests/              tests (fuite, séparation, seuil, API)
data/               CSV Sparkov (à ajouter)
artifacts/          modèles et résultats générés
```
