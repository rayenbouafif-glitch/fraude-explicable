"""Configuration globale du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# --- Séparation chronologique ---
SPLIT = (0.60, 0.20, 0.20)          # entraînement / validation / test

# --- Coût métier ---
REVIEW_COST = 5.0                   # coût d'analyse d'une alerte ($), modifiable dans l'app
# Coût d'une fraude manquée = son montant (amt)

# --- Budget d'alertes k ---
# k/jour = BUDGET_FACTOR x (nb moyen de fraudes attendues par jour)
# => on se donne la capacité d'examiner 2x plus de dossiers qu'il n'y a de fraudes
BUDGET_FACTOR = 2.0

# --- Échantillonnage (pour PC modestes) ---
CARD_FRACTION = 1.0                 # fraction des cartes conservées (échantillonnage par carte => séquences intactes)
LOF_SAMPLE = 20_000                 # LOF est coûteux : entraîné sur un sous-échantillon
AE_SAMPLE = 60_000                  # autoencodeur : sous-échantillon d'entraînement
SHAP_SAMPLE = 3_000                 # lignes pour le beeswarm

KAGGLE_DATASET = "kartik2112/fraud-detection"   # Sparkov (fraudTrain.csv / fraudTest.csv)
