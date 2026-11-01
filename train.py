"""Entraînement complet en ligne de commande.

Exemples :
  python train.py                                  # CSV de data/ s'ils existent, sinon jeu synthétique
  python train.py --source kaggle                  # téléchargement automatique (kagglehub)
  python train.py --source csv --files data/fraudTrain.csv data/fraudTest.csv --card-fraction 0.3
  python train.py --source synthetic
"""
import argparse

import config
from src.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "csv", "kaggle", "synthetic"])
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--card-fraction", type=float, default=config.CARD_FRACTION)
    ap.add_argument("--review-cost", type=float, default=config.REVIEW_COST)
    a = ap.parse_args()
    res = run(a.source, a.files, a.card_fraction, a.review_cost)
    t = res["comparison"]
    print("\n=== Comparaison sur le TEST ===")
    print(t[t.Jeu == "Test"][["Modèle", "PR-AUC", "ROC-AUC", "Précision@k", "Rappel@k", "F2", "MCC",
                              "Coût total"]].round(4).to_string(index=False))
