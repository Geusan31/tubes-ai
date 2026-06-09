"""
hybrid_pipeline.py
──────────────────
Pipeline Hybrid: Majority Vote dari Naive Bayes + Decision Tree + KNN
from scratch (numpy only).

ARSITEKTUR MAJORITY VOTE:
  Ketiga model memberikan suara pada SETIAP sampel.
  Kelas dengan ≥ 2 suara (dari 3) menjadi prediksi final.

  NB  : voting berbasis probabilitas (soft vote — P(Buy|x))
  DT  : voting berbasis aturan IF-THEN (hard vote — 0 atau 1)
  KNN : voting berbasis kemiripan historis (hard vote — 0 atau 1)

LOGIKA:
  ┌────────────────────────────────────────┐
  │  Untuk setiap sampel X[i]:            │
  │                                        │
  │  nb_vote  = argmax P(y|x)   → 0 or 1 │
  │  dt_vote  = DT.predict(x)   → 0 or 1 │
  │  knn_vote = KNN.predict(x)  → 0 or 1 │
  │                                        │
  │  total_votes = nb + dt + knn          │
  │  final = 1 (Buy)  jika total ≥ 2     │
  │        = 0 (Sell) jika total < 2      │
  └────────────────────────────────────────┘

KEUNGGULAN vs cascade lama:
  - Ketiga model dipakai pada 100% data (bukan 8%)
  - Tidak ada threshold confidence yang arbitrary
  - Kelemahan satu model dikompensasi dua model lainnya
  - Ketika ≥2 model sepakat → prediksi lebih reliable
"""

import numpy as np


class HybridPipeline:
    """
    Pipeline Hybrid Majority Vote tiga model.

    Parameter:
      nb_model  : instance GaussianNaiveBayes (sudah di-fit)
      dt_model  : instance DecisionTree (sudah di-fit)
      knn_model : instance KNearestNeighbors (sudah di-fit)
    """

    def __init__(self, nb_model, dt_model, knn_model, confidence_threshold=0.60):
        self.nb  = nb_model
        self.dt  = dt_model
        self.knn = knn_model
        # confidence_threshold dipertahankan untuk kompatibilitas signature

    def predict(self, X):
        """
        Majority vote: setiap model memberikan satu suara pada setiap sampel.

        Langkah:
          1. NB  → predict_proba → ambil kelas argmax (soft → hard vote)
          2. DT  → predict langsung (hard vote)
          3. KNN → predict langsung (hard vote)
          4. Jumlahkan ketiga suara (0 atau 1 per model)
          5. total ≥ 2 → Buy (1), total < 2 → Sell (0)

        Return: ndarray shape (n_samples,) berisi 0 (Sell) atau 1 (Buy)
        """
        n = len(X)
        print(f"    [Hybrid] Majority Vote pada {n:,} sampel...")

        # ── Vote 1: Naive Bayes ───────────────────────────────
        print(f"    [Hybrid]   NB prediksi...")
        nb_probs = self.nb.predict_proba(X)          # (n, 2)
        nb_votes = nb_probs.argmax(axis=1)           # (n,) → 0 atau 1

        # ── Vote 2: Decision Tree ─────────────────────────────
        print(f"    [Hybrid]   DT prediksi...")
        dt_votes = self.dt.predict(X)                # (n,) → 0 atau 1

        # ── Vote 3: KNN ───────────────────────────────────────
        print(f"    [Hybrid]   KNN prediksi...")
        knn_votes = self.knn.predict(X)              # (n,) → 0 atau 1

        # ── Majority Vote ─────────────────────────────────────
        # total_votes ∈ {0, 1, 2, 3}
        # ≥ 2 → Buy, < 2 → Sell
        total_votes = nb_votes + dt_votes + knn_votes
        final_preds = (total_votes >= 2).astype(np.int32)

        # ── Statistik voting ──────────────────────────────────
        unanimous_buy  = int((total_votes == 3).sum())
        unanimous_sell = int((total_votes == 0).sum())
        split_buy      = int((total_votes == 2).sum())
        split_sell     = int((total_votes == 1).sum())
        buy_total      = int((final_preds == 1).sum())
        sell_total     = int((final_preds == 0).sum())

        print(f"    [Hybrid]   Suara bulat Buy  (3/3): {unanimous_buy:,}")
        print(f"    [Hybrid]   Suara bulat Sell (0/3): {unanimous_sell:,}")
        print(f"    [Hybrid]   Mayoritas Buy    (2/3): {split_buy:,}")
        print(f"    [Hybrid]   Mayoritas Sell   (1/3): {split_sell:,}")
        print(f"    [Hybrid] Prediksi final: Buy={buy_total:,} | Sell={sell_total:,}")

        return final_preds
