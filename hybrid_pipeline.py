"""
hybrid_pipeline.py
──────────────────
Pipeline Hybrid Bertingkat: Naive Bayes → Decision Tree → KNN
from scratch (numpy only).

ARSITEKTUR TIGA TAHAP:
  Stage 1 — Naive Bayes  : Pre-filter probabilistik
  Stage 2 — Decision Tree: Klasifikasi berbasis aturan IF-THEN
  Stage 3 — KNN          : Validasi berbasis kemiripan historis

LOGIKA ALUR:
  ┌─────────────────────────────────────────────────────┐
  │ Untuk setiap sampel X[i]:                          │
  │                                                     │
  │ Stage 1 (NB): Hitung P(Buy|x) dan P(Sell|x)       │
  │   → Jika confidence < threshold (0.60):            │
  │     LANGSUNG output prediksi NB (tidak lanjut)     │
  │   → Jika confidence ≥ threshold: lanjut Stage 2   │
  │                                                     │
  │ Stage 2 (DT): Traversal pohon → dapat label DT     │
  │   → Jika NB ≠ DT: gunakan DT (lebih definitif)    │
  │   → Jika NB = DT: lanjut ke Stage 3               │
  │                                                     │
  │ Stage 3 (KNN): Cari k tetangga historis paling mirip│
  │   → Jika KNN = DT: konfirmasi final                │
  │   → Jika KNN ≠ DT: KNN menang (validator akhir)   │
  └─────────────────────────────────────────────────────┘
"""

import numpy as np


class HybridPipeline:
    """
    Pipeline Hybrid Bertingkat tiga model.

    Parameter:
      nb_model             : instance GaussianNaiveBayes (sudah di-fit)
      dt_model             : instance DecisionTree (sudah di-fit)
      knn_model            : instance KNearestNeighbors (sudah di-fit)
      confidence_threshold : batas minimum confidence NB (default 0.60)
    """

    def __init__(self, nb_model, dt_model, knn_model, confidence_threshold=0.60):
        self.nb = nb_model
        self.dt = dt_model
        self.knn = knn_model
        self.threshold = confidence_threshold

    def predict(self, X):
        """
        Jalankan pipeline hybrid tiga tahap secara batch.

        Langkah implementasi:
          1. Jalankan NB.predict_proba() untuk seluruh X sekaligus
          2. Pisahkan sampel: confidence rendah vs tinggi
          3. Untuk confident samples → jalankan DT.predict()
          4. Pisahkan: NB=DT (sepakat) vs NB≠DT (tidak sepakat)
          5. Untuk yang tidak sepakat → pakai DT
          6. Untuk yang sepakat → jalankan KNN.predict() sebagai validator
          7. Gabungkan semua prediksi

        Return: ndarray shape (n_samples,) berisi 0 (Sell) atau 1 (Buy)
        """
        n = len(X)
        final_preds = np.full(n, -1, dtype=np.int32)

        # ═══════════════════════════════════════════════════════
        # STAGE 1: Naive Bayes — Pre-filter probabilistik
        # ═══════════════════════════════════════════════════════
        print(f"    [Hybrid] Stage 1: Naive Bayes pada {n:,} sampel...")
        nb_probs = self.nb.predict_proba(X)  # (n, 2)
        nb_conf = nb_probs.max(axis=1)  # confidence tertinggi
        nb_preds = nb_probs.argmax(axis=1)  # prediksi kelas NB

        # Sampel dengan confidence rendah → langsung output NB
        low_mask = nb_conf < self.threshold
        high_mask = ~low_mask
        final_preds[low_mask] = nb_preds[low_mask]

        n_low = int(low_mask.sum())
        n_high = int(high_mask.sum())
        print(
            f"    [Hybrid]   Confidence < {self.threshold}: {n_low:,} sampel "
            f"→ langsung prediksi NB"
        )
        print(
            f"    [Hybrid]   Confidence ≥ {self.threshold}: {n_high:,} sampel "
            f"→ lanjut ke Stage 2"
        )

        if n_high == 0:
            print("    [Hybrid]   Semua sampel ditangani NB. Stage 2 & 3 dilewati.")
            return final_preds

        # ═══════════════════════════════════════════════════════
        # STAGE 2: Decision Tree — Klasifikasi berbasis aturan
        # ═══════════════════════════════════════════════════════
        print(
            f"    [Hybrid] Stage 2: Decision Tree pada {n_high:,} sampel confident..."
        )
        X_high = X[high_mask]
        nb_p_high = nb_preds[high_mask]
        dt_preds = self.dt.predict(X_high)

        # Pisahkan: sepakat vs tidak sepakat
        agree_mask = nb_p_high == dt_preds
        disagree_mask = ~agree_mask
        n_agree = int(agree_mask.sum())
        n_disagree = int(disagree_mask.sum())

        # Indeks global untuk sampel confident
        high_indices = np.where(high_mask)[0]

        # NB ≠ DT → pakai DT (lebih rule-based & interpretable)
        final_preds[high_indices[disagree_mask]] = dt_preds[disagree_mask]
        print(f"    [Hybrid]   NB ≠ DT: {n_disagree:,} sampel → pakai DT")
        print(f"    [Hybrid]   NB = DT: {n_agree:,} sampel → lanjut Stage 3")

        if n_agree == 0:
            print("    [Hybrid]   Tidak ada sampel sepakat. Stage 3 dilewati.")
            return final_preds

        # ═══════════════════════════════════════════════════════
        # STAGE 3: KNN — Validator berbasis kemiripan historis
        # ═══════════════════════════════════════════════════════
        print(f"    [Hybrid] Stage 3: KNN validasi {n_agree:,} sampel sepakat...")
        X_agree = X_high[agree_mask]
        dt_agree = dt_preds[agree_mask]
        knn_preds = self.knn.predict(X_agree)

        # KNN = DT → konfirmasi; KNN ≠ DT → KNN menang (validator akhir)
        n_confirm = int((knn_preds == dt_agree).sum())
        n_override = n_agree - n_confirm
        print(f"    [Hybrid]   KNN konfirmasi DT: {n_confirm:,} sampel")
        print(f"    [Hybrid]   KNN override DT  : {n_override:,} sampel")

        final_vote = np.where(knn_preds == dt_agree, dt_agree, knn_preds)
        final_preds[high_indices[agree_mask]] = final_vote

        # Ringkasan
        buy_count = int((final_preds == 1).sum())
        sell_count = int((final_preds == 0).sum())
        print(
            f"    [Hybrid] Prediksi final: " f"Buy={buy_count:,} | Sell={sell_count:,}"
        )

        return final_preds
