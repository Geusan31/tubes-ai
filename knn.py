"""
knn.py
──────
K-Nearest Neighbor Classifier — from scratch (numpy only).

PROSES DETAIL TIAP TAHAP:
  1. fit()     : Simpan seluruh data training (lazy learning — tidak ada
                 model yang dibangun, semua komputasi saat prediksi)
  2. predict() : Untuk setiap sampel uji:
                 a. Hitung jarak Euclidean ke semua titik training
                 b. Ambil k tetangga terdekat (argpartition)
                 c. Majority vote → prediksi kelas

Formula Euclidean Distance:
  d(a,b) = √Σ(aᵢ - bᵢ)²

Optimasi:
  ||a-b||² = ||a||² + ||b||² - 2(a·b)
  → Dihitung secara vectorized batch (bukan loop per sampel)
  → np.argpartition O(n) lebih cepat dari np.argsort O(n log n)
"""

import numpy as np


class KNearestNeighbors:
    """
    K-Nearest Neighbor Classifier.

    Parameter:
      k          : jumlah tetangga (selalu ganjil untuk hindari draw)
      batch_size : ukuran batch saat prediksi (efisiensi memori)
    """

    def __init__(self, k=7, batch_size=512):
        if k % 2 == 0:
            k += 1           # pastikan ganjil
        self.k          = k
        self.batch_size = batch_size

    # ── TRAINING (Lazy — hanya simpan data) ───────────────────────

    def fit(self, X, y):
        """
        TAHAP 'TRAINING' — KNN adalah lazy learner.
        Tidak ada model yang dibangun. Data training hanya disimpan
        dan akan digunakan secara langsung saat prediksi.

        Langkah:
          1. Simpan X_train dan y_train ke memori
          2. Tidak ada komputasi apapun di tahap ini
        """
        self.X_train = X.astype(np.float64)
        self.y_train = y.astype(np.int32)
        print(f"    [KNN] Data training tersimpan: {len(X):,} sampel, "
              f"{X.shape[1]} fitur, k={self.k}")
        print(f"    [KNN] Catatan: KNN adalah lazy learner — "
              f"semua komputasi terjadi saat predict()")

    # ── PREDIKSI ──────────────────────────────────────────────────

    def predict(self, X):
        """
        TAHAP PREDIKSI — Untuk setiap sampel uji:

        Langkah:
          1. Bagi X menjadi batch-batch kecil (efisiensi memori)
          2. Per batch:
             a. Hitung jarak kuadrat Euclidean ke semua titik training
                menggunakan identitas: ||a-b||² = ||a||² + ||b||² - 2a·b
             b. np.argpartition untuk menemukan k indeks terdekat (O(n))
             c. Ambil label k tetangga
             d. Majority vote: kelas dengan frekuensi terbanyak

        Return: ndarray shape (n_samples,) berisi 0 (Sell) atau 1 (Buy)
        """
        X      = X.astype(np.float64)
        n      = len(X)
        preds  = np.empty(n, dtype=np.int32)

        # Pre-komputasi norm kuadrat data training (sekali saja)
        sq_train = np.sum(self.X_train ** 2, axis=1)   # (n_train,)

        for start in range(0, n, self.batch_size):
            end   = min(start + self.batch_size, n)
            batch = X[start:end]                          # (b, d)

            # ── a. Hitung jarak Euclidean (vectorized) ────────────
            # ||a-b||² = ||a||² + ||b||² - 2(a·b)
            sq_batch = np.sum(batch ** 2, axis=1, keepdims=True)  # (b, 1)
            dot      = batch @ self.X_train.T                      # (b, n_train)
            dist_sq  = np.clip(sq_batch + sq_train - 2 * dot, 0, None)
            # np.clip untuk menghilangkan nilai negatif kecil akibat
            # floating-point imprecision

            # ── b. Ambil k tetangga terdekat ──────────────────────
            # argpartition: O(n) — lebih cepat dari argsort O(n log n)
            # Hasilnya tidak tersortir, tapi kita hanya butuh k indeks
            k_idx = np.argpartition(dist_sq, self.k, axis=1)[:, :self.k]

            # ── c & d. Majority vote ──────────────────────────────
            for i, indices in enumerate(k_idx):
                neighbors          = self.y_train[indices]
                # bincount: hitung frekuensi tiap label
                preds[start + i]   = int(np.bincount(neighbors).argmax())

        return preds
