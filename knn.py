"""
knn.py
──────
K-Nearest Neighbor — dari scratch (numpy only).
Mendukung dua mode:
  task='regressor'  : prediksi = mean target k tetangga terdekat
  task='classifier' : prediksi = kelas mayoritas k tetangga terdekat

Formula Euclidean Distance (vectorized):
  ||a-b||² = ||a||² + ||b||² - 2(a·b)
  → np.argpartition O(n) lebih cepat dari argsort O(n log n)
"""

import numpy as np


class KNearestNeighbors:
    """
    K-Nearest Neighbor Regressor / Classifier.

    Parameter:
      k          : jumlah tetangga (untuk classifier: ganjil agar tidak seri)
      batch_size : ukuran batch prediksi (efisiensi memori)
      task       : 'regressor' atau 'classifier'
    """

    def __init__(self, k=7, batch_size=512, task='regressor'):
        self.task       = task
        self.batch_size = batch_size
        if task == 'classifier' and k % 2 == 0:
            k += 1   # pastikan ganjil untuk classifier
        self.k = k

    def fit(self, X, y):
        self.X_train = X.astype(np.float64)
        if self.task == 'regressor':
            self.y_train = y.astype(np.float64)
        else:
            self.y_train = y.astype(np.int32)
        print(f"    [KNN] Data training tersimpan: {len(X):,} sampel, "
              f"{X.shape[1]} fitur, k={self.k}, task={self.task}")
        print(f"    [KNN] Lazy learner — semua komputasi terjadi saat predict()")

    def predict(self, X):
        """
        Prediksi batch menggunakan jarak Euclidean vectorized.

        Regresi:   prediksi = mean(y[k_tetangga])
        Klasifikasi: prediksi = argmax(bincount(y[k_tetangga]))
        """
        X        = X.astype(np.float64)
        n        = len(X)
        dtype    = np.float64 if self.task == 'regressor' else np.int32
        preds    = np.empty(n, dtype=dtype)
        sq_train = np.sum(self.X_train ** 2, axis=1)   # (n_train,)

        for start in range(0, n, self.batch_size):
            end      = min(start + self.batch_size, n)
            batch    = X[start:end]

            # ||a-b||² = ||a||² + ||b||² - 2(a·b)
            sq_batch = np.sum(batch ** 2, axis=1, keepdims=True)
            dot      = batch @ self.X_train.T
            dist_sq  = np.clip(sq_batch + sq_train - 2 * dot, 0, None)

            # k tetangga terdekat — O(n) via argpartition
            k_idx = np.argpartition(dist_sq, self.k, axis=1)[:, :self.k]

            for i, indices in enumerate(k_idx):
                neighbors = self.y_train[indices]
                if self.task == 'regressor':
                    preds[start + i] = float(np.mean(neighbors))
                else:
                    preds[start + i] = int(np.bincount(neighbors).argmax())

        return preds
