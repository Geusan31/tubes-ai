"""
linear_regression.py
────────────────────
Ridge Regression (L2) dari scratch — numpy only.

Menggantikan Naive Bayes untuk task regresi.

Formula:
  Normal Equation dengan regularisasi L2:
    β = (X^T X + λI)^{-1} X^T y

  Prediksi:
    ŷ = X_bias @ β

Kenapa Ridge bukan OLS biasa?
  - OLS (λ=0) tidak stabil jika fitur berkorelasi tinggi (singular matrix)
  - Lagged Close features sangat berkorelasi → Ridge mencegah overfitting
"""

import numpy as np


class RidgeRegression:
    """
    Ridge Regression (Ordinary Least Squares + L2 regularization).

    Parameter:
      alpha : koefisien regularisasi L2 (default 1.0)
              semakin besar → semakin smooth, semakin kecil → mendekati OLS
    """

    def __init__(self, alpha=1.0):
        self.alpha   = alpha
        self.coef_   = None
        self.intercept_ = None
        self._beta   = None   # koefisien termasuk bias

    def fit(self, X, y):
        """
        Hitung β menggunakan normal equation:
          β = (X_b^T X_b + λI)^{-1} X_b^T y

        X_b = [1 | X]  (kolom bias ditambahkan di kiri)
        λI  = regularisasi (baris/kolom pertama dibiarkan 0 agar bias tidak diregularisasi)

        Langkah:
          1. Tambah kolom bias (semua 1) ke X
          2. Bangun matrix regularisasi λI (kecuali pojok [0,0])
          3. Hitung (X_b^T X_b + λI)
          4. Invert matrix tersebut
          5. Hitung β = hasil_invert @ X_b^T @ y
        """
        n_samples, n_features = X.shape
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Tambah kolom bias
        X_b = np.column_stack([np.ones(n_samples), X])   # (n, 1+p)

        # Regularisasi: λI, tapi bias tidak diregularisasi → pojok [0,0] = 0
        reg = self.alpha * np.eye(n_features + 1)
        reg[0, 0] = 0.0

        # Normal equation: β = (X_b^T X_b + λI)^{-1} X_b^T y
        XtX      = X_b.T @ X_b                   # (1+p, 1+p)
        XtX_reg  = XtX + reg                      # tambah regularisasi
        Xty      = X_b.T @ y                      # (1+p,)
        self._beta = np.linalg.solve(XtX_reg, Xty)  # lebih stabil dari inv()

        self.intercept_ = self._beta[0]
        self.coef_       = self._beta[1:]

        n_fit = n_samples
        print(f"    [LR] Training selesai: {n_fit:,} sampel, {n_features} fitur")
        print(f"    [LR] alpha={self.alpha} | intercept={self.intercept_:.4f}")

    def predict(self, X):
        """
        Prediksi: ŷ = X_b @ β = intercept + X @ coef_
        """
        X    = np.array(X, dtype=np.float64)
        X_b  = np.column_stack([np.ones(len(X)), X])
        return X_b @ self._beta
