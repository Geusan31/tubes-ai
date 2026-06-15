"""
hybrid_pipeline.py
──────────────────
Pipeline Hybrid: Naive Bayes (classifier) + Decision Tree + KNN (regressors)

ARSITEKTUR:
  NB  → prediksi arah (Buy=1 / Sell=0) dari probabilitas Bayesian
  DT  → prediksi harga Close besok (regresi, MSE criterion)
  KNN → prediksi harga Close besok (regresi, mean k-tetangga)

  Final price = weighted average DT + KNN
  Bobot disesuaikan berdasarkan R² masing-masing model pada training set.

  NB berperan sebagai sinyal arah konfirmasi — confidence-nya bisa diekstrak
  secara terpisah untuk analisis.
"""

class HybridPipeline:
    """
    Hybrid Ensemble: NB classifier + DT regressor + KNN regressor.

    Parameter:
      nb_model  : GaussianNaiveBayes (sudah di-fit pada y_direction)
      dt_model  : DecisionTree regressor (sudah di-fit pada y_price)
      knn_model : KNearestNeighbors regressor (sudah di-fit pada y_price)
      weights   : (w_dt, w_knn) bobot price ensemble; None → bobot sama
    """

    def __init__(self, nb_model, dt_model, knn_model, weights=None, **_):
        self.nb  = nb_model
        self.dt  = dt_model
        self.knn = knn_model
        self.weights = weights if weights is not None else (1.0, 1.0)

    def fit_weights(self, X_val, y_val_price):
        """Sesuaikan bobot DT dan KNN berdasarkan R² pada data validasi."""
        from evaluation import evaluate_regression

        r2_dt,  *_ = evaluate_regression(y_val_price, self.dt.predict(X_val),  "DT (weight)")
        r2_knn, *_ = evaluate_regression(y_val_price, self.knn.predict(X_val), "KNN (weight)")

        w_dt  = max(0.0, r2_dt)
        w_knn = max(0.0, r2_knn)
        total = w_dt + w_knn or 1.0
        self.weights = (w_dt / total, w_knn / total)
        print(f"    [Hybrid] Bobot price: DT={self.weights[0]:.3f} | KNN={self.weights[1]:.3f}")

    def predict(self, X):
        """
        Prediksi harga (weighted DT+KNN) dan arah (NB).

        Return: y_pred_price (float ndarray, shape n)
        """
        n = len(X)
        print(f"    [Hybrid] Prediksi pada {n:,} sampel...")

        w_dt, w_knn = self.weights

        print(f"    [Hybrid]   NB arah...")
        nb_probs = self.nb.predict_proba(X)           # (n,2)
        self._last_nb_probs = nb_probs                # simpan untuk analisis

        print(f"    [Hybrid]   DT harga...")
        dt_price  = self.dt.predict(X)

        print(f"    [Hybrid]   KNN harga...")
        knn_price = self.knn.predict(X)

        final = (w_dt * dt_price + w_knn * knn_price) / (w_dt + w_knn)
        print(f"    [Hybrid]   Bobot: DT={w_dt:.3f} | KNN={w_knn:.3f}")
        print(f"    [Hybrid]   Rentang: ${final.min():.2f} – ${final.max():.2f}")
        return final
