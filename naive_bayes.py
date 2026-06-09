"""
naive_bayes.py
──────────────
Gaussian Naive Bayes Classifier — from scratch (numpy only).

PROSES DETAIL TIAP TAHAP:
  1. fit()         : Estimasi parameter distribusi Gaussian per kelas
                     (prior P(y), mean μ, variance σ²)
  2. predict_proba(): Hitung posterior P(y|x) dengan Bayes' Theorem
                      menggunakan log-likelihood untuk stabilitas numerik
  3. predict()     : Ambil kelas dengan posterior tertinggi (argmax)

Formula:
  P(y | x₁..xₙ) ∝ P(y) × Π P(xᵢ | y)
  P(xᵢ | y)      = Gaussian(xᵢ ; μᵢᵧ, σ²ᵢᵧ)
  Gaussian(x;μ,σ²) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
"""

import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier from scratch.

    Asumsi: setiap fitur berdistribusi normal (Gaussian) secara
    independen kondisional terhadap kelasnya.

    Parameter:
      var_smoothing      : nilai minimum variansi (hindari div-by-zero)
      use_uniform_prior  : jika True, gunakan P(c)=1/K untuk semua kelas
                           (mencegah bias akibat sedikit ketidakseimbangan label)
    """

    def __init__(self, var_smoothing=1e-9, use_uniform_prior=True):
        self.var_smoothing     = var_smoothing
        self.use_uniform_prior = use_uniform_prior

    def fit(self, X, y):
        """
        TAHAP TRAINING:
        Estimasi parameter distribusi Gaussian untuk setiap kelas.

        Langkah:
          1. Identifikasi semua kelas unik (0=Sell, 1=Buy)
          2. Untuk setiap kelas c:
             a. Pisahkan sampel X_c = X[y == c]
             b. Hitung prior: uniform (0.5,0.5) atau empiris
             c. Hitung mean μ per fitur
             d. Hitung variance σ² per fitur (+ smoothing)
        """
        self.classes_ = np.unique(y)
        n_classes      = len(self.classes_)
        n_samples, n_features = X.shape

        self.mean_   = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var_    = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        print(f"    [NB] Training pada {n_samples:,} sampel, {n_features} fitur")
        print(f"    [NB] Prior: {'uniform (1/K)' if self.use_uniform_prior else 'empiris'}")
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            if self.use_uniform_prior:
                self.priors_[idx] = 1.0 / n_classes
            else:
                self.priors_[idx] = X_c.shape[0] / n_samples
            self.mean_[idx]   = X_c.mean(axis=0)
            self.var_[idx]    = X_c.var(axis=0) + self.var_smoothing
            label = "Buy (1)" if c == 1 else "Sell (0)"
            print(f"    [NB]   Kelas {label}: {X_c.shape[0]:,} sampel | "
                  f"prior={self.priors_[idx]:.4f}")

        print(f"    [NB] Parameter tersimpan: mean shape={self.mean_.shape}, "
              f"var shape={self.var_.shape}")

    def _gaussian_log_pdf(self, class_idx, X):
        """
        Hitung log P(X | class) menggunakan Gaussian PDF.

        Formula log PDF:
          log P(x|μ,σ²) = -0.5 × [log(2πσ²) + (x-μ)²/σ²]

        Menggunakan log agar tidak terjadi numerical underflow
        saat mengalikan banyak probabilitas kecil.

        X shape: (n_samples, n_features)
        Return : (n_samples,) — log-likelihood per sampel
        """
        mean = self.mean_[class_idx]   # shape: (n_features,)
        var  = self.var_[class_idx]    # shape: (n_features,)

        # Vectorized: sum atas semua fitur untuk tiap sampel
        log_pdf = -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var,
            axis=1
        )
        return log_pdf   # shape: (n_samples,)

    def predict_proba(self, X):
        """
        TAHAP INFERENSI — Hitung probabilitas posterior setiap kelas.

        Langkah:
          1. Untuk setiap kelas c:
             a. log P(c)       = log prior
             b. log P(X|c)     = Σ log Gaussian(xᵢ; μᵢc, σ²ᵢc)
             c. log posterior  = log P(c) + log P(X|c)
          2. Gabungkan log-posterior semua kelas → matrix (n, K)
          3. Log-sum-exp trick: konversi ke probabilitas yang stabil

        Return: ndarray shape (n_samples, n_classes)
        """
        X = np.array(X, dtype=np.float64)
        log_posteriors = []

        for idx in range(len(self.classes_)):
            log_prior    = np.log(self.priors_[idx])             # scalar
            log_likeli   = self._gaussian_log_pdf(idx, X)        # (n_samples,)
            log_posteriors.append(log_prior + log_likeli)

        # (n_classes, n_samples) → (n_samples, n_classes)
        log_post = np.array(log_posteriors).T

        # Log-sum-exp trick untuk stabilitas numerik:
        # softmax(z) = exp(z - max(z)) / Σ exp(z - max(z))
        max_log   = log_post.max(axis=1, keepdims=True)
        exp_val   = np.exp(log_post - max_log)
        probs     = exp_val / exp_val.sum(axis=1, keepdims=True)

        return probs   # shape: (n_samples, n_classes)

    def predict(self, X):
        """
        Prediksi kelas dengan posterior probability tertinggi.
        Return: ndarray shape (n_samples,) berisi 0 atau 1
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
