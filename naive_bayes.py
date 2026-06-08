import numpy as np


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]), dtype=np.float64)
        self.var = np.zeros((len(self.classes), X.shape[1]), dtype=np.float64)
        self.priors = np.zeros(len(self.classes), dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-6  # Menghindari error pembagian nol
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return (numerator / denominator) + 1e-9

    def predict_proba(self, X):
        probas = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                # Jumlahkan log probabilitas untuk menghindari underflow
                conditional = np.sum(np.log(self._pdf(idx, x)))
                posterior = prior + conditional
                posteriors.append(posterior)
            # Konversi balik jadi probabilitas (Softmax sederhana)
            posteriors = np.exp(posteriors - np.max(posteriors))
            probas.append(posteriors / posteriors.sum())
        return np.array(probas)
