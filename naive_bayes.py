import numpy as np


class GaussianNaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.parameters = {}

        # Hitung mean dan varians untuk setiap kelas
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.parameters[c] = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0) + 1e-6,  # Tambah epsilon agar tidak dibagi nol
                "prior": X_c.shape[0] / X_train.shape[0],
            }

    def _calculate_likelihood(self, mean, var, x):
        # Rumus Gaussian Probability Density Function (PDF)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.parameters[c]["prior"])

                # Gunakan log-likelihood untuk menghindari underflow pada angka kecil
                likelihood = np.sum(
                    np.log(
                        self._calculate_likelihood(
                            self.parameters[c]["mean"], self.parameters[c]["var"], x
                        )
                    )
                )
                posteriors.append(prior + likelihood)

            # Pilih kelas dengan probabilitas posterior tertinggi
            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)
