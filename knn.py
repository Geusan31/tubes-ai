import numpy as np


class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Hitung jarak Euclidean
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Ambil k-index terdekat
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Voting terbanyak (Majority vote)
        return np.bincount(k_nearest_labels).argmax()
