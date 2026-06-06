import numpy as np


class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        total_data = len(X_test)  # Hitung total data

        for i, x in enumerate(X_test):
            # Hitung Jarak Euclidean
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.y_train[j] for j in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)

            # --- INDIKATOR PROGRESS ---
            # Update setiap 1% atau jika data sudah di akhir agar terminal tidak nge-lag karena keseringan print
            if (i + 1) % max(1, total_data // 100) == 0 or i == total_data - 1:
                persentase = ((i + 1) / total_data) * 100
                print(
                    f"\r[KNN Predict] Progress: {persentase:.1f}% ({i+1}/{total_data} data)",
                    end="",
                    flush=True,
                )

        print()  # Beri enter baru setelah selesai 100%
        return np.array(predictions)
