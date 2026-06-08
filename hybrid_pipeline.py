import numpy as np


class HybridPipeline:
    def __init__(self, nb_model, dt_model, knn_model, confidence_threshold=0.6):
        self.nb = nb_model
        self.dt = dt_model
        self.knn = knn_model
        self.threshold = confidence_threshold

    def predict(self, X):
        final_predictions = []

        # Stage 1: Pre-filter Naive Bayes
        nb_probs = self.nb.predict_proba(X)

        for i, x in enumerate(X):
            max_prob = np.max(nb_probs[i])
            nb_pred = np.argmax(nb_probs[i])

            # Jika confidence Naive Bayes di bawah threshold, keluarkan sinyal Hold (0)
            if max_prob < self.threshold:
                final_predictions.append(0)
                continue

            # Stage 2: Decision Tree membedah pola yang lolos Stage 1
            x_reshaped = x.reshape(1, -1)
            dt_pred = self.dt.predict(x_reshaped)[0]

            # Stage 3: KNN sebagai Validator
            # Cek ke kemiripan sejarah, apakah DT dan NB terkonfirmasi?
            if dt_pred == nb_pred:
                knn_pred = self.knn.predict(x_reshaped)[0]
                final_predictions.append(knn_pred)
            else:
                # Jika beda, prioritaskan Rules dari Decision Tree (lebih kaku dan definitif)
                final_predictions.append(dt_pred)

        return np.array(final_predictions)
