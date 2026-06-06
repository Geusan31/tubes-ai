import numpy as np


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        p1 = len(y[y == 1]) / m
        p0 = 1 - p1
        return 1 - (p0**2 + p1**2)

    def _best_split(self, X, y):
        m, n = X.shape
        best_gini = float("inf")
        best_idx, best_thr = None, None

        for feature_idx in range(n):
            thresholds = np.unique(X[:, feature_idx])
            for thr in thresholds:
                left_mask = X[:, feature_idx] <= thr
                right_mask = X[:, feature_idx] > thr

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])

                # Weighted Gini Impurity
                n_left, n_right = sum(left_mask), sum(right_mask)
                gini = (n_left / m) * gini_left + (n_right / m) * gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_idx = feature_idx
                    best_thr = thr

        return best_idx, best_thr

    def _build_tree(self, X, y, depth):
        print(
            f"\r[Decision Tree Fit] Memproses kedalaman pohon (depth): {depth}...",
            end="",
            flush=True,
        )
        # Base cases: node murni atau mencapai max depth
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = (
            np.argmax(num_samples_per_class) if len(num_samples_per_class) > 0 else 0
        )

        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < 2:
            return Node(value=predicted_class)

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return Node(value=predicted_class)

        left_mask = X[:, feat_idx] <= threshold
        right_mask = X[:, feat_idx] > threshold

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feat_idx, threshold, left_child, right_child)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
