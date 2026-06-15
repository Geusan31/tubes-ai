"""
decision_tree.py
────────────────
CART Decision Tree — dari scratch (numpy only).
Mendukung dua mode:
  task='regressor'   : minimasi MSE/variance → prediksi mean (float)
  task='classifier'  : minimasi Gini Impurity → prediksi kelas mayoritas (int)

Formula Regresi (MSE Reduction):
  Variance(S)   = (1/n) Σ (yᵢ - ȳ)²
  Weighted_Var  = (n_L/n)×Var(L) + (n_R/n)×Var(R)
  Split terbaik = yang meminimalkan Weighted_Var

Formula Klasifikasi (Gini):
  Gini(S) = 1 - Σ pᵢ²
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════
#  NODE
# ══════════════════════════════════════════════════════════════════

class Node:
    __slots__ = ("feature", "threshold", "left", "right",
                 "value", "gini", "n_samples", "depth")

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, gini=0.0, n_samples=0, depth=0):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value
        self.gini      = gini
        self.n_samples = n_samples
        self.depth     = depth


# ══════════════════════════════════════════════════════════════════
#  DECISION TREE
# ══════════════════════════════════════════════════════════════════

class DecisionTree:
    """
    CART Decision Tree: classifier atau regressor.

    Parameter:
      task              : 'regressor' atau 'classifier'
      max_depth         : kedalaman maksimum pohon
      min_samples_split : minimal sampel agar node di-split
      min_samples_leaf  : minimal sampel di setiap leaf
      max_thresholds    : maks kandidat threshold per fitur
      feature_names     : nama fitur untuk print_tree (opsional)
    """

    def __init__(self, task='regressor', max_depth=8, min_samples_split=20,
                 min_samples_leaf=10, max_thresholds=30, feature_names=None):
        self.task              = task
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_thresholds    = max_thresholds
        self.feature_names     = feature_names
        self.root              = None
        self._n_nodes          = 0
        self._n_leaves         = 0

    # ── TRAINING ──────────────────────────────────────────────────

    def fit(self, X, y):
        self._n_nodes = 0
        self._n_leaves = 0
        n_samples, n_features = X.shape
        mode = "regresi (MSE)" if self.task == 'regressor' else "klasifikasi (Gini)"
        print(f"    [DT] Membangun pohon {mode}: {n_samples:,} sampel, "
              f"{n_features} fitur, max_depth={self.max_depth}")
        self.root = self._build_tree(X, y, depth=0)
        print(f"    [DT] Pohon selesai: {self._n_nodes} node total, "
              f"{self._n_leaves} leaf node")

    def _build_tree(self, X, y, depth):
        self._n_nodes += 1
        n_samples  = X.shape[0]
        node_score = self._node_score(y)

        # Stopping criteria
        stop = (depth >= self.max_depth or n_samples < self.min_samples_split)
        if self.task == 'classifier':
            stop = stop or (len(np.unique(y)) == 1)

        if stop:
            self._n_leaves += 1
            return Node(value=self._leaf_value(y), gini=node_score,
                        n_samples=n_samples, depth=depth)

        best_feat, best_thresh = self._best_criteria(X, y)
        if best_feat is None:
            self._n_leaves += 1
            return Node(value=self._leaf_value(y), gini=node_score,
                        n_samples=n_samples, depth=depth)

        left_mask  = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            self._n_leaves += 1
            return Node(value=self._leaf_value(y), gini=node_score,
                        n_samples=n_samples, depth=depth)

        left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh,
                    left=left, right=right, gini=node_score,
                    n_samples=n_samples, depth=depth)

    # ── SPLIT TERBAIK ─────────────────────────────────────────────

    def _best_criteria(self, X, y):
        best_score = float('inf')
        best_feat  = None
        best_thresh = None

        for feat_idx in range(X.shape[1]):
            col         = X[:, feat_idx]
            unique_vals = np.unique(col)

            if len(unique_vals) > self.max_thresholds:
                percs      = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
                thresholds = np.percentile(col, percs)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask  = col <= thresh
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                score = self._weighted_score(y[left_mask], y[right_mask])
                if score < best_score:
                    best_score  = score
                    best_feat   = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    # ── SKOR NODE ─────────────────────────────────────────────────

    def _node_score(self, y):
        """MSE (variance) untuk regresi, Gini untuk klasifikasi."""
        if len(y) == 0:
            return 0.0
        if self.task == 'regressor':
            return float(np.var(y))
        else:
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return float(1.0 - np.sum(probs ** 2))

    def _weighted_score(self, left_y, right_y):
        n   = len(left_y) + len(right_y)
        w_l = len(left_y)  / n
        w_r = len(right_y) / n
        return w_l * self._node_score(left_y) + w_r * self._node_score(right_y)

    def _leaf_value(self, y):
        """Mean untuk regresi, kelas mayoritas untuk klasifikasi."""
        if self.task == 'regressor':
            return float(np.mean(y))
        else:
            vals, counts = np.unique(y, return_counts=True)
            return int(vals[np.argmax(counts)])

    # ── PREDIKSI ──────────────────────────────────────────────────

    def predict(self, X):
        if self.task == 'regressor':
            return np.array([self._traverse(x, self.root) for x in X], dtype=np.float64)
        else:
            return np.array([self._traverse(x, self.root) for x in X], dtype=np.int32)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ── VISUALISASI POHON ─────────────────────────────────────────

    def print_tree(self, max_display_depth=4):
        if self.root is None:
            print("    [DT] Pohon belum dilatih.")
            return
        feat_names = self.feature_names or [f"feat_{i}" for i in range(100)]
        print(f"\n{'─'*60}")
        print(f"  STRUKTUR DECISION TREE (max tampil depth={max_display_depth})")
        print(f"  Mode: {self.task} | Total node: {self._n_nodes} | Leaf: {self._n_leaves}")
        print(f"{'─'*60}")
        self._print_node(self.root, "", None, feat_names, max_display_depth)
        print(f"{'─'*60}\n")

    def _print_node(self, node, prefix, is_left, feat_names, max_depth):
        if node is None or node.depth > max_depth:
            return
        if is_left is None:
            connector, new_prefix = "", ""
        elif is_left:
            connector, new_prefix = "├── L: ", prefix + "│   "
        else:
            connector, new_prefix = "└── R: ", prefix + "    "

        score_label = "var" if self.task == 'regressor' else "gini"

        if node.value is not None:
            val_str = f"{node.value:.2f}" if self.task == 'regressor' else \
                      ("Buy(1)" if node.value == 1 else "Sell(0)")
            print(f"{prefix}{connector}[LEAF | pred={val_str} | "
                  f"{score_label}={node.gini:.4f} | n={node.n_samples:,}]")
        else:
            fname = (feat_names[node.feature]
                     if node.feature < len(feat_names) else f"feat_{node.feature}")
            print(f"{prefix}{connector}[NODE | {fname} ≤ {node.threshold:.4f} | "
                  f"{score_label}={node.gini:.4f} | n={node.n_samples:,}]")
            if node.depth < max_depth:
                self._print_node(node.left,  new_prefix, True,  feat_names, max_depth)
                self._print_node(node.right, new_prefix, False, feat_names, max_depth)
            else:
                print(f"{new_prefix}├── ... (depth limit)")
                print(f"{new_prefix}└── ... (depth limit)")
