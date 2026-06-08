"""
decision_tree.py
────────────────
CART Decision Tree Classifier — from scratch (numpy only).

PROSES DETAIL TIAP TAHAP:
  1. fit()           : Bangun pohon secara rekursif dari root
  2. _build_tree()   : Rekursi — cek stopping, cari best split, buat node
  3. _best_criteria() : Cari fitur + threshold terbaik (Gini Impurity minimum)
  4. _weighted_gini() : Hitung Gini Impurity terbobot untuk satu split
  5. predict()        : Traversal pohon untuk setiap sampel
  6. print_tree()     : Tampilkan struktur pohon ke terminal (ASCII)

Formula Gini Impurity:
  Gini(S) = 1 - Σ pᵢ²
  Weighted Gini(split) = (|left|/|S|)×Gini(left) + (|right|/|S|)×Gini(right)
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════
#  NODE
# ══════════════════════════════════════════════════════════════════


class Node:
    """
    Satu node dalam Decision Tree.

    Atribut:
      feature   : indeks fitur yang digunakan untuk split (None = leaf)
      threshold : nilai threshold split (None = leaf)
      left      : subtree kiri  (X <= threshold)
      right     : subtree kanan (X >  threshold)
      value     : prediksi kelas (hanya pada leaf node)
      gini      : nilai Gini Impurity node ini
      n_samples : jumlah sampel di node ini
      depth     : kedalaman node
    """

    __slots__ = (
        "feature",
        "threshold",
        "left",
        "right",
        "value",
        "gini",
        "n_samples",
        "depth",
    )

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        gini=0.0,
        n_samples=0,
        depth=0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gini = gini
        self.n_samples = n_samples
        self.depth = depth


# ══════════════════════════════════════════════════════════════════
#  DECISION TREE
# ══════════════════════════════════════════════════════════════════


class DecisionTree:
    """
    CART Decision Tree untuk klasifikasi binary (Buy=1 / Sell=0).

    Parameter:
      max_depth         : kedalaman maksimum pohon
      min_samples_split : minimal sampel agar node di-split
      min_samples_leaf  : minimal sampel di setiap leaf
      max_thresholds    : maks kandidat threshold per fitur (efisiensi)
      feature_names     : nama fitur untuk print_tree (opsional)
    """

    def __init__(
        self,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_thresholds=30,
        feature_names=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.feature_names = feature_names
        self.root = None
        self._n_nodes = 0
        self._n_leaves = 0

    # ── TRAINING ──────────────────────────────────────────────────

    def fit(self, X, y):
        """
        TAHAP TRAINING — Bangun decision tree secara rekursif.

        Langkah:
          1. Mulai dari root node dengan seluruh data X, y
          2. Di setiap node, cari split terbaik (fitur + threshold)
             yang meminimalkan Gini Impurity terbobot
          3. Bagi data menjadi subset kiri dan kanan
          4. Rekursi ke kiri dan kanan hingga stopping criteria terpenuhi
          5. Stopping criteria:
             a. Kedalaman sudah mencapai max_depth
             b. Semua sampel satu kelas (pure node, Gini = 0)
             c. Jumlah sampel < min_samples_split
             d. Tidak ada split yang valid
        """
        self._n_nodes = 0
        self._n_leaves = 0
        n_samples, n_features = X.shape
        print(
            f"    [DT] Membangun pohon: {n_samples:,} sampel, "
            f"{n_features} fitur, max_depth={self.max_depth}"
        )
        self.root = self._build_tree(X, y, depth=0)
        print(
            f"    [DT] Pohon selesai: {self._n_nodes} node total, "
            f"{self._n_leaves} leaf node"
        )

    def _build_tree(self, X, y, depth):
        """
        Rekursi — bangun satu node dan dua subtreenya.

        Proses per node:
          1. Hitung Gini Impurity node saat ini
          2. Cek stopping criteria → buat leaf jika terpenuhi
          3. Cari split terbaik dengan _best_criteria()
          4. Bagi data dengan np.where
          5. Rekursi ke anak kiri dan kanan
        """
        self._n_nodes += 1
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))
        node_gini = self._gini(y)

        # ── Stopping criteria ────────────────────────────────────
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            self._n_leaves += 1
            return Node(
                value=self._majority(y),
                gini=node_gini,
                n_samples=n_samples,
                depth=depth,
            )

        # ── Cari split terbaik ────────────────────────────────────
        best_feat, best_thresh = self._best_criteria(X, y)
        if best_feat is None:
            self._n_leaves += 1
            return Node(
                value=self._majority(y),
                gini=node_gini,
                n_samples=n_samples,
                depth=depth,
            )

        # ── Split data ────────────────────────────────────────────
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        n_left, n_right = left_mask.sum(), right_mask.sum()

        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            self._n_leaves += 1
            return Node(
                value=self._majority(y),
                gini=node_gini,
                n_samples=n_samples,
                depth=depth,
            )

        # ── Rekursi ───────────────────────────────────────────────
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_feat,
            threshold=best_thresh,
            left=left,
            right=right,
            gini=node_gini,
            n_samples=n_samples,
            depth=depth,
        )

    # ── MENCARI SPLIT TERBAIK ─────────────────────────────────────

    def _best_criteria(self, X, y):
        """
        Iterasi semua fitur dan kandidat threshold untuk menemukan
        split dengan Gini Impurity terbobot paling kecil.

        Efisiensi: threshold dibatasi maks max_thresholds per fitur
        menggunakan persentil merata (bukan semua nilai unik).

        Return: (best_feature_idx, best_threshold)
        """
        best_gini = 1.0
        best_feat = None
        best_thresh = None

        for feat_idx in range(X.shape[1]):
            col = X[:, feat_idx]
            unique_vals = np.unique(col)

            # Batasi kandidat threshold untuk efisiensi komputasi
            if len(unique_vals) > self.max_thresholds:
                percs = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
                thresholds = np.percentile(col, percs)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                gini = self._weighted_gini(y[left_mask], y[right_mask])

                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    # ── GINI IMPURITY ─────────────────────────────────────────────

    def _gini(self, y):
        """
        Gini Impurity untuk satu node.
        Gini(S) = 1 - Σ pᵢ²
        Nilai 0 = perfectly pure, 0.5 = perfectly impure (binary).
        """
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs**2))

    def _weighted_gini(self, left_y, right_y):
        """
        Weighted Gini untuk satu split.
        Gini_w = (n_L/n)×Gini(L) + (n_R/n)×Gini(R)
        Makin kecil nilainya, makin baik splitnya.
        """
        n = len(left_y) + len(right_y)
        w_l = len(left_y) / n
        w_r = len(right_y) / n
        return w_l * self._gini(left_y) + w_r * self._gini(right_y)

    def _majority(self, y):
        """Kelas mayoritas — digunakan sebagai prediksi leaf."""
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    # ── PREDIKSI ──────────────────────────────────────────────────

    def predict(self, X):
        """
        Traversal pohon dari root ke leaf untuk setiap sampel.
        Di setiap node: jika x[feature] <= threshold → kiri, else → kanan.
        Sampai leaf → kembalikan node.value.
        """
        return np.array([self._traverse(x, self.root) for x in X], dtype=np.int32)

    def _traverse(self, x, node):
        """Traversal rekursif satu sampel."""
        if node.value is not None:  # leaf
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ── VISUALISASI POHON (ASCII, tanpa library) ──────────────────

    def print_tree(self, max_display_depth=4):
        """
        Tampilkan struktur Decision Tree ke terminal dalam bentuk ASCII.

        Contoh output:
          [Node | feat=RSI ≤ 0.4500 | gini=0.4990 | n=69986]
          ├── L: [Node | feat=MACD ≤ 0.3210 | gini=0.4871 | n=35123]
          │   ├── L: [Leaf | pred=Sell(0) | gini=0.4502 | n=17800]
          │   └── R: [Leaf | pred=Buy(1)  | gini=0.4710 | n=17323]
          └── R: [Leaf | pred=Buy(1) | gini=0.4812 | n=34863]
        """
        if self.root is None:
            print("    [DT] Pohon belum dilatih.")
            return

        feat_names = self.feature_names or [f"feat_{i}" for i in range(100)]
        print(f"\n{'─'*60}")
        print(f"  STRUKTUR DECISION TREE (max tampil depth={max_display_depth})")
        print(f"  Total node: {self._n_nodes} | Leaf: {self._n_leaves}")
        print(f"{'─'*60}")
        self._print_node(
            self.root,
            prefix="",
            is_left=None,
            feat_names=feat_names,
            max_depth=max_display_depth,
        )
        print(f"{'─'*60}\n")

    def _print_node(self, node, prefix, is_left, feat_names, max_depth):
        """Rekursi cetak satu node."""
        if node is None or node.depth > max_depth:
            return

        # Label posisi
        if is_left is None:
            connector = ""
            new_prefix = ""
        elif is_left:
            connector = "├── L: "
            new_prefix = prefix + "│   "
        else:
            connector = "└── R: "
            new_prefix = prefix + "    "

        if node.value is not None:
            label = "Buy(1)" if node.value == 1 else "Sell(0)"
            print(
                f"{prefix}{connector}"
                f"[LEAF | pred={label} | "
                f"gini={node.gini:.4f} | n={node.n_samples:,}]"
            )
        else:
            fname = (
                feat_names[node.feature]
                if node.feature < len(feat_names)
                else f"feat_{node.feature}"
            )
            print(
                f"{prefix}{connector}"
                f"[NODE | {fname} ≤ {node.threshold:.4f} | "
                f"gini={node.gini:.4f} | n={node.n_samples:,}]"
            )
            if node.depth < max_depth:
                self._print_node(node.left, new_prefix, True, feat_names, max_depth)
                self._print_node(node.right, new_prefix, False, feat_names, max_depth)
            else:
                print(f"{new_prefix}├── ... (depth limit)")
                print(f"{new_prefix}└── ... (depth limit)")
