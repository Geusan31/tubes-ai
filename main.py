"""
=================================================================
 Sequential Stock Trading Signal Predictor
─────────────────────────────────────────────────────────────────
 Judul : Implementasi Model Hybrid Bertingkat Naive Bayes,
         Decision Tree, dan K-Nearest Neighbor untuk Prediksi
         Arah Pergerakan Harga Saham pada Indeks S&P 500
         Berbasis Indikator Teknikal
─────────────────────────────────────────────────────────────────
 Library: numpy, pandas (preprocessing) — NO scikit-learn
 Visualisasi: ASCII art (tanpa matplotlib)
=================================================================
"""

import numpy as np
import os

from preprocessing import load_and_preprocess
from features import add_technical_indicators
from naive_bayes import GaussianNaiveBayes
from decision_tree import DecisionTree
from knn import KNearestNeighbors
from hybrid_pipeline import HybridPipeline
from evaluation import evaluate_model, plot_all_confusion_matrices, save_summary_csv

np.random.seed(42)
os.makedirs("output", exist_ok=True)

DATA_PATH = "data/SP500_Historical_Data.csv"
FEATURE_NAMES = [
    "RSI",
    "MACD",
    "BB_Upper",
    "BB_Lower",
    "SMA_20",
    "EMA_12",
    "Close",
    "Volume_Change",
]


def main():

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: LOAD & PREPROCESSING
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 1: Load & Preprocessing Data")
    print("═" * 62)
    df = load_and_preprocess(DATA_PATH, sample_size=100_000)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: FEATURE ENGINEERING (Indikator Teknikal)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 2: Feature Engineering — Indikator Teknikal")
    print("═" * 62)
    X, y = add_technical_indicators(df)
    print(f"  ✓ Fitur: {FEATURE_NAMES}")
    print(f"  ✓ Shape X={X.shape}, y={y.shape}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: TRAIN / TEST SPLIT
    # Time-series split: urutan waktu dijaga, tidak di-shuffle
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 3: Train/Test Split (70:30 — time-series order)")
    print("═" * 62)
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  ✓ Train: {len(X_train):,} sampel")
    print(f"  ✓ Test : {len(X_test):,} sampel")
    buy_tr = int((y_train == 1).sum())
    sell_tr = int((y_train == 0).sum())
    buy_te = int((y_test == 1).sum())
    sell_te = int((y_test == 0).sum())
    print(f"  ✓ Train label — Buy: {buy_tr:,} | Sell: {sell_tr:,}")
    print(f"  ✓ Test  label — Buy: {buy_te:,} | Sell: {sell_te:,}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: TRAINING SEMUA MODEL
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 4: Training Model")
    print("═" * 62)

    # ── Naive Bayes ────────────────────────────────────────────
    print("\n  [1/3] Gaussian Naive Bayes")
    print("  " + "─" * 50)
    nb_model = GaussianNaiveBayes()
    nb_model.fit(X_train, y_train)

    # ── Decision Tree ──────────────────────────────────────────
    print("\n  [2/3] Decision Tree (CART, Gini Impurity)")
    print("  " + "─" * 50)
    dt_model = DecisionTree(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_thresholds=30,
        feature_names=FEATURE_NAMES,
    )
    dt_model.fit(X_train, y_train)

    # Tampilkan struktur pohon (ASCII, tanpa library)
    dt_model.print_tree(max_display_depth=3)

    # ── KNN ────────────────────────────────────────────────────
    print("\n  [3/3] K-Nearest Neighbors (k=7, Euclidean)")
    print("  " + "─" * 50)
    knn_model = KNearestNeighbors(k=7, batch_size=512)
    knn_model.fit(X_train, y_train)

    # ── Hybrid Pipeline ────────────────────────────────────────
    print("\n  [Hybrid] Menyusun Pipeline NB → DT → KNN")
    print("  " + "─" * 50)
    hybrid = HybridPipeline(nb_model, dt_model, knn_model, confidence_threshold=0.60)
    print("  ✓ Hybrid Pipeline siap")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: PREDIKSI & EVALUASI
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 5: Prediksi & Evaluasi Model")
    print("═" * 62)

    # 1. Naive Bayes
    print("\n  [1/4] Prediksi — Naive Bayes")
    y_pred_nb = nb_model.predict(X_test)
    acc_nb, prec_nb, rec_nb, f1_nb = evaluate_model(y_test, y_pred_nb, "Naive Bayes")

    # 2. Decision Tree
    print("\n  [2/4] Prediksi — Decision Tree")
    y_pred_dt = dt_model.predict(X_test)
    acc_dt, prec_dt, rec_dt, f1_dt = evaluate_model(y_test, y_pred_dt, "Decision Tree")

    # 3. KNN
    print("\n  [3/4] Prediksi — K-Nearest Neighbors")
    print("  (harap tunggu, komputasi batch vectorized...)")
    y_pred_knn = knn_model.predict(X_test)
    acc_knn, prec_knn, rec_knn, f1_knn = evaluate_model(
        y_test, y_pred_knn, "K-Nearest Neighbors"
    )

    # 4. Hybrid Pipeline
    print("\n  [4/4] Prediksi — Hybrid Pipeline (NB → DT → KNN)")
    y_pred_hy = hybrid.predict(X_test)
    acc_hy, prec_hy, rec_hy, f1_hy = evaluate_model(
        y_test, y_pred_hy, "Hybrid (NB→DT→KNN)"
    )

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: CONFUSION MATRIX (ASCII, tanpa matplotlib)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 6: Confusion Matrix (ASCII)")
    print("═" * 62)
    plot_all_confusion_matrices(
        {
            "Naive Bayes": (y_test, y_pred_nb),
            "Decision Tree": (y_test, y_pred_dt),
            "K-Nearest Neighbors": (y_test, y_pred_knn),
            "Hybrid (NB→DT→KNN)": (y_test, y_pred_hy),
        },
        save_path="output/confusion_matrices.png",  # <--- UBAH JADI .png
    )
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 7: TABEL KOMPARASI AKHIR
    # ═══════════════════════════════════════════════════════════════
    rows = [
        ("Naive Bayes", acc_nb, prec_nb, rec_nb, f1_nb),
        ("Decision Tree", acc_dt, prec_dt, rec_dt, f1_dt),
        ("K-Nearest Neighbors", acc_knn, prec_knn, rec_knn, f1_knn),
        ("Hybrid (NB→DT→KNN)", acc_hy, prec_hy, rec_hy, f1_hy),
    ]

    print("\n" + "═" * 74)
    print(f"{'TABEL KOMPARASI EVALUASI':^74}")
    print("═" * 74)
    print(
        f"  {'METODE':<24} | {'ACCURACY':>9} | {'PRECISION':>9} | "
        f"{'RECALL':>9} | {'F1-SCORE':>9}"
    )
    print("  " + "─" * 70)
    for name, acc, prec, rec, f1 in rows:
        print(
            f"  {name:<24} | {acc:>9.4f} | {prec:>9.4f} | " f"{rec:>9.4f} | {f1:>9.4f}"
        )
    print("═" * 74)

    # Simpan ke CSV
    save_summary_csv(rows, save_path="output/evaluation_summary.csv")

    print("\n" + "═" * 62)
    print("  SELESAI — Hybrid Pipeline")
    print("  Output tersimpan di folder: output/")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
