"""
=================================================================
 Sequential Stock Trading Signal Predictor — Regression Pipeline
─────────────────────────────────────────────────────────────────
 Judul : Implementasi Model Hybrid Bertingkat Ridge Regression,
         Decision Tree, dan K-Nearest Neighbor untuk Prediksi
         Harga Saham pada Indeks S&P 500 Berbasis Indikator Teknikal
─────────────────────────────────────────────────────────────────
 Task   : Regresi — prediksi harga Close hari berikutnya
 Metrik : R², RMSE, MAE, MAPE, Accuracy, Precision, Recall,
          F1-Score, Specificity, AUC-test, AUC-CV (5-fold)
 Library: numpy, pandas — NO scikit-learn
=================================================================
"""

import numpy as np
import os

from preprocessing     import load_and_preprocess
from features          import add_technical_indicators
from linear_regression import RidgeRegression
from decision_tree     import DecisionTree
from knn               import KNearestNeighbors
from hybrid_pipeline   import HybridPipeline
from evaluation        import (evaluate_full, compute_cv_auc,
                                plot_regression_predictions)

np.random.seed(42)
os.makedirs("output", exist_ok=True)

DATA_PATH = "data/SP500_Historical_Data.csv"


def main():

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: LOAD & PREPROCESSING
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 1: Load & Preprocessing Data")
    print("═" * 62)
    df = load_and_preprocess(DATA_PATH, sample_size=100_000)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 2: Feature Engineering — Indikator Teknikal + Lag")
    print("═" * 62)
    X, y, feature_names, current_close = add_technical_indicators(df)
    print(f"  ✓ Shape X={X.shape}, y={y.shape}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: TRAIN / TEST SPLIT + NORMALISASI
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 3: Train/Test Split (80:20) + Normalisasi")
    print("═" * 62)
    split = int(len(X) * 0.8)

    X_train, X_test           = X[:split],             X[split:]
    y_train, y_test           = y[:split],             y[split:]
    cc_train, cc_test         = current_close[:split], current_close[split:]

    # Normalisasi X: fit pada train saja
    X_min   = X_train.min(axis=0)
    X_max   = X_train.max(axis=0)
    denom   = np.where((X_max - X_min) == 0, 1.0, X_max - X_min)
    X_train = (X_train - X_min) / denom
    X_test  = (X_test  - X_min) / denom

    print(f"  ✓ Train: {len(X_train):,} sampel | Test: {len(X_test):,} sampel")
    print(f"  ✓ Normalisasi Min-Max (fit pada train, transform train+test)")
    print(f"  ✓ Rentang y_train: ${y_train.min():.2f} – ${y_train.max():.2f}")
    print(f"  ✓ Rentang y_test : ${y_test.min():.2f} – ${y_test.max():.2f}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: TRAINING SEMUA MODEL
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 4: Training Model")
    print("═" * 62)

    print("\n  [1/3] Ridge Regression (Normal Equation, L2)")
    print("  " + "─" * 50)
    lr_model = RidgeRegression(alpha=1.0)
    lr_model.fit(X_train, y_train)

    print("\n  [2/3] Decision Tree Regressor (CART, Variance Reduction)")
    print("  " + "─" * 50)
    dt_model = DecisionTree(
        task='regressor', max_depth=8, min_samples_split=20,
        min_samples_leaf=10, max_thresholds=30, feature_names=feature_names,
    )
    dt_model.fit(X_train, y_train)
    dt_model.print_tree(max_display_depth=3)

    print("\n  [3/3] K-Nearest Neighbors Regressor (k=7)")
    print("  " + "─" * 50)
    knn_model = KNearestNeighbors(k=7, batch_size=512, task='regressor')
    knn_model.fit(X_train, y_train)

    print("\n  [Hybrid] Menyusun Weighted Average Ensemble...")
    print("  " + "─" * 50)
    hybrid = HybridPipeline(lr_model, dt_model, knn_model)
    hybrid.fit_weights(X_train, y_train)
    print("  ✓ Hybrid Pipeline siap")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: CROSS-VALIDATION AUC (5-fold time-series, pada X_train)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 5: Cross-Validation AUC (5-fold time-series)")
    print("═" * 62)

    print("\n  CV AUC — Ridge Regression...")
    # Buat salinan agar CV tidak merusak model yang sudah di-fit
    lr_cv  = RidgeRegression(alpha=1.0)
    auc_cv_lr  = compute_cv_auc(lr_cv,  X_train, y_train, cc_train, n_folds=5, verbose=True)
    print(f"  Ridge LR  AUC-CV: {auc_cv_lr[0]:.4f} ± {auc_cv_lr[1]:.4f}")

    print("\n  CV AUC — Decision Tree...")
    dt_cv  = DecisionTree(task='regressor', max_depth=8, min_samples_split=20,
                          min_samples_leaf=10, max_thresholds=30)
    auc_cv_dt  = compute_cv_auc(dt_cv,  X_train, y_train, cc_train, n_folds=5, verbose=True)
    print(f"  DT        AUC-CV: {auc_cv_dt[0]:.4f} ± {auc_cv_dt[1]:.4f}")

    print("\n  CV AUC — KNN (harap tunggu)...")
    knn_cv = KNearestNeighbors(k=7, batch_size=512, task='regressor')
    auc_cv_knn = compute_cv_auc(knn_cv, X_train, y_train, cc_train, n_folds=5, verbose=True)
    print(f"  KNN       AUC-CV: {auc_cv_knn[0]:.4f} ± {auc_cv_knn[1]:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: PREDIKSI & EVALUASI LENGKAP PADA TEST SET
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 6: Prediksi & Evaluasi Lengkap pada Test Set")
    print("═" * 62)

    print("\n  [1/4] Ridge Regression")
    y_pred_lr  = lr_model.predict(X_test)
    m_lr  = evaluate_full(y_test, y_pred_lr,  cc_test, "Ridge Regression", auc_cv=auc_cv_lr)

    print("\n  [2/4] Decision Tree Regressor")
    y_pred_dt  = dt_model.predict(X_test)
    m_dt  = evaluate_full(y_test, y_pred_dt,  cc_test, "Decision Tree",    auc_cv=auc_cv_dt)

    print("\n  [3/4] KNN Regressor (harap tunggu...)")
    y_pred_knn = knn_model.predict(X_test)
    m_knn = evaluate_full(y_test, y_pred_knn, cc_test, "KNN Regressor",    auc_cv=auc_cv_knn)

    print("\n  [4/4] Hybrid Ensemble")
    y_pred_hy  = hybrid.predict(X_test)
    # Hybrid CV AUC = rata-rata berbobot dari ketiga model
    w      = hybrid.weights
    cv_hy_mean = w[0]*auc_cv_lr[0] + w[1]*auc_cv_dt[0] + w[2]*auc_cv_knn[0]
    cv_hy_std  = w[0]*auc_cv_lr[1] + w[1]*auc_cv_dt[1] + w[2]*auc_cv_knn[1]
    m_hy  = evaluate_full(y_test, y_pred_hy,  cc_test, "Hybrid Ensemble",
                          auc_cv=(cv_hy_mean, cv_hy_std))

    # ═══════════════════════════════════════════════════════════════
    # STEP 7: PLOT PREDIKSI
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("  STEP 7: Plot Actual vs Predicted")
    print("═" * 62)
    plot_regression_predictions(
        {
            "Ridge Regression": (y_test, y_pred_lr),
            "Decision Tree"   : (y_test, y_pred_dt),
            "KNN Regressor"   : (y_test, y_pred_knn),
            "Hybrid Ensemble" : (y_test, y_pred_hy),
        },
        save_path="output/regression_predictions.png",
    )

    # ═══════════════════════════════════════════════════════════════
    # STEP 8: TABEL KOMPARASI AKHIR
    # ═══════════════════════════════════════════════════════════════
    models = [
        ("Ridge Regression", m_lr),
        ("Decision Tree",    m_dt),
        ("KNN Regressor",    m_knn),
        ("Hybrid Ensemble",  m_hy),
    ]

    print("\n" + "═" * 90)
    print(f"{'TABEL KOMPARASI EVALUASI LENGKAP':^90}")
    print("═" * 90)
    hdr = (f"  {'METODE':<20} | {'R²':>6} | {'RMSE':>7} | {'MAE':>7} | "
           f"{'MAPE':>6} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | "
           f"{'F1':>6} | {'AUC-t':>6} | {'AUC-CV':>9}")
    print(hdr)
    print("  " + "─" * 86)
    for name, m in models:
        cv_str = f"{m['auc_cv_mean']:.4f}" if m['auc_cv_mean'] is not None else "  —  "
        print(
            f"  {name:<20} | {m['r2']:>6.4f} | {m['rmse']:>7.2f} | {m['mae']:>7.2f} | "
            f"{m['mape']:>5.1f}% | {m['accuracy']:>6.4f} | {m['precision']:>6.4f} | "
            f"{m['recall']:>6.4f} | {m['f1']:>6.4f} | {m['auc_test']:>6.4f} | {cv_str:>9}"
        )
    print("═" * 90)

    # Simpan CSV
    header = "Model,R2,RMSE,MAE,MAPE,Accuracy,Precision,Recall,F1,AUC_test,AUC_CV_mean,AUC_CV_std\n"
    lines  = [header]
    for name, m in models:
        cv_m = f"{m['auc_cv_mean']:.4f}" if m['auc_cv_mean'] is not None else ""
        cv_s = f"{m['auc_cv_std']:.4f}"  if m['auc_cv_std']  is not None else ""
        lines.append(
            f"{name},{m['r2']:.4f},{m['rmse']:.4f},{m['mae']:.4f},{m['mape']:.2f},"
            f"{m['accuracy']:.4f},{m['precision']:.4f},{m['recall']:.4f},{m['f1']:.4f},"
            f"{m['auc_test']:.4f},{cv_m},{cv_s}\n"
        )
    with open("output/evaluation_summary.csv", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("  ✓ Summary disimpan → output/evaluation_summary.csv")

    print("\n" + "═" * 62)
    print("  SELESAI — Hybrid Regression Pipeline")
    print("  Output tersimpan di folder: output/")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
