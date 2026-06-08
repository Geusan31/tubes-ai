import numpy as np
from preprocessing import load_and_preprocess
from features import add_technical_indicators
from naive_bayes import GaussianNaiveBayes
from decision_tree import DecisionTree
from knn import KNearestNeighbors
from hybrid_pipeline import HybridPipeline
from evaluation import evaluate_model


def main():
    print("Membaca dan memproses file Excel/CSV S&P 500...")
    df = load_and_preprocess("data/SP500_Historical_Data.csv", sample_size=100000)

    print("Menghitung indikator teknikal (RSI, MACD, Bollinger Bands)...")
    X, y = add_technical_indicators(df)

    # Split data Time Series (70% Train, 30% Test)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Total Data Latih: {len(X_train)} baris")
    print(f"Total Data Uji  : {len(X_test)} baris\n")

    print("Melatih Model Naive Bayes...")
    nb_model = GaussianNaiveBayes()
    nb_model.fit(X_train, y_train)

    print("Melatih Model Decision Tree...")
    dt_model = DecisionTree(max_depth=5)
    dt_model.fit(X_train, y_train)

    print("Menyiapkan Model K-Nearest Neighbors...")
    knn_model = KNearestNeighbors(k=5)
    knn_model.fit(X_train, y_train)

    print("Menyiapkan Pipeline Hybrid...")
    hybrid_system = HybridPipeline(
        nb_model, dt_model, knn_model, confidence_threshold=0.6
    )

    print("\n" + "=" * 50)
    print("PROSES PREDIKSI DAN EVALUASI MASING-MASING METODE")
    print("=" * 50)

    # 1. Naive Bayes Standalone
    print("1. Menjalankan Prediksi Naive Bayes...")
    # Pakai argmax karena fungsi di NB mengembalikan probabilitas
    y_pred_nb = np.argmax(nb_model.predict_proba(X_test), axis=1)
    acc_nb, prec_nb, rec_nb, f1_nb = evaluate_model(y_test, y_pred_nb)

    # 2. Decision Tree Standalone
    print("\n2. Menjalankan Prediksi Decision Tree...")
    y_pred_dt = dt_model.predict(X_test)
    acc_dt, prec_dt, rec_dt, f1_dt = evaluate_model(y_test, y_pred_dt)

    # 3. KNN Standalone
    print("\n3. Menjalankan Prediksi K-Nearest Neighbors...")
    print("   (Harap tunggu, proses ini butuh komputasi lebih lama...)")
    y_pred_knn = knn_model.predict(X_test)
    acc_knn, prec_knn, rec_knn, f1_knn = evaluate_model(y_test, y_pred_knn)

    # 4. Hybrid Pipeline
    print("\n4. Menjalankan Prediksi Model Hybrid...")
    y_pred_hybrid = hybrid_system.predict(X_test)
    acc_hybrid, prec_hybrid, rec_hybrid, f1_hybrid = evaluate_model(
        y_test, y_pred_hybrid
    )

    # ================= TABEL KOMPARASI AKHIR =================
    print("\n\n" + "=" * 70)
    print(f"{'TABEL KOMPARASI AKHIR':^70}")
    print("=" * 70)
    print(
        f"{'METODE':<22} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}"
    )
    print("-" * 70)
    print(
        f"{'Naive Bayes':<22} | {acc_nb:<10.4f} | {prec_nb:<10.4f} | {rec_nb:<10.4f} | {f1_nb:<10.4f}"
    )
    print(
        f"{'Decision Tree':<22} | {acc_dt:<10.4f} | {prec_dt:<10.4f} | {rec_dt:<10.4f} | {f1_dt:<10.4f}"
    )
    print(
        f"{'K-Nearest Neighbors':<22} | {acc_knn:<10.4f} | {prec_knn:<10.4f} | {rec_knn:<10.4f} | {f1_knn:<10.4f}"
    )
    print(
        f"{'Hybrid Pipeline':<22} | {acc_hybrid:<10.4f} | {prec_hybrid:<10.4f} | {rec_hybrid:<10.4f} | {f1_hybrid:<10.4f}"
    )
    print("=" * 70)

if __name__ == "__main__":
    main()
