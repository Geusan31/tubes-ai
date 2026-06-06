from data_loader import load_and_preprocess_data
from metrics import calculate_metrics
from knn import KNearestNeighbors
from naive_bayes import GaussianNaiveBayes
from decision_tree import DecisionTree


def print_metrics(name, metrics):
    print(f"\n=== Hasil {name} ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def main():
    # Pastikan file CSV s&p500 berada di direktori yang sama
    filepath = "SP500_Historical_Data.csv"  # Ganti dengan nama file CSV S&P 500 milikmu

    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # 1. K-Nearest Neighbors
    print("\nMelatih K-Nearest Neighbors...")
    knn = KNearestNeighbors(k=15)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_metrics = calculate_metrics(y_test, knn_preds)
    print_metrics("K-Nearest Neighbors", knn_metrics)

    # 2. Gaussian Naive Bayes
    print("\nMelatih Gaussian Naive Bayes...")
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    nb_metrics = calculate_metrics(y_test, nb_preds)
    print_metrics("Naive Bayes", nb_metrics)

    # 3. Decision Tree
    # Catatan: Karena Decision Tree custom bisa sangat lambat di dataset besar,
    # kita batasi max_depth menjadi 4 untuk efisiensi komputasi.
    print("\nMelatih Decision Tree...")
    dt = DecisionTree(max_depth=4)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_metrics = calculate_metrics(y_test, dt_preds)
    print_metrics("Decision Tree", dt_metrics)

    # Perbandingan Kesimpulan
    print("\n" + "=" * 40)
    print("KESIMPULAN PERBANDINGAN MODEL")
    print("=" * 40)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 45)
    print(
        f"{'K-Nearest Neighbors':<20} | {knn_metrics['Accuracy']:.4f}     | {knn_metrics['F1-Score']:.4f}"
    )
    print(
        f"{'Naive Bayes':<20} | {nb_metrics['Accuracy']:.4f}     | {nb_metrics['F1-Score']:.4f}"
    )
    print(
        f"{'Decision Tree':<20} | {dt_metrics['Accuracy']:.4f}     | {dt_metrics['F1-Score']:.4f}"
    )


if __name__ == "__main__":
    main()
