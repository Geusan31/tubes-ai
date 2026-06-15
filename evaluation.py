"""
evaluation.py
─────────────
Evaluasi model:
  - Regresi : R², RMSE, MAE, MAPE
  - Klasifikasi: Accuracy, Precision, Recall, F1, Specificity

Visualisasi: matplotlib (confusion matrix / prediction plot → PNG)
"""

import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════
#  HELPERS INTERNAL
# ══════════════════════════════════════════════════════════════════

def _roc_auc(y_true_bin, scores):
    """
    Hitung AUC-ROC dari scratch menggunakan trapezoid rule.

    y_true_bin : array biner (0/1)
    scores     : skor kontinyu (lebih tinggi = lebih likely positif)

    Return: (auc, fpr_array, tpr_array)
    """
    y_true_bin = np.array(y_true_bin, dtype=np.int32)
    scores     = np.array(scores,     dtype=np.float64)

    n_pos = y_true_bin.sum()
    n_neg = len(y_true_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Urutkan score menurun
    order      = np.argsort(scores)[::-1]
    y_sorted   = y_true_bin[order]

    tpr_list, fpr_list = [0.0], [0.0]
    tp = fp = 0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    tpr_list.append(1.0)
    fpr_list.append(1.0)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    auc = float(np.trapezoid(tpr, fpr))
    return auc, fpr, tpr


def _direction_from_price(y_price, current_close):
    """
    Konversi harga prediksi/aktual ke arah: 1=naik (Buy), 0=turun/sama (Sell).
    Dasar perbandingan = harga hari ini (current_close).
    """
    return (np.array(y_price) > np.array(current_close)).astype(np.int32)


def _clf_metrics(y_true_bin, y_pred_bin):
    """Hitung accuracy, precision, recall, F1 dari array biner."""
    TP = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    TN = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    FP = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    FN = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    n  = len(y_true_bin)

    acc   = (TP + TN) / n if n > 0 else 0.0
    prec  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec   = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return acc, prec, rec, f1, spec, TP, TN, FP, FN


# ══════════════════════════════════════════════════════════════════
#  EVALUASI LENGKAP (REGRESI + KLASIFIKASI ARAH + AUC)
# ══════════════════════════════════════════════════════════════════

def evaluate_full(y_true_price, y_pred_price, current_close,
                  model_name="Model", auc_cv=None):
    """
    Evaluasi lengkap satu model: metrik regresi + metrik klasifikasi arah + AUC.

    Parameter:
      y_true_price  : harga aktual hari berikutnya
      y_pred_price  : harga prediksi hari berikutnya
      current_close : harga penutupan hari ini (basis arah)
      model_name    : label untuk print
      auc_cv        : (mean_auc, std_auc) dari cross-validation, atau None

    Return: dict dengan semua metrik
    """
    y_true_price  = np.array(y_true_price,  dtype=np.float64)
    y_pred_price  = np.array(y_pred_price,  dtype=np.float64)
    current_close = np.array(current_close, dtype=np.float64)

    # ── Regresi ──────────────────────────────────────────────────
    n      = len(y_true_price)
    ss_res = np.sum((y_true_price - y_pred_price) ** 2)
    ss_tot = np.sum((y_true_price - y_true_price.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse   = float(np.sqrt(ss_res / n))
    mae    = float(np.mean(np.abs(y_true_price - y_pred_price)))
    mape   = float(np.mean(np.abs(
        (y_true_price - y_pred_price) / (np.abs(y_true_price) + 1e-10)
    )) * 100)

    # ── Arah (klasifikasi) ────────────────────────────────────────
    true_dir = _direction_from_price(y_true_price, current_close)
    pred_dir = _direction_from_price(y_pred_price, current_close)
    scores   = y_pred_price - current_close   # skor kontinu untuk AUC

    acc, prec, rec, f1, spec, TP, TN, FP, FN = _clf_metrics(true_dir, pred_dir)

    # ── AUC-test ──────────────────────────────────────────────────
    auc_test, _, _ = _roc_auc(true_dir, scores)

    # ── Print ─────────────────────────────────────────────────────
    w   = 52
    bar = "=" * w
    print(f"\n{bar}")
    print(f"  EVALUASI LENGKAP: {model_name}")
    print(bar)
    print(f"  [REGRESI]")
    print(f"    R²        : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"    RMSE      : ${rmse:.4f}")
    print(f"    MAE       : ${mae:.4f}")
    print(f"    MAPE      : {mape:.2f}%")
    print(f"  [KLASIFIKASI ARAH  Buy=naik / Sell=turun]")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    Specificity:{spec:.4f}")
    print(f"    TP={TP:,}  TN={TN:,}  FP={FP:,}  FN={FN:,}")
    print(f"  [AUC]")
    print(f"    AUC-test  : {auc_test:.4f}")
    if auc_cv is not None:
        mean_cv, std_cv = auc_cv
        print(f"    AUC-CV    : {mean_cv:.4f} ± {std_cv:.4f}  (5-fold time-series)")
    else:
        print(f"    AUC-CV    : —")
    print(bar)

    return dict(r2=r2, rmse=rmse, mae=mae, mape=mape,
                accuracy=acc, precision=prec, recall=rec, f1=f1,
                specificity=spec, auc_test=auc_test,
                auc_cv_mean=auc_cv[0] if auc_cv else None,
                auc_cv_std =auc_cv[1] if auc_cv else None)


# ══════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION AUC (5-fold time-series)
# ══════════════════════════════════════════════════════════════════

def compute_cv_auc(model, X, y_price, current_close, n_folds=5, verbose=False):
    """
    Hitung AUC menggunakan k-fold time-series cross-validation.

    Menggunakan walk-forward expanding window:
      Fold 1: train [0 .. 1/k), test [1/k .. 2/k)
      Fold 2: train [0 .. 2/k), test [2/k .. 3/k)
      ...

    Parameter:
      model         : object dengan metode fit(X, y) dan predict(X)
      X             : fitur (sudah dinormalisasi)
      y_price       : target harga (float)
      current_close : harga hari ini per sampel
      n_folds       : jumlah fold (default 5)

    Return: (mean_auc, std_auc)
    """
    n         = len(X)
    fold_size = n // (n_folds + 1)   # ukuran setiap fold
    aucs      = []

    for fold in range(1, n_folds + 1):
        train_end  = fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        if test_end <= test_start:
            break

        X_tr  = X[:train_end]
        y_tr  = y_price[:train_end]
        X_te  = X[test_start:test_end]
        y_te  = y_price[test_start:test_end]
        cc_te = current_close[test_start:test_end]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        true_dir = _direction_from_price(y_te, cc_te)
        scores   = y_pred - cc_te

        if true_dir.sum() == 0 or true_dir.sum() == len(true_dir):
            continue   # skip fold dengan satu kelas saja

        auc, _, _ = _roc_auc(true_dir, scores)
        aucs.append(auc)
        if verbose:
            print(f"    Fold {fold}: AUC = {auc:.4f}  "
                  f"(train={train_end:,}, test={test_end-test_start:,})")

    if len(aucs) == 0:
        return 0.5, 0.0
    return float(np.mean(aucs)), float(np.std(aucs))


# ══════════════════════════════════════════════════════════════════
#  EVALUASI NAIVE BAYES (KLASIFIKASI MURNI)
# ══════════════════════════════════════════════════════════════════

def evaluate_nb(y_true_dir, y_pred_dir, nb_proba, model_name="Naive Bayes", auc_cv=None):
    """
    Evaluasi Naive Bayes sebagai classifier arah Buy/Sell.

    Parameter:
      y_true_dir : label aktual (0=Sell, 1=Buy)
      y_pred_dir : label prediksi (0=Sell, 1=Buy)
      nb_proba   : probabilitas posterior shape (n, 2) → kolom 1 = P(Buy)
      auc_cv     : (mean, std) dari cross-validation, atau None

    Return: dict metrik
    """
    y_true_dir = np.array(y_true_dir, dtype=np.int32)
    y_pred_dir = np.array(y_pred_dir, dtype=np.int32)
    scores     = np.array(nb_proba)[:, 1]   # P(Buy) sebagai skor AUC

    acc, prec, rec, f1, spec, TP, TN, FP, FN = _clf_metrics(y_true_dir, y_pred_dir)
    auc_test, _, _ = _roc_auc(y_true_dir, scores)

    w   = 52
    bar = "=" * w
    print(f"\n{bar}")
    print(f"  EVALUASI KLASIFIKASI: {model_name}")
    print(bar)
    print(f"  [KLASIFIKASI ARAH  Buy=1 / Sell=0]")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    Specificity:{spec:.4f}")
    print(f"    TP={TP:,}  TN={TN:,}  FP={FP:,}  FN={FN:,}")
    print(f"  [AUC]")
    print(f"    AUC-test  : {auc_test:.4f}")
    if auc_cv is not None:
        mean_cv, std_cv = auc_cv
        print(f"    AUC-CV    : {mean_cv:.4f} ± {std_cv:.4f}  (5-fold time-series)")
    else:
        print(f"    AUC-CV    : —")
    print(bar)

    return dict(r2=None, rmse=None, mae=None, mape=None,
                accuracy=acc, precision=prec, recall=rec, f1=f1,
                specificity=spec, auc_test=auc_test,
                auc_cv_mean=auc_cv[0] if auc_cv else None,
                auc_cv_std =auc_cv[1] if auc_cv else None)


def compute_cv_auc_nb(nb_class, nb_kwargs, X, y_dir, n_folds=5, verbose=False):
    """
    5-fold time-series CV AUC khusus untuk Naive Bayes classifier.

    Parameter:
      nb_class  : kelas GaussianNaiveBayes
      nb_kwargs : dict kwargs konstruktor (misal use_uniform_prior=True)
      X         : fitur
      y_dir     : label biner (0/1)
      n_folds   : jumlah fold

    Return: (mean_auc, std_auc)
    """
    n         = len(X)
    fold_size = n // (n_folds + 1)
    aucs      = []

    for fold in range(1, n_folds + 1):
        train_end  = fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)
        if test_end <= test_start:
            break

        X_tr = X[:train_end];  y_tr = y_dir[:train_end]
        X_te = X[test_start:test_end]; y_te = y_dir[test_start:test_end]

        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue

        model = nb_class(**nb_kwargs)
        model.fit(X_tr, y_tr)
        proba  = model.predict_proba(X_te)
        scores = proba[:, 1]   # P(Buy)

        auc, _, _ = _roc_auc(y_te, scores)
        aucs.append(auc)
        if verbose:
            print(f"    Fold {fold}: AUC = {auc:.4f}  "
                  f"(train={train_end:,}, test={test_end-test_start:,})")

    if not aucs:
        return 0.5, 0.0
    return float(np.mean(aucs)), float(np.std(aucs))


# ══════════════════════════════════════════════════════════════════
#  CHART KOMPREHENSIF SEMUA MODEL
# ══════════════════════════════════════════════════════════════════

def plot_all_models_chart(nb_data, dt_data, knn_data, hybrid_data,
                          save_path="output/all_models_predictions.png",
                          n_points=300):
    """
    Buat chart komprehensif 4 panel untuk semua model.

    nb_data     : dict {y_true_dir, y_pred_dir, nb_proba, current_close, y_true_price}
    dt_data     : dict {y_true_price, y_pred_price, current_close}
    knn_data    : dict {y_true_price, y_pred_price, current_close}
    hybrid_data : dict {y_true_price, y_pred_price, current_close}
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig = plt.figure(figsize=(16, 20))
    fig.patch.set_facecolor("#F0F4F8")
    fig.suptitle("Hasil Prediksi Semua Model — Hybrid Pipeline",
                 fontsize=15, fontweight="bold", y=0.98)

    # ── Panel 1: Naive Bayes (klasifikasi arah) ───────────────────
    ax1 = fig.add_subplot(4, 1, 1)
    y_true_dir  = np.array(nb_data["y_true_dir"])[:n_points]
    y_pred_dir  = np.array(nb_data["y_pred_dir"])[:n_points]
    y_price_ref = np.array(nb_data["y_true_price"])[:n_points]
    x_idx       = np.arange(len(y_price_ref))

    ax1.plot(x_idx, y_price_ref, color="#455A64", linewidth=1.0,
             alpha=0.5, label="Harga Aktual", zorder=1)

    # Warna marker: hijau=benar, merah=salah
    correct   = y_true_dir == y_pred_dir
    buy_ok    = correct & (y_pred_dir == 1)
    sell_ok   = correct & (y_pred_dir == 0)
    buy_wrong = (~correct) & (y_pred_dir == 1)
    sell_wrong= (~correct) & (y_pred_dir == 0)

    ax1.scatter(x_idx[buy_ok],     y_price_ref[buy_ok],
                c="#43A047", marker="^", s=25, label="Pred Buy ✓",  zorder=3)
    ax1.scatter(x_idx[sell_ok],    y_price_ref[sell_ok],
                c="#1E88E5", marker="v", s=25, label="Pred Sell ✓", zorder=3)
    ax1.scatter(x_idx[buy_wrong],  y_price_ref[buy_wrong],
                c="#E53935", marker="^", s=25, label="Pred Buy ✗",  zorder=3, alpha=0.6)
    ax1.scatter(x_idx[sell_wrong], y_price_ref[sell_wrong],
                c="#FB8C00", marker="v", s=25, label="Pred Sell ✗", zorder=3, alpha=0.6)

    # Hitung metrik
    acc = np.mean(correct)
    scores = np.array(nb_data["nb_proba"])[:n_points, 1]
    auc, fpr, tpr = _roc_auc(y_true_dir, scores)
    ax1.set_title(f"Naive Bayes — Prediksi Arah (Accuracy={acc*100:.2f}%  AUC={auc:.4f})",
                  fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, ncol=4)
    ax1.set_ylabel("Harga ($)")
    ax1.set_facecolor("#FAFAFA")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Decision Tree (regresi harga) ────────────────────
    ax2 = fig.add_subplot(4, 1, 2)
    y_true_p = np.array(dt_data["y_true_price"])[:n_points]
    y_pred_p = np.array(dt_data["y_pred_price"])[:n_points]
    cc       = np.array(dt_data["current_close"])[:n_points]

    ax2.plot(y_true_p, color="#1565C0", linewidth=1.5, label="Harga Aktual")
    ax2.plot(y_pred_p, color="#E53935", linewidth=1.5, linestyle="--", label="Prediksi DT")
    ax2.fill_between(range(len(y_true_p)), y_true_p, y_pred_p, alpha=0.15, color="#E53935")

    ss_res = np.sum((y_true_p - y_pred_p)**2)
    ss_tot = np.sum((y_true_p - y_true_p.mean())**2)
    r2_dt = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    dir_dt_acc = np.mean(_direction_from_price(y_true_p, cc) ==
                         _direction_from_price(y_pred_p, cc))
    ax2.set_title(f"Decision Tree Regressor — R²={r2_dt:.4f}  "
                  f"Acc Arah={dir_dt_acc*100:.2f}%",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_ylabel("Harga ($)")
    ax2.set_facecolor("#FAFAFA")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: KNN (regresi harga) ──────────────────────────────
    ax3 = fig.add_subplot(4, 1, 3)
    y_true_k = np.array(knn_data["y_true_price"])[:n_points]
    y_pred_k = np.array(knn_data["y_pred_price"])[:n_points]
    cc_k     = np.array(knn_data["current_close"])[:n_points]

    ax3.plot(y_true_k, color="#1565C0", linewidth=1.5, label="Harga Aktual")
    ax3.plot(y_pred_k, color="#7B1FA2", linewidth=1.5, linestyle="--", label="Prediksi KNN")
    ax3.fill_between(range(len(y_true_k)), y_true_k, y_pred_k, alpha=0.15, color="#7B1FA2")

    ss_res = np.sum((y_true_k - y_pred_k)**2)
    ss_tot = np.sum((y_true_k - y_true_k.mean())**2)
    r2_knn = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    dir_knn_acc = np.mean(_direction_from_price(y_true_k, cc_k) ==
                          _direction_from_price(y_pred_k, cc_k))
    ax3.set_title(f"KNN Regressor — R²={r2_knn:.4f}  "
                  f"Acc Arah={dir_knn_acc*100:.2f}%",
                  fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_ylabel("Harga ($)")
    ax3.set_facecolor("#FAFAFA")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Hybrid Ensemble ──────────────────────────────────
    ax4 = fig.add_subplot(4, 1, 4)
    y_true_h = np.array(hybrid_data["y_true_price"])[:n_points]
    y_pred_h = np.array(hybrid_data["y_pred_price"])[:n_points]
    cc_h     = np.array(hybrid_data["current_close"])[:n_points]

    ax4.plot(y_true_h, color="#1565C0", linewidth=1.5, label="Harga Aktual")
    ax4.plot(y_pred_h, color="#00897B", linewidth=1.5, linestyle="--", label="Prediksi Hybrid")
    ax4.fill_between(range(len(y_true_h)), y_true_h, y_pred_h, alpha=0.15, color="#00897B")

    ss_res = np.sum((y_true_h - y_pred_h)**2)
    ss_tot = np.sum((y_true_h - y_true_h.mean())**2)
    r2_hy = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    dir_hy_acc = np.mean(_direction_from_price(y_true_h, cc_h) ==
                         _direction_from_price(y_pred_h, cc_h))
    ax4.set_title(f"Hybrid Ensemble (DT+KNN) — R²={r2_hy:.4f}  "
                  f"Acc Arah={dir_hy_acc*100:.2f}%",
                  fontsize=11, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.set_xlabel("Sampel Test ke-n")
    ax4.set_ylabel("Harga ($)")
    ax4.set_facecolor("#FAFAFA")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Chart semua model disimpan → {save_path}")


def plot_roc_curves(roc_data: dict, save_path="output/roc_curves.png"):
    """
    Plot ROC curve untuk semua model dalam satu gambar.
    roc_data: {model_name: (y_true_bin, scores)}
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#F0F4F8")

    colors = ["#E53935", "#1E88E5", "#7B1FA2", "#00897B"]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)", alpha=0.5)

    for (name, (y_true, scores)), color in zip(roc_data.items(), colors):
        auc, fpr, tpr = _roc_auc(np.array(y_true), np.array(scores))
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  (AUC={auc:.4f})")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve — Semua Model", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ ROC curves disimpan → {save_path}")


# ══════════════════════════════════════════════════════════════════
#  METRIK REGRESI (dipertahankan — dipakai untuk weight fitting)
# ══════════════════════════════════════════════════════════════════

def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Hitung R², RMSE, MAE, MAPE dari scratch (tanpa sklearn).

    Return: (r2, rmse, mae, mape)
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    n       = len(y_true)
    ss_res  = np.sum((y_true - y_pred) ** 2)
    ss_tot  = np.sum((y_true - y_true.mean()) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse    = np.sqrt(ss_res / n)
    mae     = np.mean(np.abs(y_true - y_pred))
    mape    = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

    w   = 48
    bar = "=" * w
    print(f"\n{bar}")
    print(f"  EVALUASI REGRESI: {model_name}")
    print(bar)
    print(f"  R²   : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"  RMSE : ${rmse:.4f}")
    print(f"  MAE  : ${mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(bar)

    return r2, rmse, mae, mape


def plot_all_predictions(results: dict,
                         save_path="output/prediction_chart.png",
                         n_points=500):
    """
    Dua bagian chart:
      Atas  : Overlay semua prediksi vs actual dalam satu axes
      Bawah : Grid subplot individual per model (actual vs predicted)

    results: {model_name: (y_true, y_pred)}
    """
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True
    )

    names    = list(results.keys())
    n_models = len(names)
    colors   = ["#E53935", "#43A047", "#FB8C00", "#8E24AA"]   # merah, hijau, oranye, ungu

    # Layout: 1 overlay (atas) + n_models subplots (bawah)
    n_rows = 1 + n_models
    fig    = plt.figure(figsize=(14, 4 + 3.5 * n_models))
    gs     = fig.add_gridspec(n_rows, 1, hspace=0.5)

    # ── Panel atas: Overlay ────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0])

    # Ambil y_true dari model pertama (sama untuk semua)
    y_true_all = np.array(list(results.values())[0][0])[:n_points]
    ax_top.plot(y_true_all, label="Actual", color="#1565C0",
                linewidth=2.0, zorder=5)

    for i, (name, (_, y_pred)) in enumerate(results.items()):
        y_pred_p = np.array(y_pred)[:n_points]
        ax_top.plot(y_pred_p, label=name, color=colors[i % len(colors)],
                    linewidth=1.2, linestyle="--", alpha=0.8)

    ax_top.set_title("Overlay: Actual vs Semua Prediksi Model",
                     fontsize=12, fontweight="bold")
    ax_top.set_ylabel("Price ($)")
    ax_top.legend(fontsize=8, loc="upper left", ncol=2)
    ax_top.grid(True, alpha=0.3)

    # ── Panel bawah: Per-model subplots ───────────────────────────
    for i, (name, (y_true, y_pred)) in enumerate(results.items()):
        y_true_s = np.array(y_true)[:n_points]
        y_pred_s = np.array(y_pred)[:n_points]

        ss_res = np.sum((y_true_s - y_pred_s) ** 2)
        ss_tot = np.sum((y_true_s - y_true_s.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax = fig.add_subplot(gs[i + 1])
        ax.plot(y_true_s, label="Actual",    color="#1565C0",
                linewidth=1.8, zorder=4)
        ax.plot(y_pred_s, label="Predicted", color=colors[i % len(colors)],
                linewidth=1.2, linestyle="--", alpha=0.85)

        # Shaded area (error)
        ax.fill_between(range(len(y_true_s)), y_true_s, y_pred_s,
                        alpha=0.12, color=colors[i % len(colors)])

        ax.set_title(f"{name}   R² = {r2:.4f}  ({r2*100:.2f}%)",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Price ($)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        if i == n_models - 1:
            ax.set_xlabel("Sample index (test set)")

    fig.suptitle("Prediksi Harga Saham — NB · DT · KNN · Hybrid",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Chart prediksi disimpan → {save_path}")


# dipertahankan untuk kompatibilitas
def plot_regression_predictions(results: dict,
                                 save_path="output/regression_predictions.png",
                                 n_points=500):
    plot_all_predictions(results, save_path=save_path, n_points=n_points)


# ══════════════════════════════════════════════════════════════════
#  METRIK KLASIFIKASI (dipertahankan untuk kompatibilitas)
# ══════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Hitung semua metrik evaluasi binary classification dari scratch.
    Positif = 1 (Buy), Negatif = 0 (Sell)

    Return: (accuracy, precision, recall, f1_score)
    """
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = len(y_true)
    accuracy = (TP + TN) / n if n > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    w = 44
    bar = "=" * w
    print(f"\n{bar}")
    print(f"  EVALUASI: {model_name}")
    print(bar)
    print(f"  Accuracy    : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1-Score    : {f1_score:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  TP={TP:,}  TN={TN:,}  FP={FP:,}  FN={FN:,}")
    print(bar)

    return accuracy, precision, recall, f1_score


# ══════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX — matplotlib
# ══════════════════════════════════════════════════════════════════


def plot_all_confusion_matrices(
    results: dict, save_path="output/confusion_matrices.png"
):
    """
    Plot confusion matrix semua model dalam satu figure PNG.
    results: dict {model_name: (y_true, y_pred)}
    """
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Confusion Matrix — Hybrid Pipeline",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    if n_models == 1:
        axes = [axes]

    colors = [["#C8E6C9", "#FFCDD2"], ["#FFCDD2", "#C8E6C9"]]

    for ax, (name, (y_true, y_pred)) in zip(axes, results.items()):
        y_true = np.array(y_true, dtype=np.int32)
        y_pred = np.array(y_pred, dtype=np.int32)

        TP = int(np.sum((y_true == 1) & (y_pred == 1)))
        TN = int(np.sum((y_true == 0) & (y_pred == 0)))
        FP = int(np.sum((y_true == 0) & (y_pred == 1)))
        FN = int(np.sum((y_true == 1) & (y_pred == 0)))
        total = TP + TN + FP + FN

        mat = [[TN, FP], [FN, TP]]
        labels = [["TN", "FP"], ["FN", "TP"]]

        for i in range(2):
            for j in range(2):
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (j + 0.05, 1 - i + 0.05),
                        0.9,
                        0.9,
                        boxstyle="round,pad=0.02",
                        color=colors[i][j],
                        zorder=1,
                    )
                )
                val = mat[i][j]
                pct = val / total * 100 if total > 0 else 0
                ax.text(
                    j + 0.5,
                    1 - i + 0.62,
                    f"{val:,}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color="#212121",
                    zorder=2,
                )
                ax.text(
                    j + 0.5,
                    1 - i + 0.38,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#616161",
                    zorder=2,
                )
                ax.text(
                    j + 0.5,
                    1 - i + 0.18,
                    labels[i][j],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#9E9E9E",
                    fontstyle="italic",
                    zorder=2,
                )

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Sell (0)", "Buy (1)"])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Buy (1)", "Sell (0)"])
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#F8F9FA")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Confusion matrices disimpan → {save_path}")


# ══════════════════════════════════════════════════════════════════
#  SIMPAN CSV
# ══════════════════════════════════════════════════════════════════


def save_summary_csv(rows, save_path="output/evaluation_summary.csv"):
    """Simpan tabel komparasi ke CSV (tanpa library csv eksternal)."""
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )
    header = "Model,Accuracy,Precision,Recall,F1-Score\n"
    lines = [header]
    for name, acc, prec, rec, f1 in rows:
        lines.append(f"{name},{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f}\n")
    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  ✓ Summary disimpan → {save_path}")
