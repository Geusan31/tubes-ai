"""
evaluation.py
─────────────
Evaluasi model klasifikasi binary: Accuracy, Precision, Recall,
F1-Score, Specificity, dan Confusion Matrix.

Visualisasi: matplotlib (confusion matrix disimpan sebagai PNG)
"""

import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════
#  HITUNG METRIK
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
