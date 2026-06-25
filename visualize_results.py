"""
visualize_results.py
====================
Reads outputs from runs and generates visual plots for
the final LaTeX academic document.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})

def extension_tradeoff_plot():
    matrix_path = RESULTS_DIR / "extension_runs_matrix.csv"
    if not matrix_path.exists():
        print(f"[Warning] {matrix_path} not found. Skipping tradeoff plot.")
        return

    df = pd.read_csv(matrix_path)
    df_unique = df.drop_duplicates(subset=["Balance_Factor"]).sort_values("Balance_Factor")

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_mcc = "#1f77b4"
    sns.lineplot(data=df_unique, x="Balance_Factor", y="MCC", ax=ax1,
                 marker="o", color=color_mcc, linewidth=2.5, label="Matthews Corr. (MCC)")
    ax1.set_xlabel("Cost-Sensitive Balance Factor (Minority Loss Multiplier)", fontweight="bold")
    ax1.set_ylabel("Matthews Correlation Coefficient (MCC)", color=color_mcc, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=color_mcc)
    ax1.set_xticks(df_unique["Balance_Factor"])

    peak_row = df_unique.loc[df_unique["MCC"].idxmax()]
    ax1.annotate(f"Peak MCC: {peak_row['MCC']:.4f}\nSweet Spot",
                 xy=(peak_row["Balance_Factor"], peak_row["MCC"]),
                 xytext=(peak_row["Balance_Factor"] + 0.04, peak_row["MCC"] - 0.015),
                 arrowprops=dict(facecolor='black', shrink=0.1, width=1, headwidth=6),
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#fffacc", ec="gray", alpha=0.9))

    ax2 = ax1.twinx()
    color_fpr = "#d62728"
    sns.lineplot(data=df_unique, x="Balance_Factor", y="Missed_Fault_Rate", ax=ax2,
                 marker="s", color=color_fpr, linewidth=2.5, linestyle="--", label="Missed Fault Rate")
    ax2.set_ylabel("Missed Fault Rate (Lower is Better)", color=color_fpr, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor=color_fpr)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Analysis:\nImbalance Compensation Sweet Spot Optimization", pad=20, fontweight="bold")
    fig.tight_layout()

    save_path = FIGURES_DIR / "extension_optimization_tradeoff.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Saved parameter tradeoff plot to: {save_path}")

def comparative_matrices_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6))

    cm_inverted = np.array([[9, 300], [393, 30_297]])
    sns.heatmap(cm_inverted, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1,
                xticklabels=["Predicted Non-Flaky", "Predicted Flaky"],
                yticklabels=["Actual Non-Flaky", "Actual Flaky"])
    ax1.set_title("Trial Run 1: Flaky-Centric Labels (Inverted)\n(Trivial Strategy Over-fits Majority Class)", pad=10)

    cm_corrected = np.array([[21_271, 1_763], [119, 112]])
    sns.heatmap(cm_corrected, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax2,
                xticklabels=["Predicted Flaky/Pass", "Predicted Bug"],
                yticklabels=["Actual Flaky/Pass", "Actual Bug"])
    ax2.set_title("Corrected Clean-Room Pipeline\n(Targeting Rare Regression Faults Directly)", pad=10)

    for ax in [ax1, ax2]:
        ax.set_ylabel("True Ground Truth Label", fontweight="bold")
        ax.set_xlabel("Model Inference Prediction", fontweight="bold")

    plt.suptitle("Structural Confusion Matrix Evolution: Inverted vs. Corrected Frameworks", y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / "confusion_matrix_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Saved side-by-side confusion matrix comparisons to: {save_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("SEABORN PLOT GENERATION")
    print("=" * 80)

    extension_tradeoff_plot()
    comparative_matrices_plot()

    print("\n[Process Complete] Vector graphics generated inside the ./figures/ directory.")