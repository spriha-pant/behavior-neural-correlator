# =============================================================================
# src/4_evaluate.py
# PURPOSE: Load saved models, run them on test data, and generate:
#   - Accuracy, Precision, Recall, F1 score
#   - Confusion matrix (with plot)
#   - Prediction timeline plot (predicted vs actual over time)
#   - Side-by-side comparison of Linear vs Logistic Regression
#
# All plots saved to results/ folder.
# =============================================================================

import os
import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (safe for running as script)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------------------------------
# STEP 0: Load config
# --------------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config["paths"]["processed_data_dir"]
MODELS_DIR    = config["paths"]["models_dir"]
RESULTS_DIR   = config["paths"]["results_dir"]
LR_THRESHOLD  = config["model"]["lr_threshold"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# STEP 1: Load test data and models
# --------------------------------------------------------------------------

print("Loading test data and models...")

X_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

lin_reg = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
log_reg = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))

print(f"  X_test shape: {X_test.shape}")
print(f"  Actual lick rate in test set: {y_test.mean()*100:.1f}%")


# --------------------------------------------------------------------------
# STEP 2: Generate predictions
# --------------------------------------------------------------------------

# --- Linear Regression ---
# lin_reg.predict() returns continuous values (e.g., 0.23, 0.78, -0.1)
# We apply a threshold: anything > LR_THRESHOLD → 1 (lick), else → 0
lin_reg_raw  = lin_reg.predict(X_test)
lin_reg_pred = (lin_reg_raw > LR_THRESHOLD).astype(int)

# --- Logistic Regression ---
# log_reg.predict() directly returns 0 or 1 (already does thresholding at 0.5)
# log_reg.predict_proba() returns probability [P(no-lick), P(lick)]
log_reg_pred = log_reg.predict(X_test)
log_reg_prob = log_reg.predict_proba(X_test)[:, 1]  # Just P(lick)


# --------------------------------------------------------------------------
# STEP 3: Compute metrics for both models
# --------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, model_name):
    """Compute and print classification metrics."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # What these mean (quick guide):
    # Accuracy  = % of all predictions that were correct
    # Precision = of all "lick" predictions, how many were actually lick?
    # Recall    = of all actual lick timepoints, how many did we catch?
    # F1        = harmonic mean of precision and recall (overall balance)
    # For neuroscience: Recall is usually more important than Precision.
    # Missing a lick (low recall) is worse than a false alarm (low precision).

    print(f"\n  Full classification report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["no-lick", "lick"],
                                 zero_division=0))

    return {"model": model_name, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}

metrics_lin = compute_metrics(y_test, lin_reg_pred, "Linear Regression")
metrics_log = compute_metrics(y_test, log_reg_pred, "Logistic Regression")


# --------------------------------------------------------------------------
# STEP 4: Save metrics table to CSV
# --------------------------------------------------------------------------

metrics_df = pd.DataFrame([metrics_lin, metrics_log])
metrics_path = os.path.join(RESULTS_DIR, "stage1_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\n✓ Metrics saved to: {metrics_path}")


# --------------------------------------------------------------------------
# STEP 5: Confusion Matrices — side by side
# --------------------------------------------------------------------------
# Confusion matrix layout:
#
#                 Predicted: No-lick   Predicted: Lick
# Actual: No-lick      TN                  FP
# Actual: Lick         FN                  TP
#
# TN = True Negative  (correctly said no-lick)
# TP = True Positive  (correctly said lick)
# FP = False Positive (said lick, was no-lick)  — "false alarm"
# FN = False Negative (said no-lick, was lick)  — "missed lick"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Stage 1: Confusion Matrices — Test Set", fontsize=14, fontweight="bold")

for ax, y_pred, title in zip(
    axes,
    [lin_reg_pred, log_reg_pred],
    ["Linear Regression", "Logistic Regression"]
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["no-lick", "lick"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12)

plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "stage1_confusion_matrices.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"✓ Confusion matrices saved to: {cm_path}")


# --------------------------------------------------------------------------
# STEP 6: Metrics bar chart — side-by-side comparison
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

metric_names = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metric_names))
width = 0.35

bars1 = ax.bar(x - width/2,
               [metrics_lin[m] for m in metric_names],
               width, label="Linear Regression", color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width/2,
               [metrics_log[m] for m in metric_names],
               width, label="Logistic Regression", color="#DD8452", alpha=0.85)

ax.set_ylabel("Score")
ax.set_title("Stage 1: Model Comparison — Test Metrics")
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
bar_path = os.path.join(RESULTS_DIR, "stage1_metric_comparison.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"✓ Metric comparison bar chart saved to: {bar_path}")


# --------------------------------------------------------------------------
# STEP 7: Prediction timeline plot
# --------------------------------------------------------------------------
# Plot actual labels vs model predictions over time.
# This shows WHERE in time the model gets it right or wrong —
# useful for spotting systematic errors (e.g., model always lags by a few steps).

# Use a subset of the test data for readability (first 500 timepoints)
plot_n = min(500, len(y_test))
time_idx = np.arange(plot_n)

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Stage 1: Prediction Timeline (first 500 test timepoints)", fontsize=13)

# Row 0: Actual labels
axes[0].fill_between(time_idx, y_test.values[:plot_n], alpha=0.6,
                      color="steelblue", step="pre")
axes[0].set_ylabel("Actual")
axes[0].set_ylim(-0.1, 1.3)
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["no-lick", "lick"])

# Row 1: Linear Regression prediction
axes[1].fill_between(time_idx, lin_reg_pred[:plot_n], alpha=0.6,
                      color="#E45C3A", step="pre")
axes[1].set_ylabel("Lin Reg\nPrediction")
axes[1].set_ylim(-0.1, 1.3)
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["no-lick", "lick"])

# Row 2: Logistic Regression probability (continuous, more informative)
axes[2].plot(time_idx, log_reg_prob[:plot_n], color="#4CAF50",
             linewidth=0.8, alpha=0.8, label="P(lick)")
axes[2].axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
                label="Decision threshold (0.5)")
axes[2].set_ylabel("Log Reg\nP(lick)")
axes[2].set_xlabel("Timepoint (test set)")
axes[2].set_ylim(-0.05, 1.05)
axes[2].legend(loc="upper right", fontsize=8)

plt.tight_layout()
timeline_path = os.path.join(RESULTS_DIR, "stage1_prediction_timeline.png")
plt.savefig(timeline_path, dpi=150)
plt.close()
print(f"✓ Prediction timeline saved to: {timeline_path}")


# --------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------

print(f"\n{'='*50}")
print("Evaluation complete. Results saved to results/:")
print(f"  - stage1_metrics.csv")
print(f"  - stage1_confusion_matrices.png")
print(f"  - stage1_metric_comparison.png")
print(f"  - stage1_prediction_timeline.png")
print(f"\nProceed to: python src/5_predict.py (for new data)")
print(f"{'='*50}")
