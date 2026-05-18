# =============================================================================
# src/4_evaluate.py
# =============================================================================

import os
import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

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
# GUARD: Warn clearly if only one class in test set
# --------------------------------------------------------------------------

unique_classes = np.unique(y_test)
if len(unique_classes) < 2:
    print("\n" + "!"*60)
    print("WARNING: Test set contains only ONE class (no lick events).")
    print("  This means all lick events fell inside the training window.")
    print("  Metrics will be misleading (100% accuracy, 0 F1).")
    print("")
    print("  RECOMMENDED FIX:")
    print("  Open 2_preprocess.py, set USE_CV = True at the top,")
    print("  then re-run steps 2 → 3 → 4.")
    print("!"*60 + "\n")

# --------------------------------------------------------------------------
# STEP 2: Generate predictions
# --------------------------------------------------------------------------

lin_reg_raw  = lin_reg.predict(X_test)
lin_reg_pred = (lin_reg_raw > LR_THRESHOLD).astype(int)
log_reg_pred = log_reg.predict(X_test)
log_reg_prob = log_reg.predict_proba(X_test)[:, 1]

# --------------------------------------------------------------------------
# STEP 3: Metrics
# --------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, model_name):
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
    print(f"\n  Full classification report:")
    # labels=[0,1] forces both classes to appear even if one is absent in predictions
    print(classification_report(y_true, y_pred,
                                 target_names=["no-lick", "lick"],
                                 labels=[0, 1],
                                 zero_division=0))
    return {"model": model_name, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}

metrics_lin = compute_metrics(y_test, lin_reg_pred, "Linear Regression")
metrics_log = compute_metrics(y_test, log_reg_pred, "Logistic Regression")

pd.DataFrame([metrics_lin, metrics_log]).to_csv(
    os.path.join(RESULTS_DIR, "stage1_metrics.csv"), index=False)
print(f"\n✓ Metrics saved.")

# --------------------------------------------------------------------------
# STEP 4: Confusion Matrices
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Stage 1: Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
for ax, y_pred, title in zip(axes, [lin_reg_pred, log_reg_pred],
                               ["Linear Regression", "Logistic Regression"]):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["no-lick", "lick"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage1_confusion_matrices.png"), dpi=150)
plt.close()
print(f"✓ Confusion matrices saved.")

# --------------------------------------------------------------------------
# STEP 5: Metrics bar chart
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
metric_names = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metric_names))
width = 0.35
bars1 = ax.bar(x - width/2, [metrics_lin[m] for m in metric_names],
               width, label="Linear Regression", color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width/2, [metrics_log[m] for m in metric_names],
               width, label="Logistic Regression", color="#DD8452", alpha=0.85)
ax.set_ylabel("Score")
ax.set_title("Stage 1: Model Comparison — Test Metrics")
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage1_metric_comparison.png"), dpi=150)
plt.close()
print(f"✓ Metric comparison chart saved.")

# --------------------------------------------------------------------------
# STEP 6: Prediction timeline
# --------------------------------------------------------------------------

plot_n = min(500, len(y_test))
time_idx = np.arange(plot_n)
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Stage 1: Prediction Timeline (first 500 test timepoints)", fontsize=13)

axes[0].fill_between(time_idx, y_test.values[:plot_n], alpha=0.6, color="steelblue", step="pre")
axes[0].set_ylabel("Actual"); axes[0].set_ylim(-0.1, 1.3)
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["no-lick", "lick"])

axes[1].fill_between(time_idx, lin_reg_pred[:plot_n], alpha=0.6, color="#E45C3A", step="pre")
axes[1].set_ylabel("Lin Reg\nPrediction"); axes[1].set_ylim(-0.1, 1.3)
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["no-lick", "lick"])

axes[2].plot(time_idx, log_reg_prob[:plot_n], color="#4CAF50", linewidth=0.8, alpha=0.8, label="P(lick)")
axes[2].axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="Threshold (0.5)")
axes[2].set_ylabel("Log Reg\nP(lick)"); axes[2].set_xlabel("Timepoint (test set)")
axes[2].set_ylim(-0.05, 1.05); axes[2].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage1_prediction_timeline.png"), dpi=150)
plt.close()
print(f"✓ Prediction timeline saved.")

print(f"\n{'='*50}")
print("Evaluation complete. Proceed to: python src/5_predict.py")
if len(unique_classes) < 2:
    print("\n⚠ But first: fix the test set issue by setting USE_CV=True in 2_preprocess.py")
print(f"{'='*50}")