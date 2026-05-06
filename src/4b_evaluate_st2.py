# =============================================================================
# src/4b_evaluate_stage2.py
# PURPOSE: Evaluate Stage 2 models (Random Forest, XGBoost) and compare
#          them against Stage 1 (Logistic Regression) as the baseline.
#
# Outputs:
#   - Printed metrics for all models
#   - stage2_metrics.csv
#   - stage2_confusion_matrices.png
#   - stage2_metric_comparison_all.png  (all 4 models side by side)
#   - stage2_timeline_<model>.png       (3-panel timeline per model)
#       Panel 1: Actual lick/no-lick from lab data
#       Panel 2: Model's predicted lick/no-lick
#       Panel 3: Superimposed — color-coded to show TP, TN, FP, FN
#
# Run from the project/ root:
#   python src/4b_evaluate_stage2.py
# =============================================================================

import os
import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# --------------------------------------------------------------------------
# STEP 0: Config
# --------------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config["paths"]["processed_data_dir"]
MODELS_DIR    = config["paths"]["models_dir"]
RESULTS_DIR   = config["paths"]["results_dir"]
LR_THRESHOLD  = config["model"]["lr_threshold"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# STEP 1: Load test data + all models
# --------------------------------------------------------------------------

print("Loading test data and models...")

X_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

# Stage 1 models (for comparison baseline)
lin_reg = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
log_reg = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))

# Stage 2 models
rf_model  = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))

print(f"  X_test shape: {X_test.shape}")
print(f"  Actual lick rate in test set: {y_test.mean()*100:.1f}%")
print(f"  Lick events in test set: {y_test.sum()} / {len(y_test)} timepoints")

# --------------------------------------------------------------------------
# STEP 2: Generate predictions for all models
# --------------------------------------------------------------------------

lin_pred  = (lin_reg.predict(X_test) > LR_THRESHOLD).astype(int)
log_pred  = log_reg.predict(X_test)
log_prob  = log_reg.predict_proba(X_test)[:, 1]

rf_pred   = rf_model.predict(X_test)
rf_prob   = rf_model.predict_proba(X_test)[:, 1]

xgb_pred  = xgb_model.predict(X_test)
xgb_prob  = xgb_model.predict_proba(X_test)[:, 1]

# --------------------------------------------------------------------------
# STEP 3: Metrics for all 4 models
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
    print(classification_report(y_true, y_pred,
                                 target_names=["no-lick", "lick"],
                                 labels=[0, 1], zero_division=0))
    return {"model": model_name, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}

all_metrics = [
    compute_metrics(y_test, lin_pred,  "Linear Regression  [Stage 1]"),
    compute_metrics(y_test, log_pred,  "Logistic Regression [Stage 1]"),
    compute_metrics(y_test, rf_pred,   "Random Forest       [Stage 2]"),
    compute_metrics(y_test, xgb_pred,  "XGBoost             [Stage 2]"),
]

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(RESULTS_DIR, "stage2_metrics.csv"), index=False)
print(f"\n✓ Metrics saved.")

# --------------------------------------------------------------------------
# STEP 4: Confusion matrices — all 4 models
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Confusion Matrices — All Models (Test Set)", fontsize=14, fontweight="bold")

model_pairs = [
    (lin_pred,  "Linear Regression [Stage 1]"),
    (log_pred,  "Logistic Regression [Stage 1]"),
    (rf_pred,   "Random Forest [Stage 2]"),
    (xgb_pred,  "XGBoost [Stage 2]"),
]

for ax, (y_pred, title) in zip(axes.flatten(), model_pairs):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["no-lick", "lick"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage2_confusion_matrices.png"), dpi=150)
plt.close()
print(f"✓ Confusion matrices saved.")

# --------------------------------------------------------------------------
# STEP 5: Metric bar chart — all 4 models side by side
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 6))
metric_names = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metric_names))
n_models = len(all_metrics)
width = 0.18
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
labels = ["Linear Reg [S1]", "Logistic Reg [S1]", "Random Forest [S2]", "XGBoost [S2]"]

for i, (m, color, label) in enumerate(zip(all_metrics, colors, labels)):
    offset = (i - n_models / 2 + 0.5) * width
    bars = ax.bar(x + offset, [m[mn] for mn in metric_names],
                  width, label=label, color=color, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_ylabel("Score")
ax.set_title("All Models — Metric Comparison (Test Set)")
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
ax.set_ylim(0, 1.18)
ax.legend(loc="upper right", fontsize=9)
ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(0.9, 1.10, "Stage 1", ha="center", fontsize=9, color="gray")
ax.text(3.1, 1.10, "Stage 2", ha="center", fontsize=9, color="gray")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage2_metric_comparison_all.png"), dpi=150)
plt.close()
print(f"✓ All-model metric comparison chart saved.")

# --------------------------------------------------------------------------
# STEP 6: Three-panel timeline visualization
# --------------------------------------------------------------------------
# For each model, we produce a figure with three panels:
#
#   Panel 1 — ACTUAL (lab data)
#     A binary bar chart showing when the mouse actually licked (1) vs not (0).
#     This is your ground truth from the CSV.
#
#   Panel 2 — PREDICTED (model output)
#     Same format, but showing what the model predicted.
#     Visually identical style so the eye can compare directly.
#
#   Panel 3 — SUPERIMPOSED (error analysis)
#     Both signals on the same plot, color-coded by outcome:
#       GREEN  = True Positive  (model said lick, mouse was licking)     ← correct
#       BLUE   = True Negative  (model said no-lick, mouse wasn't)       ← correct
#       RED    = False Positive (model said lick, mouse wasn't)          ← false alarm
#       ORANGE = False Negative (model said no-lick, mouse was licking)  ← missed lick
#
#     In a neuroscience context:
#       False Negatives (orange) = missed neural events — most costly
#       False Positives (red)    = noise attributed to licking — also bad
#
# TIMELINE SCOPE:
#   We plot 3 segments of the test set so you're not looking at 400+ rows at once:
#     - First 150 timepoints  (early test window)
#     - Middle 150 timepoints (mid test window)
#     - Last 150 timepoints   (late test window, near where licking peaks)
#   This gives a more complete picture than just plotting the first N rows.

def make_timeline_figure(y_true, y_pred, prob, model_name, filename):
    """
    Produces a 3×3 grid: 3 time segments × 3 panels (actual / predicted / superimposed).
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    prob   = np.array(prob)

    # Define 3 time windows to plot
    seg_size = 150
    starts = [0, max(0, n//2 - seg_size//2), max(0, n - seg_size)]
    seg_labels = ["Early test window", "Mid test window", "Late test window"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 11))
    fig.suptitle(f"Prediction Timeline — {model_name}", fontsize=14, fontweight="bold")

    # Column headers
    col_titles = ["Panel 1: Actual (lab data)", "Panel 2: Predicted", "Panel 3: Superimposed"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=10)

    for row, (start, seg_label) in enumerate(zip(starts, seg_labels)):
        end = min(start + seg_size, n)
        t = np.arange(end - start)

        actual = y_true[start:end]
        pred   = y_pred[start:end]
        p      = prob[start:end]

        # ── Panel 1: Actual ──────────────────────────────────────────────
        ax1 = axes[row, 0]
        ax1.fill_between(t, actual, step="pre",
                         color="#2196F3", alpha=0.75, label="Lick")
        ax1.set_ylim(-0.15, 1.3)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["no-lick", "lick"])
        ax1.set_ylabel(seg_label, fontsize=9)
        ax1.set_xlabel("Timepoints" if row == 2 else "")
        ax1.grid(axis="x", linestyle=":", alpha=0.4)

        lick_pct = actual.mean() * 100
        ax1.text(0.98, 0.92, f"lick: {lick_pct:.1f}%",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=8, color="#2196F3")

        # ── Panel 2: Predicted ───────────────────────────────────────────
        ax2 = axes[row, 1]
        ax2.fill_between(t, pred, step="pre",
                         color="#FF9800", alpha=0.75, label="Predicted lick")
        ax2.set_ylim(-0.15, 1.3)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["no-lick", "lick"])
        ax2.set_xlabel("Timepoints" if row == 2 else "")
        ax2.grid(axis="x", linestyle=":", alpha=0.4)

        pred_pct = pred.mean() * 100
        ax2.text(0.98, 0.92, f"predicted lick: {pred_pct:.1f}%",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=8, color="#FF9800")

        # ── Panel 3: Superimposed, color-coded by TP/TN/FP/FN ──────────
        ax3 = axes[row, 2]

        # Classify each timepoint
        tp = (actual == 1) & (pred == 1)   # True Positive  — correct lick detection
        tn = (actual == 0) & (pred == 0)   # True Negative  — correct no-lick
        fp = (actual == 0) & (pred == 1)   # False Positive — false alarm
        fn = (actual == 1) & (pred == 0)   # False Negative — missed lick

        # Draw actual signal as thin reference line
        ax3.step(t, actual, color="black", linewidth=0.6, alpha=0.3,
                 where="pre", label="Actual (ref)")

        # Shade background by outcome category
        # We draw a thin vertical stripe for each timepoint
        for i in range(len(t) - 1):
            x0, x1 = t[i], t[i+1]
            if tp[i]:
                color, alpha = "#4CAF50", 0.55   # Green — TP
            elif tn[i]:
                color, alpha = "#90CAF9", 0.25   # Light blue — TN (faint, expected)
            elif fp[i]:
                color, alpha = "#F44336", 0.65   # Red — FP
            else:  # fn
                color, alpha = "#FF9800", 0.75   # Orange — FN
            ax3.axvspan(x0, x1, color=color, alpha=alpha)

        # Overlay the model probability as a line (continuous confidence signal)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(t, p, color="purple", linewidth=0.7, alpha=0.6, label="P(lick)")
        ax3_twin.axhline(0.5, color="purple", linestyle="--", linewidth=0.5, alpha=0.4)
        ax3_twin.set_ylim(-0.05, 1.4)
        ax3_twin.set_ylabel("P(lick)", fontsize=8, color="purple")
        ax3_twin.tick_params(axis="y", labelcolor="purple", labelsize=7)

        ax3.set_ylim(-0.15, 1.3)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["no-lick", "lick"])
        ax3.set_xlabel("Timepoints" if row == 2 else "")
        ax3.grid(axis="x", linestyle=":", alpha=0.4)

        # Counts for this segment
        ax3.text(0.02, 0.92,
                 f"TP:{tp.sum()} TN:{tn.sum()} FP:{fp.sum()} FN:{fn.sum()}",
                 transform=ax3.transAxes, ha="left", va="top",
                 fontsize=7.5, family="monospace")

    # Legend for the superimposed panels (add once at bottom)
    legend_patches = [
        mpatches.Patch(color="#4CAF50", alpha=0.7, label="True Positive (correct lick)"),
        mpatches.Patch(color="#90CAF9", alpha=0.5, label="True Negative (correct no-lick)"),
        mpatches.Patch(color="#F44336", alpha=0.7, label="False Positive (false alarm)"),
        mpatches.Patch(color="#FF9800", alpha=0.8, label="False Negative (missed lick)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), frameon=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Timeline saved: {filename}")


# Generate timeline for each Stage 2 model
# (also generating for Logistic so you have a direct visual comparison)
make_timeline_figure(y_test, log_pred,  log_prob,
                     "Logistic Regression [Stage 1 baseline]",
                     "timeline_logistic.png")

make_timeline_figure(y_test, rf_pred,   rf_prob,
                     "Random Forest [Stage 2]",
                     "timeline_random_forest.png")

make_timeline_figure(y_test, xgb_pred,  xgb_prob,
                     "XGBoost [Stage 2]",
                     "timeline_xgboost.png")


# --------------------------------------------------------------------------
# STEP 7: Feature importance comparison plot (RF vs XGBoost)
# --------------------------------------------------------------------------

feature_names = X_test.columns.tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Feature Importances — Stage 2 Models", fontsize=13, fontweight="bold")

for ax, model, title in [
    (axes[0], rf_model,  "Random Forest"),
    (axes[1], xgb_model, "XGBoost"),
]:
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    ax.barh([feature_names[i] for i in sorted_idx],
            [importances[i] for i in sorted_idx],
            color="#4C72B0", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage2_feature_importances.png"), dpi=150)
plt.close()
print(f"✓ Feature importance plot saved.")


# --------------------------------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------------------------------

print(f"\n{'='*60}")
print("Stage 2 evaluation complete. Files saved to results/:")
print("  - stage2_metrics.csv")
print("  - stage2_confusion_matrices.png")
print("  - stage2_metric_comparison_all.png")
print("  - stage2_feature_importances.png")
print("  - timeline_logistic.png")
print("  - timeline_random_forest.png")
print("  - timeline_xgboost.png")
print(f"{'='*60}")

# Print a quick summary table
print("\nQuick comparison:")
print(f"  {'Model':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7}")
print("  " + "-"*68)
for m in all_metrics:
    print(f"  {m['model']:<30} {m['accuracy']*100:>8.1f}%"
          f" {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>7.4f}")