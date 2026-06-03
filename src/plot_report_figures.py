# =============================================================================
# src/plot_report_figures.py
# PURPOSE: Generate publication-quality comparison figures for the report.
#
# Figures produced:
#   report_radar.png            — radar/spider chart: all models × all metrics
#   report_heatmap.png          — heatmap: models × metrics (colour = score)
#   report_pr_scatter.png       — precision vs recall scatter, one dot per model
#   report_f1_progression.png   — F1 score bar showing S1→S2→S3 improvement arc
#   report_version_history.png  — metric trends across your manual config versions
#                                 (requires results/version_history.csv — see below)
#
# HOW TO USE VERSION HISTORY:
#   Create results/version_history.csv with this structure:
#     version, label, accuracy, precision, recall, f1, notes
#   Fill one row per configuration you tested. Example rows are pre-filled
#   at the bottom of this file — edit them to match your actual numbers.
#
# Run from project/ root:
#   python src/plot_report_figures.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# DATA: load from saved metrics CSVs + merge into one master table
# =============================================================================
# Each evaluate script saves a *_metrics.csv. We load whichever exist.
# If a file is missing (e.g. you haven't run Stage 3 yet), it's skipped.

def load_metrics_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

s1 = load_metrics_csv(os.path.join(RESULTS_DIR, "stage1_metrics.csv"))
s2 = load_metrics_csv(os.path.join(RESULTS_DIR, "stage2_metrics.csv"))
s3 = load_metrics_csv(os.path.join(RESULTS_DIR, "stage3_metrics.csv"))

frames = [df for df in [s1, s2, s3] if df is not None]
if not frames:
    raise FileNotFoundError(
        "No metrics CSV files found in results/. "
        "Run the evaluate scripts first."
    )

all_df = pd.concat(frames, ignore_index=True)

# Clean up model name column (strip whitespace, shorten for display)
all_df["model"] = all_df["model"].str.strip()

# Keep only one row per unique model name (last occurrence wins — most recent run)
all_df = all_df.drop_duplicates(subset="model", keep="last").reset_index(drop=True)

print(f"Models loaded ({len(all_df)}):")
for _, row in all_df.iterrows():
    print(f"  {row['model']:<40}  F1={row['f1']:.3f}  Recall={row['recall']:.3f}")

# Short display names for chart labels
SHORT_NAMES = {
    "Linear Regression  [Stage 1]":        "Lin Reg\n[S1]",
    "Linear Regression  [S1]":             "Lin Reg\n[S1]",
    "Logistic Regression [Stage 1]":       "Log Reg\n[S1]",
    "Logistic Regression [S1]":            "Log Reg\n[S1]",
    "Random Forest       [Stage 2]":       "Rand\nForest [S2]",
    "Random Forest       [S2]":            "Rand\nForest [S2]",
    "XGBoost             [Stage 2]":       "XGBoost\n[S2]",
    "XGBoost             [S2]":            "XGBoost\n[S2]",
    "LSTM raw [Stage 3]":                  "LSTM\nRaw [S3]",
    "LSTM postprocessed [Stage 3]":        "LSTM\nPost [S3]",
    "LSTM [Stage 3]":                      "LSTM\n[S3]",
}
all_df["short"] = all_df["model"].map(SHORT_NAMES).fillna(all_df["model"])

STAGE_COLORS = {
    "[S1]": "#4C72B0",
    "[S2]": "#55A868",
    "[S3]": "#9C27B0",
}

def get_stage_color(model_name):
    for key, col in STAGE_COLORS.items():
        if key in model_name or key.replace("[","").replace("]","") in model_name:
            return col
    return "#888888"

all_df["color"] = all_df["model"].apply(get_stage_color)

metrics    = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

# =============================================================================
# FIGURE 1 — RADAR / SPIDER CHART
# =============================================================================
# Each model is a polygon on a circular grid.
# The further out each vertex, the higher the score on that metric.
# Great for seeing each model's "shape" at a glance.

N_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, N_metrics, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#FAFAFA")

for _, row in all_df.iterrows():
    values = [row[m] for m in metrics]
    values += values[:1]
    color = row["color"]
    ax.plot(angles, values, linewidth=2.0, color=color, label=row["short"].replace("\n", " "))
    ax.fill(angles, values, alpha=0.08, color=color)
    # Mark the vertex points
    ax.scatter(angles[:-1], values[:-1], s=50, color=color, zorder=5)

# Grid and labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=13, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
ax.spines["polar"].set_color("#CCCCCC")
ax.grid(color="#DDDDDD", linewidth=0.8)

ax.set_title("Model Performance Comparison\n(Radar Chart)", fontsize=14,
             fontweight="bold", pad=25)

legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                   fontsize=9, framealpha=0.9, title="Model", title_fontsize=10)

# Stage legend patches
stage_patches = [mpatches.Patch(color=c, label=s, alpha=0.85)
                 for s, c in STAGE_COLORS.items()]
fig.legend(handles=stage_patches, loc="lower center", ncol=3,
           fontsize=10, title="Stage", title_fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "report_radar.png"), dpi=180, bbox_inches="tight")
plt.close()
print("✓ report_radar.png") # CHANGE

# =============================================================================
# FIGURE 2 — METRIC HEATMAP
# =============================================================================
# Rows = models, columns = metrics. Colour intensity = score.
# Immediately shows which model excels on which metric.
# Numbers are printed inside each cell.

fig, ax = plt.subplots(figsize=(10, max(4, len(all_df) * 0.75)))
fig.patch.set_facecolor("#FAFAFA")

data_matrix = all_df[metrics].values
short_labels = all_df["short"].str.replace("\n", " ").tolist()

im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metric_labels, fontsize=12, fontweight="bold")
ax.set_yticks(range(len(all_df)))
ax.set_yticklabels(short_labels, fontsize=10)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# Print score value inside each cell
for i in range(len(all_df)):
    for j, m in enumerate(metrics):
        val = data_matrix[i, j]
        text_color = "black" if 0.3 < val < 0.75 else "white"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=11, color=text_color, fontweight="bold")

# Stage color strip on the left side
for i, (_, row) in enumerate(all_df.iterrows()):
    ax.add_patch(FancyBboxPatch(
        (-0.48, i - 0.45), 0.08, 0.9,
        boxstyle="round,pad=0.01",
        facecolor=row["color"], linewidth=0
    ))

plt.colorbar(im, ax=ax, label="Score", shrink=0.8, pad=0.02)
ax.set_title("Model × Metric Heatmap", fontsize=14, fontweight="bold", pad=40)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "report_heatmap.png"), dpi=180, bbox_inches="tight")
plt.close()
print("✓ report_heatmap.png") # CHANGE

# =============================================================================
# FIGURE 3 — PRECISION vs RECALL SCATTER
# =============================================================================
# Each model is one dot. X = Recall, Y = Precision.
# F1 iso-curves are drawn as background contours.
# A model in the top-right corner is best.
# The trade-off between catching licks (recall) vs being correct (precision)
# is visible immediately.

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("#FAFAFA")

# F1 iso-curves (dashed background lines)
r_grid = np.linspace(0.01, 1.0, 300)
for f1_val in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
    # P = F1 * R / (2R - F1), only where denominator > 0
    denom = 2 * r_grid - f1_val
    with np.errstate(divide="ignore", invalid="ignore"):
        p_grid = np.where(denom > 0, f1_val * r_grid / denom, np.nan)
    mask = (p_grid >= 0) & (p_grid <= 1)
    ax.plot(r_grid[mask], p_grid[mask], color="#CCCCCC",
            linewidth=0.8, linestyle="--", zorder=0)
    # Label the curve at the right edge
    idx = np.where(mask)[0]
    if len(idx) > 0:
        last = idx[-1]
        ax.text(r_grid[last] + 0.01, p_grid[last], f"F1={f1_val}",
                fontsize=7, color="#AAAAAA", va="center")

# Plot each model
for _, row in all_df.iterrows():
    ax.scatter(row["recall"], row["precision"],
               s=200, color=row["color"], zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.annotate(row["short"].replace("\n", " "),
                xy=(row["recall"], row["precision"]),
                xytext=(8, 4), textcoords="offset points",
                fontsize=9, color=row["color"], fontweight="bold")

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Recall  (fraction of real lick events caught)", fontsize=12)
ax.set_ylabel("Precision  (fraction of predicted licks that are real)", fontsize=12)
ax.set_title("Precision vs Recall Trade-off\n(dashed lines = F1 iso-curves)",
             fontsize=13, fontweight="bold")
ax.grid(linestyle=":", alpha=0.5)

stage_patches = [mpatches.Patch(color=c, label=s, alpha=0.9)
                 for s, c in STAGE_COLORS.items()]
ax.legend(handles=stage_patches, title="Stage", fontsize=10, title_fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "report_pr_scatter.png"), dpi=180, bbox_inches="tight")
plt.close()
print("✓ report_pr_scatter.png") # CHANGE

# =============================================================================
# FIGURE 4 — F1 PROGRESSION (horizontal lollipop chart)
# =============================================================================
# Shows the arc of improvement from Stage 1 → Stage 2 → Stage 3.
# Lollipop style (dot + line to zero) is cleaner than a bar chart for this.

fig, ax = plt.subplots(figsize=(10, max(4, len(all_df) * 0.65)))
fig.patch.set_facecolor("#FAFAFA")

y_positions = range(len(all_df))
f1_vals  = all_df["f1"].values
rec_vals = all_df["recall"].values
colors   = all_df["color"].values
labels   = all_df["short"].str.replace("\n", " ").tolist()

for i, (y, f1, rec, col, lbl) in enumerate(
        zip(y_positions, f1_vals, rec_vals, colors, labels)):
    # F1 lollipop
    ax.hlines(y, 0, f1, colors=col, linewidth=2.5, alpha=0.7)
    ax.scatter(f1, y, s=120, color=col, zorder=5, edgecolors="white", linewidths=1.2)
    ax.text(f1 + 0.015, y, f"{f1:.3f}", va="center", fontsize=9,
            color=col, fontweight="bold")

    # Recall as a dimmer secondary dot
    ax.scatter(rec, y, s=60, color=col, alpha=0.4, zorder=4,
               marker="D", edgecolors="white", linewidths=0.8)
    ax.text(rec + 0.015, y - 0.3, f"R={rec:.2f}", va="center",
            fontsize=7.5, color=col, alpha=0.6)

ax.set_yticks(list(y_positions))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0, 1.1)
ax.set_xlabel("Score", fontsize=12)
ax.set_title("F1 Score Progression Across Stages\n(● = F1,  ◆ = Recall)",
             fontsize=13, fontweight="bold")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
           label="F1 = 0.5 reference")
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.legend(fontsize=9)

stage_patches = [mpatches.Patch(color=c, label=s, alpha=0.9)
                 for s, c in STAGE_COLORS.items()]
fig.legend(handles=stage_patches, loc="lower right", fontsize=10,
           title="Stage", bbox_to_anchor=(0.98, 0.02))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "report_f1_progression.png"), dpi=180, bbox_inches="tight")
plt.close()
print("✓ report_f1_progression.png") # CHANGE

# =============================================================================
# FIGURE 5 — VERSION HISTORY
# =============================================================================
# Plots how model performance changed as you changed configurations.
# YOU NEED TO FILL IN results/version_history.csv with your actual numbers.
# A template is created automatically if it doesn't exist.
#
# CSV columns:
#   version  — integer, increment by 1 each time you try a new config
#   label    — short name for this config (e.g. "RF_thresh0.3_SMOTE")
#   model    — which model: LinReg / LogReg / RF / XGB / LSTM
#   f1       — F1 score for lick class
#   recall   — Recall for lick class
#   precision— Precision for lick class
#   notes    — free text describing what changed (shown as annotations)

HISTORY_PATH = os.path.join(RESULTS_DIR, "version_history.csv")

# Create template if it doesn't exist
if not os.path.exists(HISTORY_PATH):
    template = pd.DataFrame([
        # ── fill these in with your actual numbers ──────────────────────────
        # Delete or edit these rows; add one row per configuration you tested.
        # Keep version as sequential integers so the x-axis is ordered.
        {"version": 1,  "label": "LinReg\n1file",      "model": "LinReg",
         "f1": 0.00,  "recall": 0.00,  "precision": 0.00,
         "notes": "Linear Regression, 1 file, thresh=0.5"},
        {"version": 2,  "label": "LogReg\n1file",      "model": "LogReg",
         "f1": 0.54,  "recall": 0.66,  "precision": 0.46,
         "notes": "Logistic Regression, 1 file, balanced"},
        {"version": 3,  "label": "RF\n1file",          "model": "RF",
         "f1": 0.44,  "recall": 0.38,  "precision": 0.52,
         "notes": "Random Forest, 1 file, thresh=0.5"},
        {"version": 4,  "label": "XGB\n1file",         "model": "XGB",
         "f1": 0.33,  "recall": 0.24,  "precision": 0.52,
         "notes": "XGBoost, 1 file, thresh=0.5"},
        {"version": 5,  "label": "RF\nthresh=0.3",     "model": "RF",
         "f1": 0.61,  "recall": 0.76,  "precision": 0.51,
         "notes": "RF thresh lowered to 0.3"},
        {"version": 6,  "label": "RF+SMOTE\n3files",   "model": "RF",
         "f1": 0.49,  "recall": 0.69,  "precision": 0.37,
         "notes": "RF + SMOTE + 3 files"},
        {"version": 7,  "label": "LSTM\nraw",          "model": "LSTM",
         "f1": 0.16,  "recall": 0.11,  "precision": 0.31,
         "notes": "LSTM Stage 3, raw predictions"},
        # ── add your post-processed LSTM numbers here once you have them ────
        {"version": 8,  "label": "LSTM\npost-proc",    "model": "LSTM",
         "f1": 0.00,  "recall": 0.00,  "precision": 0.00,
         "notes": "FILL IN: LSTM post-processed"},
    ])
    template.to_csv(HISTORY_PATH, index=False)
    print(f"\n  ⚠ Created template: {HISTORY_PATH}")
    print("  Fill in your actual numbers, then re-run this script.\n")

hist = pd.read_csv(HISTORY_PATH)
hist = hist.sort_values("version").reset_index(drop=True)

model_colors = {
    "LinReg": "#4C72B0", "LogReg": "#6495ED",
    "RF": "#55A868", "XGB": "#C44E52", "LSTM": "#9C27B0",
}

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle("Configuration Version History — Metric Trends",
             fontsize=14, fontweight="bold", y=0.98)

for ax, metric, ylabel in zip(
    axes,
    ["f1", "recall", "precision"],
    ["F1 Score", "Recall", "Precision"]
):
    for model_name, grp in hist.groupby("model"):
        color = model_colors.get(model_name, "#888888")
        ax.plot(grp["version"], grp[metric],
                marker="o", color=color, linewidth=2,
                markersize=7, label=model_name)
        # Annotate each point with its label
        for _, row in grp.iterrows():
            ax.annotate(row["label"],
                        xy=(row["version"], row[metric]),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=7, color=color, alpha=0.8)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(title="Model", fontsize=9, title_fontsize=9,
               loc="upper left", ncol=5)

# X-axis: version numbers with notes as tick labels
axes[-1].set_xticks(hist["version"])
axes[-1].set_xticklabels(
    [f"v{row['version']}" for _, row in hist.iterrows()],
    fontsize=8
)
axes[-1].set_xlabel("Configuration Version", fontsize=11)

# Notes as a text box below the chart
notes_text = "\n".join(
    [f"v{row['version']}: {row['notes']}" for _, row in hist.iterrows()]
)
fig.text(0.01, -0.01, notes_text, fontsize=7, color="#555555",
         va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0F0", alpha=0.8))

plt.tight_layout(rect=[0, 0.0, 1, 0.97])
plt.savefig(os.path.join(RESULTS_DIR, "report_version_history.png"), # CHANGE
            dpi=180, bbox_inches="tight")
plt.close()
print("✓ report_version_history.png") # CHANGE

# =============================================================================
print(f"\nAll report figures saved to {RESULTS_DIR}/")
print("  report_radar.png")
print("  report_heatmap.png")
print("  report_pr_scatter.png")
print("  report_f1_progression.png")
print("  report_version_history.png")