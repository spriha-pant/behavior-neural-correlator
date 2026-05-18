# =============================================================================
# src/5_predict.py
# PURPOSE: Load a trained model and run it on brand new, unseen data.
#
# Use this when you have a new recording CSV and want to predict lick/no-lick
# without going through the full training pipeline again.
#
# The new CSV must have the same columns as your training data:
#   time | dF/F | behavior
# (The behavior column can be all "neutral" or even absent — it's not required
#  for prediction, only for evaluation of accuracy if you have ground truth.)
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import yaml
import joblib

# --------------------------------------------------------------------------
# STEP 0: Load config
# --------------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config["paths"]["processed_data_dir"]
MODELS_DIR    = config["paths"]["models_dir"]
RESULTS_DIR   = config["paths"]["results_dir"]

TIME_COL      = config["data"]["time_col"]
NEURAL_COL    = config["data"]["neural_col"]
BEHAVIOR_COL  = config["data"]["behavior_col"]
LICK_LABEL    = config["behavior"]["lick_label"]
LAG_STEPS     = config["features"]["lag_steps"]
ROLL_WIN      = config["features"]["rolling_window"]
LR_THRESHOLD  = config["model"]["lr_threshold"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# STEP 1: Accept input file path
# --------------------------------------------------------------------------
# Run this script as:
#   python src/5_predict.py path/to/new_recording.csv [model_choice]
#
# model_choice can be "linear" or "logistic" (default: logistic)

if len(sys.argv) < 2:
    print("Usage: python src/5_predict.py <path_to_new_csv> [linear|logistic]")
    print("Example: python src/5_predict.py data/raw/session2.csv logistic")
    sys.exit(1)

new_csv_path = sys.argv[1]
model_choice = sys.argv[2] if len(sys.argv) > 2 else "logistic"

if not os.path.exists(new_csv_path):
    raise FileNotFoundError(f"File not found: {new_csv_path}")

print(f"New data file: {new_csv_path}")
print(f"Model to use: {model_choice}")


# --------------------------------------------------------------------------
# STEP 2: Load new data
# --------------------------------------------------------------------------

df = pd.read_csv(new_csv_path)
print(f"Loaded {len(df)} rows.")

# Check required columns
if TIME_COL not in df.columns or NEURAL_COL not in df.columns:
    raise ValueError(
        f"New CSV must have at least '{TIME_COL}' and '{NEURAL_COL}' columns."
    )

has_labels = BEHAVIOR_COL in df.columns


# --------------------------------------------------------------------------
# STEP 3: Apply the SAME preprocessing as during training
# --------------------------------------------------------------------------
# CRITICAL: You must apply exactly the same feature engineering steps.
# If you change this, predictions will be garbage.

# --- Load the saved scaler ---
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# --- Normalize dF/F using SAME scaler fit during training ---
df["dFF_scaled"] = scaler.transform(df[[NEURAL_COL]])

# --- Lag features ---
for lag in LAG_STEPS:
    df[f"dFF_lag_{lag}"] = df["dFF_scaled"].shift(lag)

# --- Rolling statistics ---
df["dFF_roll_mean"] = df["dFF_scaled"].rolling(window=ROLL_WIN).mean()
df["dFF_roll_std"]  = df["dFF_scaled"].rolling(window=ROLL_WIN).std()

# --- Drop NaN rows created by lag/rolling ---
n_before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Dropped {n_before - len(df)} NaN rows. Predicting on {len(df)} timepoints.")

# --- Build feature matrix ---
feature_cols = pd.read_csv(
    os.path.join(PROCESSED_DIR, "feature_names.csv")
)["feature"].tolist()

X_new = df[feature_cols]


# --------------------------------------------------------------------------
# STEP 4: Load model and predict
# --------------------------------------------------------------------------

if model_choice == "linear":
    model = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
    raw_output = model.predict(X_new)
    predictions = (raw_output > LR_THRESHOLD).astype(int)
    confidence  = raw_output  # Raw regression value (not a true probability)
    print("Using Linear Regression (threshold = {LR_THRESHOLD})")

elif model_choice == "logistic":
    model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    predictions = model.predict(X_new)
    confidence  = model.predict_proba(X_new)[:, 1]  # P(lick)
    print("Using Logistic Regression")

else:
    raise ValueError(f"Unknown model choice: '{model_choice}'. Use 'linear' or 'logistic'.")


# --------------------------------------------------------------------------
# STEP 5: Assemble output DataFrame
# --------------------------------------------------------------------------

output_df = df[[TIME_COL, NEURAL_COL]].copy()
output_df["predicted_lick"] = predictions
output_df["confidence"]      = confidence.round(4)
output_df["predicted_label"] = output_df["predicted_lick"].map(
    {1: "lick", 0: "no-lick"}
)

# If ground-truth labels are available, add them for comparison
if has_labels:
    # Map original label to binary for comparison
    df["true_label"] = (df[BEHAVIOR_COL] == LICK_LABEL).astype(int)
    output_df["true_lick"] = df["true_label"].values

    # Quick accuracy report
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(output_df["true_lick"], output_df["predicted_lick"])
    f1  = f1_score(output_df["true_lick"], output_df["predicted_lick"], zero_division=0)
    print(f"\n  Ground truth available → Quick evaluation:")
    print(f"  Accuracy: {acc*100:.2f}%   F1: {f1:.4f}")


# --------------------------------------------------------------------------
# STEP 6: Save predictions
# --------------------------------------------------------------------------

base_name = os.path.splitext(os.path.basename(new_csv_path))[0]
out_path = os.path.join(RESULTS_DIR, f"predictions_{base_name}_{model_choice}.csv")
output_df.to_csv(out_path, index=False)

print(f"\n✓ Predictions saved to: {out_path}")
print(f"\nFirst 10 predictions:")
print(output_df.head(10).to_string(index=False))

# Summary
lick_pct = output_df["predicted_lick"].mean() * 100
print(f"\nPredicted lick rate: {lick_pct:.1f}% of timepoints")
