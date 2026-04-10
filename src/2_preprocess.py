# =============================================================================
# src/2_preprocess.py
# PURPOSE: Clean the raw data, engineer features, split into train/test sets,
#          and save processed data to data/processed/.
#
# What this script does, in order:
#   1. Load raw CSV
#   2. Create binary label (lick=1, everything else=0)
#   3. Normalize the dF/F signal (z-score)
#   4. Add lag features (dF/F from previous timepoints)
#   5. Add rolling mean and std features
#   6. Drop rows with NaN (created by lag/rolling operations)
#   7. Split into train and test sets (no shuffling — time-series!)
#   8. Save X_train, X_test, y_train, y_test to data/processed/
# =============================================================================

import os
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler

# --------------------------------------------------------------------------
# STEP 0: Load config
# --------------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DIR       = config["paths"]["raw_data_dir"]
PROCESSED_DIR = config["paths"]["processed_data_dir"]
MODELS_DIR    = config["paths"]["models_dir"]

TIME_COL      = config["data"]["time_col"]
NEURAL_COL    = config["data"]["neural_col"]
BEHAVIOR_COL  = config["data"]["behavior_col"]

LICK_LABEL    = config["behavior"]["lick_label"]
LAG_STEPS     = config["features"]["lag_steps"]
ROLL_WIN      = config["features"]["rolling_window"]
TEST_SIZE     = config["split"]["test_size"]

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# STEP 1: Load raw data
# --------------------------------------------------------------------------
# For Phase 1, we work with a single CSV file.
# Find the first CSV in the raw directory.

csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"No CSV files in {RAW_DIR}. Run 1_load_data.py first.")

# Using the first file found. Later phases will loop over all files.
raw_path = os.path.join(RAW_DIR, csv_files[0])
print(f"Loading raw data from: {raw_path}")
df = pd.read_csv(raw_path)
print(f"  Raw shape: {df.shape}")


# --------------------------------------------------------------------------
# STEP 2: Create binary label column
# --------------------------------------------------------------------------
# lick → 1, everything else (neutral, groom, etc.) → 0
# This is Phase 1: we only care about lick vs. no-lick.

df["label"] = (df[BEHAVIOR_COL] == LICK_LABEL).astype(int)

print(f"\nBinary label created:")
print(f"  Lick (1): {df['label'].sum()} rows  ({df['label'].mean()*100:.1f}%)")
print(f"  No-lick (0): {(df['label']==0).sum()} rows  ({(df['label']==0).mean()*100:.1f}%)")


# --------------------------------------------------------------------------
# STEP 3: Normalize the dF/F signal (Z-score normalization)
# --------------------------------------------------------------------------
# dF/F values can vary a lot between sessions/animals.
# Z-score makes the signal: mean=0, std=1
# This helps the model treat all sessions equally.
#
# IMPORTANT: We fit the scaler ONLY on training data later.
# For now, we fit on all data (since we haven't split yet).
# We'll re-do a proper fit-on-train-only approach after splitting.
# The scaler is saved so we can apply the same transform to new data.

scaler = StandardScaler()
df["dFF_scaled"] = scaler.fit_transform(df[[NEURAL_COL]])

print(f"\nNormalization (Z-score):")
print(f"  Original dF/F — mean: {df[NEURAL_COL].mean():.4f}, std: {df[NEURAL_COL].std():.4f}")
print(f"  Scaled dF/F  — mean: {df['dFF_scaled'].mean():.4f}, std: {df['dFF_scaled'].std():.4f}")

# Save scaler for use during prediction on new data
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"  Scaler saved to: {scaler_path}")


# --------------------------------------------------------------------------
# STEP 4: Add lag features
# --------------------------------------------------------------------------
# The brain's current activity often reflects what happened slightly before.
# Lag features let the model "look back" in time.
# e.g., lag_1 = dFF_scaled value 1 timepoint ago
#        lag_5 = dFF_scaled value 5 timepoints ago (0.5 seconds if 10Hz)

print(f"\nAdding lag features: {LAG_STEPS}")
for lag in LAG_STEPS:
    col_name = f"dFF_lag_{lag}"
    df[col_name] = df["dFF_scaled"].shift(lag)
    # .shift(lag) moves all values DOWN by 'lag' rows,
    # so row N gets the value that was previously at row N-lag.
    # The first 'lag' rows become NaN — we'll drop those later.

lag_feature_cols = [f"dFF_lag_{lag}" for lag in LAG_STEPS]
print(f"  Lag columns added: {lag_feature_cols}")


# --------------------------------------------------------------------------
# STEP 5: Add rolling statistics features
# --------------------------------------------------------------------------
# Rolling mean = local average of dF/F over the last N timepoints
# Rolling std  = how much dF/F was fluctuating over the last N timepoints
# These capture "context" around each timepoint — not just one value.

print(f"\nAdding rolling features (window={ROLL_WIN} timepoints):")
df["dFF_roll_mean"] = df["dFF_scaled"].rolling(window=ROLL_WIN).mean()
df["dFF_roll_std"]  = df["dFF_scaled"].rolling(window=ROLL_WIN).std()
print(f"  Rolling mean and std added.")


# --------------------------------------------------------------------------
# STEP 6: Drop NaN rows
# --------------------------------------------------------------------------
# Lag and rolling operations create NaN at the beginning of the data.
# We drop those rows since we can't train on incomplete data.

rows_before = len(df)
df = df.dropna().reset_index(drop=True)
rows_after = len(df)
print(f"\nDropped {rows_before - rows_after} NaN rows (expected from lag/rolling).")
print(f"  Remaining rows: {rows_after}")


# --------------------------------------------------------------------------
# STEP 7: Define feature matrix (X) and target vector (y)
# --------------------------------------------------------------------------
# X = all input columns the model will learn from
# y = what we want the model to predict (lick / no-lick)

feature_cols = ["dFF_scaled"] + lag_feature_cols + ["dFF_roll_mean", "dFF_roll_std"]

X = df[feature_cols]
y = df["label"]

print(f"\nFeature matrix X: {X.shape}  (rows × features)")
print(f"Target vector  y: {y.shape}")
print(f"Features used: {feature_cols}")


# --------------------------------------------------------------------------
# STEP 8: Train/Test split (NO shuffle — time-series!)
# --------------------------------------------------------------------------
# For time-series data, we NEVER shuffle before splitting.
# Shuffling would let the model train on "future" data and test on "past",
# which gives falsely optimistic results and doesn't reflect real use.
# We simply cut the data: first 80% = train, last 20% = test.

split_idx = int(len(X) * (1 - TEST_SIZE))

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

print(f"\nTrain/Test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, no shuffle):")
print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}  |  y_test:  {y_test.shape}")

# Double-check label balance in each split
print(f"\n  Train lick rate: {y_train.mean()*100:.1f}%")
print(f"  Test  lick rate: {y_test.mean()*100:.1f}%")


# --------------------------------------------------------------------------
# STEP 9: Save processed data
# --------------------------------------------------------------------------

X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR,  "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR,  "y_test.csv"),  index=False)

# Also save the column names list for reference
pd.Series(feature_cols).to_csv(
    os.path.join(PROCESSED_DIR, "feature_names.csv"), index=False, header=["feature"]
)

print(f"\n✓ All processed data saved to: {PROCESSED_DIR}/")
print(f"  Files: X_train.csv, X_test.csv, y_train.csv, y_test.csv, feature_names.csv")
print(f"\n  Proceed to: python src/3_train_model.py")
