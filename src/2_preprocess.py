# =============================================================================
# src/2_preprocess.py
# =============================================================================

# ▼▼▼ CHANGE THIS IF YOUR TEST SET HAS 0% LICK RATE ▼▼▼
USE_CV = True   # True = TimeSeriesSplit (recommended if licks cluster in time)
                # False = simple 80/20 end-cut
N_CV_FOLDS = 5  # Only used when USE_CV=True
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

import os
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

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

csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"No CSV files in {RAW_DIR}.")

raw_path = os.path.join(RAW_DIR, csv_files[0])
print(f"Loading raw data from: {raw_path}")
df = pd.read_csv(raw_path)
print(f"  Raw shape: {df.shape}")

# --------------------------------------------------------------------------
# STEP 2: Create binary label column
# --------------------------------------------------------------------------

df["label"] = (df[BEHAVIOR_COL] == LICK_LABEL).astype(int)
print(f"\nBinary label created:")
print(f"  Lick (1): {df['label'].sum()} rows  ({df['label'].mean()*100:.1f}%)")
print(f"  No-lick (0): {(df['label']==0).sum()} rows  ({(df['label']==0).mean()*100:.1f}%)")

# Report where in time lick events occur
lick_rows = df.index[df["label"] == 1]
if len(lick_rows) > 0:
    lick_start_pct = (lick_rows.min() / len(df)) * 100
    lick_end_pct   = (lick_rows.max() / len(df)) * 100
    print(f"\n  Lick events span rows {lick_rows.min()}–{lick_rows.max()} "
          f"({lick_start_pct:.1f}%–{lick_end_pct:.1f}% of recording)")
    if lick_start_pct > 70:
        print(f"  ⚠ Lick events start very late ({lick_start_pct:.0f}% in).")
        print(f"    Simple 80/20 split will miss them. USE_CV=True is recommended.")

# --------------------------------------------------------------------------
# STEP 3: Normalize dF/F
# --------------------------------------------------------------------------

scaler = StandardScaler()
df["dFF_scaled"] = scaler.fit_transform(df[[NEURAL_COL]])
print(f"\nNormalization: mean={df['dFF_scaled'].mean():.4f}, std={df['dFF_scaled'].std():.4f}")
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print(f"  Scaler saved.")

# --------------------------------------------------------------------------
# STEP 4: Lag features
# --------------------------------------------------------------------------

print(f"\nAdding lag features: {LAG_STEPS}")
for lag in LAG_STEPS:
    df[f"dFF_lag_{lag}"] = df["dFF_scaled"].shift(lag)
lag_feature_cols = [f"dFF_lag_{lag}" for lag in LAG_STEPS]

# --------------------------------------------------------------------------
# STEP 5: Rolling statistics
# --------------------------------------------------------------------------

df["dFF_roll_mean"] = df["dFF_scaled"].rolling(window=ROLL_WIN).mean()
df["dFF_roll_std"]  = df["dFF_scaled"].rolling(window=ROLL_WIN).std()
print(f"Adding rolling features (window={ROLL_WIN}).")

# --------------------------------------------------------------------------
# STEP 6: Drop NaN rows
# --------------------------------------------------------------------------

rows_before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"\nDropped {rows_before - len(df)} NaN rows. Remaining: {len(df)}")

# --------------------------------------------------------------------------
# STEP 7: Feature matrix and target
# --------------------------------------------------------------------------

feature_cols = ["dFF_scaled"] + lag_feature_cols + ["dFF_roll_mean", "dFF_roll_std"]
X = df[feature_cols]
y = df["label"]
print(f"\nFeature matrix X: {X.shape}   |   Features: {feature_cols}")

pd.Series(feature_cols).to_csv(
    os.path.join(PROCESSED_DIR, "feature_names.csv"), index=False, header=["feature"]
)

# --------------------------------------------------------------------------
# STEP 8: Split strategy
# --------------------------------------------------------------------------

if not USE_CV:
    # Simple 80/20 end-cut — only use this when lick events are spread throughout
    print(f"\nSplit strategy: Simple {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} end-cut")
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  Train lick rate: {y_train.mean()*100:.1f}%  |  Test lick rate: {y_test.mean()*100:.1f}%")
    if y_test.mean() == 0:
        print(f"  ⚠ Test set has 0% lick rate! Switch to USE_CV=True.")

else:
    # TimeSeriesSplit — maintains time order, but evaluates across multiple windows
    # so lick events are guaranteed to appear in at least some test folds.
    #
    # How it works with 5 folds on your data (simplified):
    #   Fold 1: train [0 → 20%],   test [20% → 40%]
    #   Fold 2: train [0 → 40%],   test [40% → 60%]
    #   Fold 3: train [0 → 60%],   test [60% → 80%]    ← lick events likely here
    #   Fold 4: train [0 → 80%],   test [80% → 100%]
    #   Fold 5: (same as 4 for 5 splits — sklearn distributes evenly)
    #
    # We save the last fold's train/test for 3_train_model.py.
    # We also save cv_fold_summary.csv so you can see which folds had lick events.

    print(f"\nSplit strategy: TimeSeriesSplit ({N_CV_FOLDS} folds)")
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)

    fold_info = []
    last_train_idx, last_test_idx = None, None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        y_fold_train = y.iloc[train_idx]
        y_fold_test  = y.iloc[test_idx]
        lick_train   = y_fold_train.mean() * 100
        lick_test    = y_fold_test.mean() * 100
        print(f"  Fold {fold}: "
              f"train rows {train_idx[0]}-{train_idx[-1]} (lick: {lick_train:.1f}%)  |  "
              f"test rows {test_idx[0]}-{test_idx[-1]} (lick: {lick_test:.1f}%)")
        fold_info.append({
            "fold": fold,
            "train_start": train_idx[0], "train_end": train_idx[-1],
            "test_start": test_idx[0],   "test_end": test_idx[-1],
            "train_lick_pct": round(lick_train, 2),
            "test_lick_pct":  round(lick_test, 2)
        })
        last_train_idx = train_idx
        last_test_idx  = test_idx

    pd.DataFrame(fold_info).to_csv(
        os.path.join(PROCESSED_DIR, "cv_fold_summary.csv"), index=False
    )
    print(f"\n  CV fold summary saved → check data/processed/cv_fold_summary.csv")
    print(f"  Look for folds where test_lick_pct > 0.")

    X_train = X.iloc[last_train_idx]
    X_test  = X.iloc[last_test_idx]
    y_train = y.iloc[last_train_idx]
    y_test  = y.iloc[last_test_idx]

    print(f"\n  Using fold {N_CV_FOLDS} for model files:")
    print(f"  X_train: {X_train.shape}  lick rate: {y_train.mean()*100:.1f}%")
    print(f"  X_test:  {X_test.shape}  lick rate: {y_test.mean()*100:.1f}%")

    if y_test.mean() == 0:
        print(f"\n  ⚠ Even the last fold test set has 0% lick rate.")
        print(f"  ⚠ Open cv_fold_summary.csv and find a fold where test_lick_pct > 0.")
        print(f"  ⚠ Then set N_CV_FOLDS to that fold number and re-run.")

X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR,  "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR,  "y_test.csv"),  index=False)

print(f"\n✓ All processed data saved to: {PROCESSED_DIR}/")
print(f"  Proceed to: python src/3_train_model.py")