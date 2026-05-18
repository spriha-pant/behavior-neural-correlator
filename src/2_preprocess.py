# =============================================================================
# src/2_preprocess.py
# =============================================================================
#
# FLAGS — change these at the top as needed:
#
#   DATA_ALREADY_ZSCORED
#     True  → skip StandardScaler (your CSVs are already Z-scored)
#     False → apply StandardScaler (raw dF/F values)
#
#   USE_CV
#     True  → TimeSeriesSplit cross-validation (recommended)
#     False → simple 80/20 end-cut
#
#   USE_SMOTE
#     True  → apply SMOTE to training set to balance lick/no-lick
#              requires: pip install imbalanced-learn
#     False → no resampling (class imbalance handled by model weights only)
#
# MULTI-FILE SUPPORT:
#   All CSVs in data/raw/ are loaded and processed.
#   Lag and rolling features are computed PER FILE before concatenating.
#   This prevents lag features from bleeding across recording sessions.
# =============================================================================

DATA_ALREADY_ZSCORED = True   # your files are already Z-scored
USE_CV               = True
N_CV_FOLDS           = 10
USE_SMOTE            = True   # pip install imbalanced-learn if not installed

# =============================================================================

import os
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

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
# STEP 1: Load and process each CSV file separately
# --------------------------------------------------------------------------
# WHY SEPARATELY:
#   Lag and rolling features look backwards in time.
#   Row 0 of file 2 is a different session from the last row of file 1.
#   If we concatenated first, the lag features at session boundaries
#   would mix data from two unrelated recordings. Processing per-file
#   ensures the first few rows of each session produce NaN (dropped),
#   and nothing crosses session boundaries.

csv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")])
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

print(f"Found {len(csv_files)} file(s): {csv_files}\n")

processed_chunks = []   # Will hold a processed DataFrame per file

for fname in csv_files:
    fpath = os.path.join(RAW_DIR, fname)
    df = pd.read_csv(fpath)
    print(f"--- {fname} ---")
    print(f"  Raw shape: {df.shape}")

    # ── Binary label ──────────────────────────────────────────────────────
    df["label"] = (df[BEHAVIOR_COL] == LICK_LABEL).astype(int)
    lick_pct = df["label"].mean() * 100
    print(f"  Lick rate: {lick_pct:.1f}%  ({df['label'].sum()} lick rows)")

    # ── Normalisation ─────────────────────────────────────────────────────
    if DATA_ALREADY_ZSCORED:
        # Data is already Z-scored. Use it directly.
        # We still save a "pass-through" scaler (fitted on this data) so
        # that 5_predict.py doesn't break when it loads the scaler file.
        df["dFF_scaled"] = df[NEURAL_COL].values
    else:
        scaler = StandardScaler()
        df["dFF_scaled"] = scaler.fit_transform(df[[NEURAL_COL]])

    # ── Lag features ──────────────────────────────────────────────────────
    # Each lag column gives the model dF/F from N timepoints ago.
    # .shift(N) pushes values down by N rows → first N rows become NaN.
    for lag in LAG_STEPS:
        df[f"dFF_lag_{lag}"] = df["dFF_scaled"].shift(lag)

    # ── Delta feature (first derivative) ─────────────────────────────────
    # dF/F[t] − dF/F[t-1]: how fast the signal is rising or falling.
    # A rising signal before lick onset is a common neural pattern.
    df["dFF_delta"] = df["dFF_scaled"].diff(1)

    # ── Rolling statistics ─────────────────────────────────────────────────
    df["dFF_roll_mean"] = df["dFF_scaled"].rolling(window=ROLL_WIN).mean()
    df["dFF_roll_std"]  = df["dFF_scaled"].rolling(window=ROLL_WIN).std()

    # ── Drop NaN rows created by lag/rolling/diff ──────────────────────────
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  After dropping NaN: {len(df)} rows  (dropped {rows_before - len(df)})")

    # ── Tag which file each row came from (useful for debugging) ───────────
    df["source_file"] = fname

    processed_chunks.append(df)
    print()

# --------------------------------------------------------------------------
# STEP 2: Concatenate all processed files
# --------------------------------------------------------------------------
# Now that lag features are cleanly computed per session, we can stack them.

full_df = pd.concat(processed_chunks, ignore_index=True)
print(f"Combined dataset: {full_df.shape[0]} rows from {len(csv_files)} files")
print(f"Overall lick rate: {full_df['label'].mean()*100:.1f}%")

lick_rows = full_df.index[full_df["label"] == 1]
print(f"Lick events span rows {lick_rows.min()}–{lick_rows.max()} "
      f"({lick_rows.min()/len(full_df)*100:.1f}%–{lick_rows.max()/len(full_df)*100:.1f}% of combined data)")

# --------------------------------------------------------------------------
# STEP 3: Save scaler
# --------------------------------------------------------------------------
# If data is already Z-scored we fit the scaler on the scaled values
# (effectively identity), just so 5_predict.py doesn't break.

scaler = StandardScaler()
scaler.fit(full_df[["dFF_scaled"]])
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print(f"\nScaler saved (DATA_ALREADY_ZSCORED={DATA_ALREADY_ZSCORED}).")

# --------------------------------------------------------------------------
# STEP 4: Build feature matrix X and target y
# --------------------------------------------------------------------------

lag_feature_cols = [f"dFF_lag_{lag}" for lag in LAG_STEPS]
feature_cols = (
    ["dFF_scaled"]
    + lag_feature_cols
    + ["dFF_delta", "dFF_roll_mean", "dFF_roll_std"]
)

X = full_df[feature_cols]
y = full_df["label"]

print(f"\nFeature matrix X: {X.shape}")
print(f"Features: {feature_cols}")

pd.Series(feature_cols).to_csv(
    os.path.join(PROCESSED_DIR, "feature_names.csv"), index=False, header=["feature"]
)

# --------------------------------------------------------------------------
# STEP 5: Train/test split
# --------------------------------------------------------------------------

if not USE_CV:
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"\nSimple split — train: {X_train.shape}, test: {X_test.shape}")
    print(f"  Train lick rate: {y_train.mean()*100:.1f}%  |  Test lick rate: {y_test.mean()*100:.1f}%")

else:
    print(f"\nTimeSeriesSplit with {N_CV_FOLDS} folds:")
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    fold_info = []
    best_fold_idx  = None
    best_test_lick = -1
    fold_splits    = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        y_fold_train = y.iloc[train_idx]
        y_fold_test  = y.iloc[test_idx]
        lt = y_fold_train.mean() * 100
        lk = y_fold_test.mean()  * 100

        print(f"  Fold {fold:2d}: "
              f"train rows {train_idx[0]:5d}–{train_idx[-1]:5d} (lick: {lt:5.1f}%)  |  "
              f"test rows {test_idx[0]:5d}–{test_idx[-1]:5d} (lick: {lk:5.1f}%)")

        fold_info.append({
            "fold": fold,
            "train_start": train_idx[0], "train_end": train_idx[-1],
            "test_start":  test_idx[0],  "test_end":  test_idx[-1],
            "train_lick_pct": round(lt, 2),
            "test_lick_pct":  round(lk, 2)
        })
        fold_splits.append((train_idx, test_idx))

        if lk > best_test_lick:
            best_test_lick = lk
            best_fold_idx  = fold - 1   # 0-indexed into fold_splits

    pd.DataFrame(fold_info).to_csv(
        os.path.join(PROCESSED_DIR, "cv_fold_summary.csv"), index=False
    )

    best_fold_num = fold_info[best_fold_idx]["fold"]
    print(f"\n  Auto-selected fold {best_fold_num} "
          f"(highest test lick rate: {best_test_lick:.1f}%)")

    train_idx, test_idx = fold_splits[best_fold_idx]
    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    print(f"  X_train: {X_train.shape}  lick rate: {y_train.mean()*100:.1f}%")
    print(f"  X_test:  {X_test.shape}  lick rate: {y_test.mean()*100:.1f}%")

# --------------------------------------------------------------------------
# STEP 6: SMOTE — oversample minority class in TRAINING set only
# --------------------------------------------------------------------------
# WHAT SMOTE DOES:
#   For each lick (minority) sample, it finds its K nearest neighbours
#   among other lick samples in feature space, then creates new synthetic
#   lick samples along the line segments between them.
#   Result: training set ends up with equal numbers of lick and no-lick.
#
# WHY TRAINING ONLY:
#   Test set must reflect real-world class distribution.
#   Evaluating on synthetic data gives false confidence.
#
# TIMING CAVEAT:
#   SMOTE doesn't know about time. The synthetic samples are feature
#   vectors without temporal position. This is fine for RF/XGBoost
#   (which treat each row independently), but for LSTM we'll revisit.

if USE_SMOTE:
    try:
        from imblearn.over_sampling import SMOTE

        print(f"\nApplying SMOTE to training set...")
        print(f"  Before — lick: {y_train.sum()}, no-lick: {(y_train==0).sum()}")

        smote = SMOTE(random_state=config["random_seed"])
        X_train_arr, y_train_arr = smote.fit_resample(X_train, y_train)

        # Convert back to DataFrame with correct column names
        X_train = pd.DataFrame(X_train_arr, columns=feature_cols)
        y_train = pd.Series(y_train_arr, name="label")

        print(f"  After  — lick: {y_train.sum()}, no-lick: {(y_train==0).sum()}")
        print(f"  New X_train shape: {X_train.shape}")

    except ImportError:
        print("\n  ⚠ imbalanced-learn not installed. Skipping SMOTE.")
        print("    Install with: pip install imbalanced-learn")
        print("    Then re-run this script.")

# --------------------------------------------------------------------------
# STEP 7: Save
# --------------------------------------------------------------------------

X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR,  "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR,  "y_test.csv"),  index=False)

print(f"\n✓ All processed data saved to: {PROCESSED_DIR}/")
print(f"  Proceed to: python src/3_train_model.py")
print(f"              python src/3b_train_stage2.py")