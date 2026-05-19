# =============================================================================
# src/2b_preprocess_lstm.py
# PURPOSE: Prepare data specifically for LSTM training.
#
# WHY THIS IS A SEPARATE FILE FROM 2_preprocess.py:
#   Stages 1 and 2 (linear models, RF, XGBoost) treat each timepoint as an
#   independent row. They receive a 2D matrix: (n_timepoints × n_features).
#
#   LSTM is fundamentally different. It receives SEQUENCES: for each
#   prediction, it sees a window of the last `seq_len` timepoints together,
#   in order. The input is 3D: (n_sequences × seq_len × n_features).
#
#   This file creates those sequences, splits them into train/test while
#   respecting session boundaries, and saves them as .npy arrays.
#
# SEQUENCE CONSTRUCTION (sliding window):
#   Given a session of N timepoints and seq_len=30:
#   - Sequence 0:  timepoints [0  … 29],  label = label[29]
#   - Sequence 1:  timepoints [1  … 30],  label = label[30]
#   - ...
#   - Sequence N-30: timepoints [N-30 … N-1], label = label[N-1]
#   Each sequence predicts the label of its LAST timepoint.
#   This is done PER FILE so no sequence crosses a session boundary.
#
# CLASS IMBALANCE:
#   No SMOTE here. SMOTE creates synthetic individual rows, but an LSTM
#   sequence is a coherent temporal window — you can't interpolate between
#   two lick-windows and get a valid synthetic lick-window.
#   Instead, class imbalance is handled by pos_weight in the loss function
#   during training (see 3c_train_stage3.py).
#
# Run from project/ root:
#   python src/2b_preprocess_lstm.py
# =============================================================================

DATA_ALREADY_ZSCORED = True
USE_CV               = True
N_CV_FOLDS           = 10

import os
import numpy as np
import pandas as pd
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
SEQ_LEN       = config["lstm"]["seq_len"]

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,    exist_ok=True)

# --------------------------------------------------------------------------
# HELPER: build sliding-window sequences from one session's data
# --------------------------------------------------------------------------

def make_sequences(X_arr: np.ndarray, y_arr: np.ndarray, seq_len: int):
    """
    Converts a 2D (timesteps × features) array into 3D sequences.

    Parameters
    ----------
    X_arr   : shape (T, F) — feature matrix for one session
    y_arr   : shape (T,)   — binary labels for one session
    seq_len : int          — number of timepoints per window

    Returns
    -------
    X_seqs : shape (T - seq_len + 1, seq_len, F)
    y_seqs : shape (T - seq_len + 1,)
        Label of the LAST timepoint in each window.
    """
    n_seqs = len(X_arr) - seq_len + 1
    if n_seqs <= 0:
        raise ValueError(
            f"Session has {len(X_arr)} timepoints but seq_len={seq_len}. "
            "Session is shorter than one sequence — reduce seq_len or use longer recordings."
        )
    # Stack windows using stride_tricks for memory efficiency
    # (equivalent to [X_arr[i:i+seq_len] for i in range(n_seqs)] but faster)
    shape   = (n_seqs, seq_len, X_arr.shape[1])
    strides = (X_arr.strides[0], X_arr.strides[0], X_arr.strides[1])
    X_seqs  = np.lib.stride_tricks.as_strided(X_arr, shape=shape, strides=strides).copy()
    y_seqs  = y_arr[seq_len - 1:]   # label of last timepoint in each window
    return X_seqs, y_seqs

# --------------------------------------------------------------------------
# STEP 1: Load and process each CSV file, build sequences per session
# --------------------------------------------------------------------------

csv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")])
if not csv_files:
    raise FileNotFoundError(f"No CSV files in {RAW_DIR}")

print(f"Found {len(csv_files)} file(s).\n")

lag_feature_cols = [f"dFF_lag_{lag}" for lag in LAG_STEPS]
feature_cols     = (["dFF_scaled"] + lag_feature_cols +
                    ["dFF_delta", "dFF_roll_mean", "dFF_roll_std"])

all_X_seqs = []
all_y_seqs = []

for fname in csv_files:
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
    print(f"--- {fname}  ({len(df)} rows) ---")

    # Binary label
    df["label"] = (df[BEHAVIOR_COL] == LICK_LABEL).astype(int)

    # Neural signal
    if DATA_ALREADY_ZSCORED:
        df["dFF_scaled"] = df[NEURAL_COL].values
    else:
        df["dFF_scaled"] = StandardScaler().fit_transform(df[[NEURAL_COL]])

    # Lag features (per-session only — no cross-session bleeding)
    for lag in LAG_STEPS:
        df[f"dFF_lag_{lag}"] = df["dFF_scaled"].shift(lag)

    # Delta
    df["dFF_delta"] = df["dFF_scaled"].diff(1)

    # Rolling stats
    df["dFF_roll_mean"] = df["dFF_scaled"].rolling(window=ROLL_WIN).mean()
    df["dFF_roll_std"]  = df["dFF_scaled"].rolling(window=ROLL_WIN).std()

    # Drop NaN rows from lag/rolling/diff operations
    df = df.dropna().reset_index(drop=True)
    print(f"  After NaN drop: {len(df)} rows | lick rate: {df['label'].mean()*100:.1f}%")

    # Build sequences
    X_arr = df[feature_cols].values.astype(np.float32)
    y_arr = df["label"].values.astype(np.float32)

    X_seqs, y_seqs = make_sequences(X_arr, y_arr, SEQ_LEN)
    print(f"  Sequences created: {X_seqs.shape}  "
          f"(lick rate: {y_seqs.mean()*100:.1f}%)")

    all_X_seqs.append(X_seqs)
    all_y_seqs.append(y_seqs)
    print()

# --------------------------------------------------------------------------
# STEP 2: Concatenate all sessions
# --------------------------------------------------------------------------

X_all = np.concatenate(all_X_seqs, axis=0)   # (total_seqs, seq_len, n_features)
y_all = np.concatenate(all_y_seqs, axis=0)   # (total_seqs,)

print(f"Combined sequences: X={X_all.shape}, y={y_all.shape}")
print(f"Overall lick rate: {y_all.mean()*100:.1f}%")

# Save feature names (same as Stage 1/2, for reference)
pd.Series(feature_cols).to_csv(
    os.path.join(PROCESSED_DIR, "feature_names_lstm.csv"),
    index=False, header=["feature"]
)

# --------------------------------------------------------------------------
# STEP 3: Save scaler (fitted on combined data)
# --------------------------------------------------------------------------

scaler = StandardScaler()
# Fit on the first feature column (dFF_scaled) across all sequences
scaler.fit(X_all[:, -1, 0:1])   # last timestep of each seq, first feature
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_lstm.pkl"))

# --------------------------------------------------------------------------
# STEP 4: Train / test split
# --------------------------------------------------------------------------
# TimeSeriesSplit on sequences (maintains temporal order).
# Auto-select fold with the highest test lick rate (same logic as Stage 1/2).

if not USE_CV:
    split_idx = int(len(X_all) * (1 - TEST_SIZE))
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    print(f"\nSimple split — train lick: {y_train.mean()*100:.1f}%  "
          f"test lick: {y_test.mean()*100:.1f}%")

else:
    print(f"\nTimeSeriesSplit with {N_CV_FOLDS} folds:")
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    fold_info      = []
    fold_splits    = []
    best_idx       = None
    best_lick      = -1

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_all), start=1):
        lt = y_all[tr_idx].mean() * 100
        lk = y_all[te_idx].mean() * 100
        print(f"  Fold {fold:2d}: train {tr_idx[0]:6d}–{tr_idx[-1]:6d} "
              f"(lick {lt:5.1f}%)  |  test {te_idx[0]:6d}–{te_idx[-1]:6d} "
              f"(lick {lk:5.1f}%)")
        fold_info.append({"fold": fold,
                          "train_lick_pct": round(lt, 2),
                          "test_lick_pct":  round(lk, 2)})
        fold_splits.append((tr_idx, te_idx))
        if lk > best_lick:
            best_lick = lk
            best_idx  = fold - 1

    pd.DataFrame(fold_info).to_csv(
        os.path.join(PROCESSED_DIR, "cv_fold_summary_lstm.csv"), index=False
    )

    best_fold = fold_info[best_idx]["fold"]
    print(f"\n  Auto-selected fold {best_fold} (test lick rate: {best_lick:.1f}%)")

    tr_idx, te_idx = fold_splits[best_idx]
    X_train, X_test = X_all[tr_idx], X_all[te_idx]
    y_train, y_test = y_all[tr_idx], y_all[te_idx]

    print(f"  X_train: {X_train.shape}  lick rate: {y_train.mean()*100:.1f}%")
    print(f"  X_test:  {X_test.shape}  lick rate: {y_test.mean()*100:.1f}%")

# --------------------------------------------------------------------------
# STEP 5: Save as .npy arrays
# --------------------------------------------------------------------------

np.save(os.path.join(PROCESSED_DIR, "lstm_X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "lstm_X_test.npy"),  X_test)
np.save(os.path.join(PROCESSED_DIR, "lstm_y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "lstm_y_test.npy"),  y_test)

print(f"\n✓ LSTM data saved to {PROCESSED_DIR}/")
print(f"  lstm_X_train.npy  {X_train.shape}")
print(f"  lstm_X_test.npy   {X_test.shape}")
print(f"  lstm_y_train.npy  {y_train.shape}")
print(f"  lstm_y_test.npy   {y_test.shape}")
print(f"\n  Proceed to: python src/3c_train_stage3.py")