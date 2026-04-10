# =============================================================================
# src/1_load_data.py
# PURPOSE: Load raw CSV data, run basic sanity checks, and report what's inside.
# This script never modifies the data — it only reads and validates.
# =============================================================================

import os
import pandas as pd
import yaml

# --------------------------------------------------------------------------
# STEP 0: Load config
# --------------------------------------------------------------------------
# All settings (paths, column names, etc.) come from config.yaml
# so we never hardcode anything here.

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DIR      = config["paths"]["raw_data_dir"]
TIME_COL     = config["data"]["time_col"]
NEURAL_COL   = config["data"]["neural_col"]
BEHAVIOR_COL = config["data"]["behavior_col"]


# --------------------------------------------------------------------------
# STEP 1: Find CSV files in the raw data directory
# --------------------------------------------------------------------------

csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in '{RAW_DIR}'. "
        "Please place your raw data CSV there and re-run."
    )

print(f"Found {len(csv_files)} CSV file(s) in '{RAW_DIR}':")
for f in csv_files:
    print(f"  - {f}")


# --------------------------------------------------------------------------
# STEP 2: Load and inspect each file
# --------------------------------------------------------------------------

def load_and_check(filepath: str) -> pd.DataFrame:
    """
    Loads a single CSV file and runs basic checks.
    Returns the DataFrame if everything looks okay.
    Prints warnings for anything suspicious.
    """
    print(f"\n{'='*60}")
    print(f"Loading: {filepath}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath)

    # --- Shape ---
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # --- Column names ---
    print(f"  Columns found: {list(df.columns)}")

    # --- Check expected columns exist ---
    expected_cols = [TIME_COL, NEURAL_COL, BEHAVIOR_COL]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected column(s): {missing}\n"
            f"Check your config.yaml column name settings."
        )
    print(f"  ✓ All expected columns present.")

    # --- Data types ---
    print(f"\n  Column dtypes:")
    for col in df.columns:
        print(f"    {col}: {df[col].dtype}")

    # --- Missing values ---
    nulls = df.isnull().sum()
    total_nulls = nulls.sum()
    if total_nulls > 0:
        print(f"\n  ⚠ WARNING: Missing values detected!")
        print(nulls[nulls > 0])
    else:
        print(f"\n  ✓ No missing values.")

    # --- Time column: check it's monotonically increasing ---
    if not df[TIME_COL].is_monotonic_increasing:
        print(f"  ⚠ WARNING: Time column is NOT monotonically increasing. "
              "This may cause issues. Check your data.")
    else:
        time_step = round(df[TIME_COL].diff().median(), 6)
        print(f"  ✓ Time is monotonically increasing. "
              f"Median timestep: {time_step}s  "
              f"(≈ {round(1/time_step)} Hz sampling rate)")

    # --- Neural signal summary ---
    print(f"\n  dF/F signal stats:")
    print(df[NEURAL_COL].describe().to_string())

    # --- Behavior label counts ---
    print(f"\n  Behavior label counts:")
    print(df[BEHAVIOR_COL].value_counts().to_string())

    # --- Check for "lick" label specifically ---
    lick_label = config["behavior"]["lick_label"]
    if lick_label not in df[BEHAVIOR_COL].values:
        print(f"\n  ⚠ WARNING: Lick label '{lick_label}' not found in behavior column. "
              f"Labels present: {df[BEHAVIOR_COL].unique()}\n"
              f"  → Update 'lick_label' in config.yaml to match exactly.")
    else:
        lick_pct = (df[BEHAVIOR_COL] == lick_label).mean() * 100
        print(f"\n  ✓ Lick label '{lick_label}' found. "
              f"Lick timepoints: {lick_pct:.1f}% of total data.")
        if lick_pct < 5:
            print(f"  ⚠ NOTE: Very few lick events (<5%). "
                  "Class imbalance may affect model training.")

    return df


# --------------------------------------------------------------------------
# STEP 3: Run on all found files
# --------------------------------------------------------------------------

all_dataframes = {}

for filename in csv_files:
    filepath = os.path.join(RAW_DIR, filename)
    df = load_and_check(filepath)
    all_dataframes[filename] = df

print(f"\n{'='*60}")
print(f"✓ Load check complete. {len(all_dataframes)} file(s) passed inspection.")
print(f"  Proceed to: python src/2_preprocess.py")
print(f"{'='*60}")
