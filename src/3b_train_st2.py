# =============================================================================
# src/3b_train_stage2.py
# PURPOSE: Train Stage 2 models — Random Forest and XGBoost — on the same
#          preprocessed data used for Stage 1.
#
# You do NOT need to re-run 2_preprocess.py. The same X_train/y_train files
# from data/processed/ are reused here directly.
#
# WHY THESE MODELS:
#   Random Forest builds many decision trees, each trained on a random
#   subset of data, and takes a majority vote. It handles nonlinear
#   relationships and class imbalance well.
#
#   XGBoost (Extreme Gradient Boosting) builds trees sequentially, where
#   each new tree tries to correct the errors of the previous one. It is
#   generally the strongest model in this category and is widely used in
#   neuroscience and bioinformatics pipelines.
#
# Run from the project/ root:
#   python src/3b_train_stage2.py
# =============================================================================

import os
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# STEP 0: Load config
# --------------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config["paths"]["processed_data_dir"]
MODELS_DIR    = config["paths"]["models_dir"]
SEED          = config["random_seed"]

os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# STEP 1: Load training data
# --------------------------------------------------------------------------

print("Loading processed training data...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()

print(f"  X_train shape: {X_train.shape}")
print(f"  Lick rate in training set: {y_train.mean()*100:.1f}%")

# --------------------------------------------------------------------------
# CLASS IMBALANCE: compute scale_pos_weight for XGBoost
# --------------------------------------------------------------------------
# Both models need to know that lick events are rarer than no-lick events.
#
# For Random Forest: class_weight="balanced" tells it to upweight the
#   minority class (lick) automatically.
#
# For XGBoost: scale_pos_weight does the same thing but explicitly.
#   The standard formula is: count(no-lick) / count(lick).
#   e.g., if 80% are no-lick and 20% are lick → scale_pos_weight = 4.
#   This tells XGBoost to treat each lick example as 4x more important.

n_nolick = (y_train == 0).sum()
n_lick   = (y_train == 1).sum()
scale_pos_weight = n_nolick / n_lick if n_lick > 0 else 1.0

print(f"\n  Class counts — no-lick: {n_nolick}, lick: {n_lick}")
print(f"  scale_pos_weight (for XGBoost): {scale_pos_weight:.2f}")

# --------------------------------------------------------------------------
# STEP 2A: Train Random Forest
# --------------------------------------------------------------------------
# Key hyperparameters explained:
#
#   n_estimators=300
#     Number of individual decision trees to build.
#     More trees = more stable predictions, but slower.
#     300 is a good default for datasets of this size (~4000 rows).
#
#   max_depth=10
#     Maximum depth of each tree. Unlimited depth leads to overfitting
#     (the tree memorises training data). 10 is a reasonable constraint.
#
#   min_samples_leaf=5
#     Each leaf node must have at least 5 samples. Prevents the tree
#     from splitting on noise or single-point anomalies.
#
#   class_weight="balanced"
#     Automatically adjusts weights inversely proportional to class freq.
#     Effectively compensates for the lick/no-lick imbalance.
#
#   n_jobs=-1
#     Use all available CPU cores to train trees in parallel. Faster.

print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("  Training complete.")

# Feature importance — which input features matter most?
feature_names = X_train.columns.tolist()
importances = rf_model.feature_importances_
print(f"\n  Feature importances (Random Forest):")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"    {name:<20} {imp:.4f}  {bar}")

rf_path = os.path.join(MODELS_DIR, "random_forest4.pkl") # CHANGE
joblib.dump(rf_model, rf_path)
print(f"\n  ✓ Random Forest saved to: {rf_path}")


# --------------------------------------------------------------------------
# STEP 2B: Train XGBoost
# --------------------------------------------------------------------------
# Key hyperparameters explained:
#
#   n_estimators=300
#     Number of boosting rounds (trees built sequentially).
#
#   max_depth=6
#     XGBoost trees are typically shallower than RF trees because
#     boosting already corrects errors across rounds. 6 is standard.
#
#   learning_rate=0.05
#     How much each new tree contributes to the final prediction.
#     Lower = more conservative, less overfitting, needs more trees.
#     0.05 with 300 trees is a well-tested combination.
#
#   subsample=0.8
#     Each tree is trained on 80% of training rows, randomly selected.
#     Adds randomness to prevent overfitting.
#
#   colsample_bytree=0.8
#     Each tree uses 80% of features, randomly selected.
#     Same idea — prevents any single feature from dominating.
#
#   scale_pos_weight
#     Handles class imbalance (computed above).
#
#   eval_metric="logloss"
#     Internal metric XGBoost monitors during training.
#     "logloss" (log loss) is standard for binary classification.
#
#   use_label_encoder=False
#     Suppresses a deprecation warning in newer XGBoost versions.

print("\n--- Training XGBoost ---")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=SEED,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
print("  Training complete.")

# Feature importance for XGBoost
print(f"\n  Feature importances (XGBoost):")
xgb_importances = xgb_model.feature_importances_
for name, imp in sorted(zip(feature_names, xgb_importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"    {name:<20} {imp:.4f}  {bar}")

xgb_path = os.path.join(MODELS_DIR, "xgboost4.pkl") # CHANGE
joblib.dump(xgb_model, xgb_path)
print(f"\n  ✓ XGBoost saved to: {xgb_path}")


# --------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------

print(f"\n{'='*50}")
print("Stage 2 training complete. Models saved:")
print(f"  {rf_path}")
print(f"  {xgb_path}")
print(f"\nProceed to: python src/4b_evaluate_stage2.py")
print(f"{'='*50}")