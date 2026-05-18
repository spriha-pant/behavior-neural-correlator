# =============================================================================
# src/3_train_model.py
# PURPOSE: Train two Stage 1 models on the preprocessed data and save them.
#
# Models trained here:
#   A) Linear Regression   — predicts a continuous value, threshold → binary
#   B) Logistic Regression — predicts probability of lick, threshold → binary
#
# Why both?
#   Linear Regression is your "Stage 1" baseline.
#   Logistic Regression is the "correct" linear model for binary classification.
#   Comparing them shows you where the linear assumption breaks down,
#   and sets the baseline for Stage 2 (Random Forest / XGBoost).
# =============================================================================

import os
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression

# --------------------------------------------------------------------------
# STEP 0: Load config
# --------------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config["paths"]["processed_data_dir"]
MODELS_DIR    = config["paths"]["models_dir"]
LR_THRESHOLD  = config["model"]["lr_threshold"]
SEED          = config["random_seed"]

os.makedirs(MODELS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# STEP 1: Load processed training data
# --------------------------------------------------------------------------

print("Loading processed training data...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
# .squeeze() converts a single-column DataFrame into a plain Series

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Lick rate in training set: {y_train.mean()*100:.1f}%")


# --------------------------------------------------------------------------
# STEP 2A: Train Linear Regression
# --------------------------------------------------------------------------
# Linear Regression finds the best-fit line through the data.
# It will output a continuous number for each row.
# We convert that to 0 or 1 using the threshold in config.yaml.
#
# This is a known mismatch (LR is not built for classification),
# but we're doing it deliberately to establish a baseline.

print("\n--- Training Linear Regression ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# After this line, lin_reg has learned its coefficients.

# Quick look at what the model learned
print(f"  Intercept: {lin_reg.intercept_:.4f}")
print(f"  Coefficients (one per feature):")
feature_names = X_train.columns.tolist()
for name, coef in zip(feature_names, lin_reg.coef_):
    print(f"    {name}: {coef:.6f}")

# Save model
lin_reg_path = os.path.join(MODELS_DIR, "linear_regression.pkl") # CHANGE
joblib.dump(lin_reg, lin_reg_path)
print(f"\n  ✓ Linear Regression saved to: {lin_reg_path}")


# --------------------------------------------------------------------------
# STEP 2B: Train Logistic Regression
# --------------------------------------------------------------------------
# Logistic Regression is the proper linear model for binary classification.
# Instead of a raw line, it passes the output through a sigmoid function
# so outputs are always between 0 and 1 (interpretable as probability).
# Prediction: if probability > 0.5 → lick, else → no-lick.
#
# max_iter=1000: Logistic Regression is solved iteratively.
#   Default is 100, which often fails to converge. 1000 is safer.
# class_weight='balanced': Automatically adjusts for class imbalance.
#   If lick events are rare, this prevents the model from just always
#   predicting "no-lick" and still getting high accuracy.

print("\n--- Training Logistic Regression ---")
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=SEED
)
log_reg.fit(X_train, y_train)

print(f"  Intercept: {log_reg.intercept_[0]:.4f}")
print(f"  Coefficients:")
for name, coef in zip(feature_names, log_reg.coef_[0]):
    print(f"    {name}: {coef:.6f}")

# Save model
log_reg_path = os.path.join(MODELS_DIR, "logistic_regression.pkl") # CHANGE
joblib.dump(log_reg, log_reg_path)
print(f"\n  ✓ Logistic Regression saved to: {log_reg_path}")


# --------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------

print(f"\n{'='*50}")
print("Training complete. Models saved:")
print(f"  {lin_reg_path}")
print(f"  {log_reg_path}")
print(f"\nProceed to: python src/4_evaluate.py")
print(f"{'='*50}")
