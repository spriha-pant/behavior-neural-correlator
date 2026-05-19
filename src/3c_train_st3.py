# =============================================================================
# src/3c_train_stage3.py
# PURPOSE: Define and train the LSTM model (Stage 3).
#
# WHAT AN LSTM IS (plain language):
#   A standard neural network treats every input independently.
#   An LSTM has a "memory cell" — a hidden state that carries information
#   forward through the sequence. After seeing timepoint 1, it updates its
#   memory. After timepoint 2, it updates again based on both timepoint 2
#   and what it remembered from timepoint 1. By the end of a 30-timepoint
#   window, the memory encodes a summary of the entire trajectory.
#
#   This is why LSTM can learn "rising signal over 1.5 seconds → lick
#   likely" in a way that a per-row model never can.
#
# MODEL ARCHITECTURE:
#   Input  → LSTM layer 1 (hidden_size=64) → Dropout
#          → LSTM layer 2 (hidden_size=64) → Dropout
#          → Linear(64 → 1) → Sigmoid → probability of lick
#
# CLASS IMBALANCE:
#   Handled via pos_weight in BCEWithLogitsLoss.
#   pos_weight = n_no_lick / n_lick in the training set.
#   This tells the loss function: "a missed lick costs more than a false
#   alarm proportional to how rare lick events are."
#
# Run from project/ root:
#   python src/3c_train_stage3.py
# =============================================================================

import os
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROCESSED_DIR = config["paths"]["processed_data_dir"]
MODELS_DIR    = config["paths"]["models_dir"]
RESULTS_DIR   = config["paths"]["results_dir"]
SEED          = config["random_seed"]

SEQ_LEN     = config["lstm"]["seq_len"]
HIDDEN      = config["lstm"]["hidden_size"]
N_LAYERS    = config["lstm"]["num_layers"]
DROPOUT     = config["lstm"]["dropout"]
BATCH_SIZE  = config["lstm"]["batch_size"]
EPOCHS      = config["lstm"]["epochs"]
LR          = config["lstm"]["learning_rate"]
PATIENCE    = config["lstm"]["patience"]
VAL_SPLIT   = config["lstm"]["val_split"]

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Use GPU if available, otherwise CPU
# For calcium imaging datasets of this size, CPU is usually fast enough
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --------------------------------------------------------------------------
# STEP 1: Load preprocessed LSTM data
# --------------------------------------------------------------------------

print("\nLoading LSTM training data...")
X_train = np.load(os.path.join(PROCESSED_DIR, "lstm_X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "lstm_y_train.npy"))

print(f"  X_train: {X_train.shape}  (sequences × seq_len × features)")
print(f"  y_train: {y_train.shape}")
print(f"  Lick rate in training set: {y_train.mean()*100:.1f}%")

n_features = X_train.shape[2]
print(f"  Features per timepoint: {n_features}")

# --------------------------------------------------------------------------
# STEP 2: Build PyTorch datasets
# --------------------------------------------------------------------------

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

full_dataset = TensorDataset(X_tensor, y_tensor)

# Split training data into train / validation
# Validation is used only for early stopping — never for evaluation
n_val   = int(len(full_dataset) * VAL_SPLIT)
n_tr    = len(full_dataset) - n_val

train_ds, val_ds = random_split(
    full_dataset, [n_tr, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"\n  Train sequences: {n_tr}  |  Val sequences: {n_val}")

# --------------------------------------------------------------------------
# STEP 3: Define the LSTM model
# --------------------------------------------------------------------------

class LickLSTM(nn.Module):
    """
    Stacked LSTM for binary lick detection.

    Input shape:  (batch_size, seq_len, n_features)
    Output shape: (batch_size,)  — probability of lick at last timepoint
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout between LSTM layers (only applies when num_layers > 1)
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True    # input: (batch, seq, feature) not (seq, batch, feature)
        )

        self.dropout = nn.Dropout(dropout)

        # Final classifier: hidden_size → 1 logit
        # We output a raw logit (not sigmoid) because BCEWithLogitsLoss
        # applies sigmoid internally with better numerical stability
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)

        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        # We only care about the output at the LAST timepoint
        # because that's what we're predicting the label for

        last_step = lstm_out[:, -1, :]          # (batch, hidden_size)
        dropped   = self.dropout(last_step)
        logit     = self.fc(dropped).squeeze(1) # (batch,)
        return logit


model = LickLSTM(
    input_size=n_features,
    hidden_size=HIDDEN,
    num_layers=N_LAYERS,
    dropout=DROPOUT
).to(device)

print(f"\nModel architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {total_params:,}")

# --------------------------------------------------------------------------
# STEP 4: Loss function and optimizer
# --------------------------------------------------------------------------

# pos_weight handles class imbalance.
# Formula: n_no_lick / n_lick
# e.g., 85% no-lick, 15% lick → pos_weight ≈ 5.67
# This makes the model penalise missed licks more heavily.
n_lick    = int(y_train.sum())
n_nolick  = int(len(y_train) - n_lick)
pos_w     = torch.tensor([n_nolick / n_lick], dtype=torch.float32).to(device)
print(f"\n  pos_weight = {pos_w.item():.2f}  "
      f"(n_no_lick={n_nolick}, n_lick={n_lick})")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Learning rate scheduler: reduce LR by half if val_loss plateaus for 5 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# --------------------------------------------------------------------------
# STEP 5: Training loop
# --------------------------------------------------------------------------

print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val Acc':>9}  {'Note'}")
print("-" * 60)

train_losses = []
val_losses   = []
best_val_loss  = float("inf")
epochs_no_improve = 0
best_model_path   = os.path.join(MODELS_DIR, "lstm_best.pt")

for epoch in range(1, EPOCHS + 1):

    # ── Training phase ────────────────────────────────────────────────────
    model.train()
    epoch_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping: prevents exploding gradients, common in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_train_loss += loss.item() * len(X_batch)

    epoch_train_loss /= n_tr

    # ── Validation phase ──────────────────────────────────────────────────
    model.eval()
    epoch_val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits  = model(X_batch)
            loss    = criterion(logits, y_batch)
            epoch_val_loss += loss.item() * len(X_batch)
            preds   = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_batch).sum().item()

    epoch_val_loss /= n_val
    val_acc = correct / n_val

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)

    scheduler.step(epoch_val_loss)

    # ── Early stopping ────────────────────────────────────────────────────
    note = ""
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        note = "✓ saved"
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"{epoch:>6}  {epoch_train_loss:>11.4f}  "
                  f"{epoch_val_loss:>10.4f}  {val_acc:>9.3f}  EARLY STOP")
            break

    # Print every epoch
    print(f"{epoch:>6}  {epoch_train_loss:>11.4f}  "
          f"{epoch_val_loss:>10.4f}  {val_acc:>9.3f}  {note}")

# --------------------------------------------------------------------------
# STEP 6: Save training curves
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_losses, label="Train loss", color="#4C72B0", linewidth=1.5)
ax.plot(val_losses,   label="Val loss",   color="#DD8452", linewidth=1.5)
ax.axvline(x=len(train_losses) - PATIENCE - 1, color="gray",
           linestyle="--", linewidth=0.8, alpha=0.7, label="Early stop trigger")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCEWithLogitsLoss")
ax.set_title("LSTM Training & Validation Loss")
ax.legend()
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage3_training_curves.png"), dpi=150)
plt.close()

print(f"\n✓ Best model saved: {best_model_path}")
print(f"✓ Training curves saved: results/stage3_training_curves.png")
print(f"\n  Proceed to: python src/4c_evaluate_stage3.py")