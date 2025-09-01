# src/train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import create_dataloader   # your dataloader (no normalization)
from model import LSTM                     # your LSTM class
import time
import os

# ------------------- Hyperparameters -------------------
file_path_train = '../data/DataNew/datadeltav2.h5'           # training dataset (12k)
file_path_val   = '../data/DataNew/ngrid32_delta_20250116-193307.h5'  # independent validation dataset 

batch_size = 16
hidden_size = 128
num_layers = 2
num_epochs = 2
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------- Load data -------------------
train_loader = create_dataloader(file_path_train, batch_size=batch_size, verbose=False)
val_loader   = create_dataloader(file_path_val, batch_size=batch_size, verbose=False)

train_dataset = train_loader.dataset
seq_len = train_dataset.get_seq_len()
num_grid = train_dataset.get_num_grid_points()
num_samples = len(train_dataset)

print(f"💪 Training LSTM with seq_len={seq_len}, num_grid={num_grid}")
print(f"   Number of training samples: {num_samples}, Batch size: {batch_size}")
print(f"   Device: {device}")

# ------------------- Model -------------------
input_size = num_grid * 3     # ψ_real, ψ_imag, potential
num_output = num_grid * 2     # predict next ψ_real, ψ_imag

model = LSTM(num_output=num_output,
             input_size=input_size,
             hidden_size=hidden_size,
             num_layers=num_layers).to(device)

# ------------------- Loss & Optimizer -------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------- Training Loop -------------------
save_best_only = True   # ✅ Option A: save only best model
time_str = time.strftime("%y%m%d-%H%M")
os.makedirs("../models", exist_ok=True)

best_loss = float('inf')
best_model_path  = f'../models/lstm_best_ngrid{num_grid}_seq{seq_len}_{time_str}.pt'
final_model_path = f'../models/lstm_final_ngrid{num_grid}_seq{seq_len}_{time_str}.pt'

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)

    # --- Validation (every epoch) ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

    # --- Save model ---
    if save_best_only:
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ Best model updated and saved to {best_model_path}")
    else:
        torch.save(model.state_dict(), final_model_path)

print("✅ Training complete!")
if save_best_only:
    print(f"🏆 Best model saved with val_loss={best_loss:.6f} at {best_model_path}")
else:
    print(f"💾 Final model saved to {final_model_path}")


