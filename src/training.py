# src/train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils import create_dataloader
from model import LSTM, LSTM_with_Attention
from validation_functions import log_density_plot, log_density_heatmap
from propagation import propagate_dataset
from accuracy_function import compute_overlap
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# ------------------- Hyperparameters -------------------
file_path = '../data/DataNew/datadeltav2.h5'

batch_size = 16
hidden_size = 128
num_layers = 2
num_epochs = 3
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_best_only = False

# ------------------- Load data -------------------
loaders = create_dataloader(file_path, batch_size=batch_size, shuffle=True, verbose=True, split=True, ratios=(0.95, 0.05, 0), seed=42)
train_loader = loaders["train"]
val_loader   = loaders["val"]
test_loader  = loaders["test"]

seq_len = train_loader.dataset.dataset.get_seq_len()
num_grid = train_loader.dataset.dataset.get_num_grid_points()
num_samples = len(train_loader.dataset)
num_samples_val = len(val_loader.dataset)

input_size = num_grid * 3
num_output = num_grid * 2

# ------------------- Model -------------------
model = LSTM(num_output=num_output, input_size=input_size, hidden_size=hidden_size,num_layers=num_layers).to(device)
# model = LSTM_with_Attention(num_output=num_output, input_size=input_size, hidden_size=hidden_size,num_layers=num_layers).to(device)

# ------------------- Print model summary -------------------
print("="*20+"🔧 Model Summary:"+"="*20)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")


# ------------------- Loss & Optimizer -------------------
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# ------------------- TensorBoard ------------------------
if not os.path.exists('../runs'):
    os.makedirs('../runs')
time_str = time.strftime("%y%m%d-%H%M")
writer = SummaryWriter(log_dir=f'../runs/{time_str}_train_samples_{num_samples}_layers{num_layers}_hs{hidden_size}_lr{learning_rate}_bs{batch_size}')
print(f"TensorBoard log directory: ../runs/{time_str}")


# Save hyperparameters
hparams = {
    "batch_size": batch_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "learning_rate": learning_rate,
    "seq_len": seq_len,
    "num_grid_points": num_grid,
    "num_samples_train": num_samples,
    "num_samples_val": num_samples_val,
    "model_class": model.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "save_best_only": save_best_only,
    "num_epochs": num_epochs
}
print("="*20+"Hyperparameters"+"="*20)
for k, v in hparams.items():
    print(f"{k}: {v}")

# ------------------- Best model path -------------------
best_loss = float('inf')
if not os.path.exists('../models'):
    os.makedirs('../models')
best_model_path = f'../models/lstm_best_ngrid{num_grid}_seq{seq_len}_{time_str}.pt'
final_model_path = f'../models/{time_str}_train_samples{num_samples}.pt'


# ------------------- Training Loop -------------------
print("="*20+"🚀 Starting training..."+"="*20)
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)

    # --- Validation propagation ---
    pred_trajs, true_trajs, delta_pred, delta_true = propagate_dataset(model, val_loader, device, seq_len_init=1)
    val_loss = np.mean([criterion(pred, true).item() for pred, true in zip(delta_pred, delta_true)])
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.6f} | Validation Loss: {val_loss:.6f}")

    # --- Fidelity / overlap ---
    if len(pred_trajs) > 0:
        overlaps, phases = [], []
        for pred, true in zip(pred_trajs, true_trajs):
            absS, theta = compute_overlap(true, pred, dx=1.0)
            overlaps.append(absS)
            phases.append(theta)
        avg_overlap = np.mean(overlaps)
        avg_phase = np.mean(phases)
        print(f"   Fidelity ⟨ψ_true|ψ_pred⟩: |S|={avg_overlap:.4f}, θ={avg_phase:.4f} rad")

        # TensorBoard logging
        writer.add_scalar('Accuracy/Absolute Magnitude', avg_overlap, epoch)
        writer.add_scalar('Accuracy/Phase', avg_phase, epoch)

        # --- Density plots every few epochs ---
        plot_interval = 1
        if (epoch + 1) % plot_interval == 0:
            traj_idx = random.randint(0, len(pred_trajs)-1)
            log_density_plot(writer, pred_trajs[traj_idx], true_trajs[traj_idx], step=epoch, num_grid_points=num_grid)
            log_density_heatmap(writer, pred_trajs[traj_idx], true_trajs[traj_idx], step=epoch, num_grid_points=num_grid)

    # TensorBoard logging for losses
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)

    # --- Save best model ---
    if save_best_only and val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✨ Best model updated and saved to {best_model_path}")

# ------------------- Save final model -------------------
print("✅ Training complete!")
if save_best_only:
    print(f"🏆 Best model saved with loss {best_loss:.6f}")
if not save_best_only:
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved to {final_model_path}")

writer.close()
