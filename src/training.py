# src/train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils import create_dataloader  # our dataloader without normalization
from model import LSTM                       # your LSTM class
from validation_functions import log_density_plot, log_density_heatmap # validation functions to plot real predicted trajectories
from propagation import propagate_trajectory, propagate_dataset  # function to propagate a single trajectory
from accuracy_function import compute_overlap  # function to compute overlap accuracy
import time
import matplotlib.pyplot as plt
import numpy as np

# ------------------- Hyperparameters -------------------
file_path_train = '../data/DataNew/datadeltav2.h5'
file_path_val = '../data/DataNew/ngrid32_delta_20250116-193307.h5'

batch_size = 16
hidden_size = 128
num_layers = 2
num_epochs = 4
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------- Load data -------------------
train_loader = create_dataloader(file_path_train, batch_size=batch_size, verbose=False)
train_dataset = train_loader.dataset
val_loader = create_dataloader(file_path_val, batch_size=batch_size, verbose=False)

seq_len = train_dataset.get_seq_len()
num_grid = train_dataset.get_num_grid_points()
num_samples = len(train_dataset)

num_samples_val = len(val_loader.dataset)

print(f"💪 Training LSTM with seq_len={seq_len}, num_grid={num_grid}")
print(f"   Number of samples: {num_samples}, Batch size: {batch_size}")
print(f"   Device: {device}")
print(f"   Number of validation samples: {num_samples_val}")

input_size = num_grid * 3
num_output = num_grid * 2

# ------------------- Model -------------------
model = LSTM(num_output=num_output, input_size=input_size, hidden_size=hidden_size,
             num_layers=num_layers).to(device)

# ------------------- Loss & Optimizer -------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------- TensorBoard -------------------
time_str = time.strftime("%y%m%d-%H%M")
writer = SummaryWriter(log_dir=f'../runs/lstm_{time_str}')

# ------------------- Best model saving -------------------
save_best_only = False
best_loss = float('inf')
best_model_path = f'../models/lstm_best_ngrid{num_grid}_seq{seq_len}_{time_str}.pt'



# ------------------- Training Loop -------------------
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

    # --- Propagate validation set using new propagate_dataset ---
    pred_trajs, true_trajs, delta_pred, delta_true = propagate_dataset(model, val_loader, device, seq_len_init=1)

    # Compute average validation loss
    val_loss = np.mean([criterion(pred, true).item() for pred, true in zip(delta_pred, delta_true)])
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.6f} | Validation Loss: {val_loss:.6f}")

    # --- Overlap accuracy ---
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
        plot_interval = 2  

        if (epoch + 1) % plot_interval == 0:
            traj_idx = 0  # pick the first trajectory
            log_density_plot(writer, pred_trajs[traj_idx], true_trajs[traj_idx],step=epoch, num_grid_points=num_grid)
            log_density_heatmap(writer, pred_trajs[traj_idx], true_trajs[traj_idx],step=epoch, num_grid_points=num_grid)

    # TensorBoard logging for losses
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)

    # --- Optional best model saving ---
    if save_best_only and val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✨ Best model updated and saved to {best_model_path}")

# ------------------- Save final model -------------------
print("✅ Training complete!")
if save_best_only:
    print(f"🏆 Best model saved with loss {best_loss:.6f}")
if not save_best_only:
    final_model_path = f'../models/lstm_final_ngrid{num_grid}_seq{seq_len}_{time_str}.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved to {final_model_path}")

writer.close()
