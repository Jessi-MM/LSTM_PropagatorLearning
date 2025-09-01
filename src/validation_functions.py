import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 
def long_rollout(model, dataloader, device, seq_len_init=1):
    """
    Generate full trajectory predictions in an autoregressive manner.

    Returns:
        avg_val_loss: average validation loss
        all_pred_trajs: list of predicted trajectories (Tensor)
        all_true_trajs: list of true trajectories (Tensor)
    """
    model.eval()
    criterion = nn.MSELoss()
    all_pred_trajs = []
    all_true_trajs = []
    total_loss = 0.0
    num_samples = 0

    num_grid_points = dataloader.dataset.get_num_grid_points()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size, T, _ = X_batch.shape

            for i in range(batch_size):
                X_seq = X_batch[i]  # (T, 3*num_grid_points)
                y_seq = y_batch[i]  # (T, 2*num_grid_points)
                 # reconstruct psi from deltas
                psi_init = X_seq[:, :2*num_grid_points]     
                batch_true_tensor = psi_init + y_seq # (T, 2*num_grid_points) Wavefunction expected values

                prediction_len = T - seq_len_init 
                if prediction_len <= 0:
                    print(f"⚠️ Warning: seq_len >= trajectory length ({seq_len_init} >= {T}). Skipping.")
                    continue

                # initialize with first psi (and potentials)
                current_seq = X_seq[:seq_len_init].unsqueeze(0)  # (1, seq_len_init, 3*num_grid_points)
                traj_pred = []

                # potentials for future steps
                V_future = X_seq[seq_len_init:, 2*num_grid_points:]  # (prediction_len, num_grid_points)

                # hidden states
                h_0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)
                c_0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)

                # rollout
                for t in range(prediction_len + 1):
                    output, (h_0, c_0) = model(current_seq, (h_0, c_0))
                    delta = output[:, -1, :]  # last step prediction

                    # next psi
                    last_psi = current_seq[0, -1, :2*num_grid_points]
                    next_psi = last_psi + delta.squeeze(0)

                    # normalize
                    r_n = torch.linspace(-1.5, 1.5, num_grid_points, device=device) * (1 / 0.5291775)
                    density = next_psi[:num_grid_points]**2 + next_psi[num_grid_points:]**2
                    integral = torch.trapz(density, r_n)
                    if integral > 0:
                        next_psi = next_psi / integral.sqrt()

                    traj_pred.append(next_psi.cpu())

                    if t < prediction_len:
                        # build input
                        V_next = V_future[t]
                        new_input = torch.cat([next_psi, V_next], dim=-1).unsqueeze(0).unsqueeze(0)
                    

                        # slide window
                        current_seq = torch.cat([current_seq[:, 1:, :], new_input], dim=1)

                # ✅ match dimensions
                if len(traj_pred) == prediction_len +1:
                    traj_pred_tensor = torch.stack(traj_pred)             # (prediction_len, 2*num_grid_points)
                                # (T, 2*num_grid_points)

                    all_pred_trajs.append(traj_pred_tensor)
                    all_true_trajs.append(batch_true_tensor)
                    print(f"Predicted trajectory shape: {traj_pred_tensor.shape}")
                    print(f"True trajectory shape: {batch_true_tensor.shape}")
                    
                    # accumulate loss
                    total_loss += criterion(traj_pred_tensor, batch_true_tensor).item()
                    num_samples += 1

    avg_val_loss = total_loss / max(num_samples, 1)
    return avg_val_loss, all_pred_trajs, all_true_trajs


def log_density_plot(writer, pred_traj, true_traj, step, num_grid_points):
    """
    Log probability density comparison between prediction and ground truth.

    pred_traj: (T, 2*num_grid_points)
    true_traj: (T, 2*num_grid_points)
    """
    
    T = pred_traj.shape[0]
    x = torch.linspace(-1.5, 1.5, num_grid_points)

    # Reshape into real/imag and convert to Angstroms (sqrt of inverse Bohr radius)
    conversion = np.sqrt(1 / 0.5291775)
    pred_real = pred_traj[:, :num_grid_points] * conversion
    pred_imag = pred_traj[:, num_grid_points:] * conversion
    true_real = true_traj[:, :num_grid_points] * conversion
    true_imag = true_traj[:, num_grid_points:] * conversion

    # Compute densities
    pred_density = pred_real**2 + pred_imag**2
    true_density = true_real**2 + true_imag**2

    # Choose a few timesteps to visualize (start, middle, end)
    timesteps = [0, T//2, T-1]

    fig, axs = plt.subplots(1, len(timesteps), figsize=(15, 4))
    for i, t in enumerate(timesteps):
        axs[i].plot(x, true_density[t], label="True", color="black")
        axs[i].plot(x, pred_density[t], label="Pred", color="red", linestyle="--")
        axs[i].set_title(f"t={t+1}")
        axs[i].set_xlabel("Position [$\AA$]")
        axs[i].set_ylabel("$|\psi(r,t)|^2$   $[1/\AA]$")
        axs[i].legend()

    plt.tight_layout()
    writer.add_figure("Density Line Comparison", fig, global_step=step)
    plt.close(fig)
