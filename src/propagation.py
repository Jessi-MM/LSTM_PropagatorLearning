# propagation.py
import torch
import torch.nn as nn


def propagate_trajectory(model, X_seq, y_seq, device, seq_len_init=1):
    """
    Propagate a single trajectory autoregressively.

    Args:
        model: trained LSTM model
        X_seq: input features (T, 3*num_grid_points)
        y_seq: deltas for psi (T, 2*num_grid_points)
        device: torch.device
        seq_len_init: number of initial steps given as context

    Returns:
        traj_pred_tensor: predicted ψ trajectory (T, 2*num_grid_points)
        batch_true_tensor: ground truth ψ trajectory (T, 2*num_grid_points)
    """
    # Number of grid points
    num_grid_points = X_seq.shape[-1] // 3
    T = X_seq.shape[0]

    # ✅ Ground truth ψ trajectory
    psi_init = X_seq[:, :2*num_grid_points]   # (T, 2*num_grid_points)
    batch_true_tensor = psi_init + y_seq     # ψ(t) = ψ_init + Δψ

    # Length of autoregressive prediction
    prediction_len = T - seq_len_init
    if prediction_len <= 0:
        raise ValueError(f"seq_len_init >= trajectory length ({seq_len_init} >= {T}).")

    # Initial context sequence
    current_seq = X_seq[:seq_len_init].unsqueeze(0).to(device)  # (1, seq_len_init, 3*num_grid_points)
    traj_pred = []

    # Future potentials (we already know them from dataset)
    V_future = X_seq[seq_len_init:, 2*num_grid_points:].to(device)  # (prediction_len, num_grid_points)

    # Hidden states init
    h_0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)
    c_0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)

    # Coordinate grid for normalization from Angström to atomic units a.u.
    r_n = torch.linspace(-1.5, 1.5, num_grid_points, device=device) * (1 / 0.5291775)

    # 🔁 Autoregressive loop
    with torch.no_grad():
        for t in range(prediction_len + 1):  # +1 because we include last context step
            # LSTM forward pass
            output, (h_0, c_0) = model(current_seq, (h_0, c_0))
            delta = output[:, -1, :]  # last time step Δψ

            # Get last ψ from input sequence
            last_psi = current_seq[0, -1, :2*num_grid_points]
            next_psi = last_psi + delta.squeeze(0)  # ψ_{t+1}

            # ✅ Normalize wavefunction
            density = next_psi[:num_grid_points]**2 + next_psi[num_grid_points:]**2
            integral = torch.trapz(density, r_n)
            if integral > 0:
                next_psi = next_psi / integral.sqrt()

            # Save prediction
            traj_pred.append(next_psi.cpu())

            # Update input sequence for next iteration
            if t < prediction_len:
                V_next = V_future[t]
                new_input = torch.cat([next_psi, V_next], dim=-1).unsqueeze(0).unsqueeze(0)
                current_seq = torch.cat([current_seq[:, 1:, :], new_input], dim=1)

    # Stack predictions into tensor
    traj_pred_tensor = torch.stack(traj_pred)  # (T, 2*num_grid_points)

    return traj_pred_tensor, batch_true_tensor

def propagate_dataset(model, dataloader, device, seq_len_init=1):
    """
    Propagate all trajectories from a dataloader using autoregressive prediction.

    Args:
        model: trained LSTM model
        dataloader: torch DataLoader providing (X_seq, y_seq)
        device: torch.device
        seq_len_init: number of initial steps given as context

    Returns:
        all_pred_trajs: list of predicted trajectories (each Tensor of shape T x 2*num_grid_points)
        all_true_trajs: list of ground truth trajectories (each Tensor of shape T x 2*num_grid_points)
    """
    model.eval()
    all_pred_trajs = []
    all_true_trajs = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            batch_size = X_batch.shape[0]
            for i in range(batch_size):
                X_seq = X_batch[i]
                y_seq = y_batch[i]

                traj_pred, batch_true = propagate_trajectory(model, X_seq, y_seq, device, seq_len_init)
                all_pred_trajs.append(traj_pred)
                all_true_trajs.append(batch_true)

    return all_pred_trajs, all_true_trajs