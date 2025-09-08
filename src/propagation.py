# propagation.py
import torch

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
        traj_pred_tensor: predicted ψ trajectory (T, 2*num_grid_points) [on device]
        batch_true_tensor: ground truth ψ trajectory (T, 2*num_grid_points) [on device]
    """
    num_grid_points = X_seq.shape[-1] // 3
    T = X_seq.shape[0]

    # ✅ Ground truth ψ trajectory
    psi_init = X_seq[:, :2*num_grid_points].to(device)   # move to device
    y_seq = y_seq.to(device)
    batch_true_tensor = psi_init + y_seq


    prediction_len = T - seq_len_init
    if prediction_len <= 0:
        raise ValueError(f"seq_len_init >= trajectory length ({seq_len_init} >= {T}).")

    # Initial context sequence
    current_seq = X_seq[:seq_len_init].unsqueeze(0).to(device)  
    traj_pred = []
    delta_pred = []

    # Future potentials
    V_future = X_seq[seq_len_init:, 2*num_grid_points:].to(device)  

    # Hidden states
    h_0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)
    c_0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)

    # Coordinate grid for normalization: from Angstroms to a.u.
    r_n = torch.linspace(-1.5, 1.5, num_grid_points, device=device) * (1 / 0.5291775)

    with torch.no_grad():
        for t in range(prediction_len + 1):
            output, (h_0, c_0) = model(current_seq, (h_0, c_0))
            delta = output[:, -1, :].squeeze(0)  
            delta_pred.append(delta)

            last_psi = current_seq[0, -1, :2*num_grid_points]
            next_psi = last_psi + delta.squeeze(0)  # in a.u. units already

            # ✅ Normalize
            density = next_psi[:num_grid_points]**2 + next_psi[num_grid_points:]**2
            integral = torch.trapz(density, r_n)
            if integral > 0:
                next_psi = next_psi / integral.sqrt()

            traj_pred.append(next_psi)  # stays on GPU

            if t < prediction_len:
                V_next = V_future[t]
                new_input = torch.cat([next_psi, V_next], dim=-1).unsqueeze(0).unsqueeze(0)
                current_seq = torch.cat([current_seq[:, 1:, :], new_input], dim=1)

    traj_pred_tensor = torch.stack(traj_pred, dim=0)  # (T, 2*num_grid_points), still on GPU
    delta_pred_tensor = torch.stack(delta_pred, dim=0)  # (T-seq_len_init+1, 2*num_grid_points)

    return traj_pred_tensor, batch_true_tensor, delta_pred_tensor, y_seq 


def propagate_dataset(model, dataloader, device, seq_len_init=1):
    """
    Propagate all trajectories from a dataloader using autoregressive prediction.
    """
    model.eval()
    all_pred_trajs = []
    all_true_trajs = []

    all_delta_pred = []
    all_delta_true = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            for i in range(X_batch.shape[0]):
                X_seq = X_batch[i].to(device)
                y_seq = y_batch[i].to(device)

                traj_pred, batch_true, delta_pred, delta_true = propagate_trajectory(model, X_seq, y_seq, device, seq_len_init)
                all_pred_trajs.append(traj_pred)      # stays on GPU
                all_true_trajs.append(batch_true)     # stays on GPU
                all_delta_pred.append(delta_pred)    # stays on GPU
                all_delta_true.append(delta_true)

    return all_pred_trajs, all_true_trajs, all_delta_pred, all_delta_true
