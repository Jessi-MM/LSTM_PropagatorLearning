import torch

def compute_overlap(psi_true, psi_pred, dx=1.0):
    """
    Compute overlap S = <psi_true | psi_pred>.
    
    psi_true, psi_pred : Tensor of shape (T, 2*num_grid_points)
        Trajectories containing [real, imag] components.
    dx : float
        Grid spacing (default = 1.0, can adjust based on your units).

    Returns:
        |S| (float), theta (float, radians)
    """
    num_grid_points = psi_true.shape[1] // 2

    # convert to complex form
    psi_true_c = psi_true[:, :num_grid_points] + 1j * psi_true[:, num_grid_points:]
    psi_pred_c = psi_pred[:, :num_grid_points] + 1j * psi_pred[:, num_grid_points:]

    # flatten time and space dims into one vector
    psi_true_c = psi_true_c.reshape(-1)
    psi_pred_c = psi_pred_c.reshape(-1)

    # normalize
    norm_true = torch.sum(torch.abs(psi_true_c)**2) * dx
    norm_pred = torch.sum(torch.abs(psi_pred_c)**2) * dx
    psi_true_c = psi_true_c / torch.sqrt(norm_true)
    psi_pred_c = psi_pred_c / torch.sqrt(norm_pred)

    # inner product
    S = torch.sum(torch.conj(psi_true_c) * psi_pred_c) * dx

    return torch.abs(S).item(), torch.angle(S).item()
