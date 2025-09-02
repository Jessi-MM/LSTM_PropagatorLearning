import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def log_density_plot(writer, pred_traj, true_traj, step, num_grid_points):
    """
    Log probability density comparison between prediction and ground truth.

    pred_traj: (T, 2*num_grid_points)
    true_traj: (T, 2*num_grid_points)
    """
    
    T = pred_traj.shape[0]
    x = torch.linspace(-1.5, 1.5, num_grid_points) # in Angströms


    pred_real = pred_traj[:, :num_grid_points] 
    pred_imag = pred_traj[:, num_grid_points:] 
    true_real = true_traj[:, :num_grid_points] 
    true_imag = true_traj[:, num_grid_points:] 
    # Compute densities
    pred_density = pred_real**2 + pred_imag**2
    true_density = true_real**2 + true_imag**2

    # Conversion from a.u. to |Ψ(x)|² in 1D: (length unit)^-1
    conversion = 1/0.5291775
    pred_density = pred_density * conversion
    true_density = true_density * conversion

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


def log_density_heatmap(writer, pred_traj, true_traj, step, num_grid_points):
    """
    Log a heatmap of probability density over time.

    x-axis: timesteps
    y-axis: positions
    color: |psi(r,t)|^2
    """

    T = pred_traj.shape[0]
    x = torch.linspace(-1.5, 1.5, num_grid_points)  # positions in Angströms
    timesteps = torch.arange(T)  # time axis


    pred_real = pred_traj[:, :num_grid_points] 
    pred_imag = pred_traj[:, num_grid_points:] 
    true_real = true_traj[:, :num_grid_points] 
    true_imag = true_traj[:, num_grid_points:] 

    # Compute densities
    pred_density = pred_real**2 + pred_imag**2
    true_density = true_real**2 + true_imag**2

    # Conversion from a.u. to |Ψ(x)|² in 1D: (length unit)^-1
    conversion = 1/0.5291775
    pred_density = pred_density * conversion
    true_density = true_density * conversion

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axs[0].imshow(
        true_density.T, origin='lower', aspect='auto',
        extent=[0, T-1, x[0], x[-1]], cmap='viridis'
    )
    axs[0].set_title("True |ψ(r,t)|²")
    axs[0].set_xlabel("Time step")
    axs[0].set_ylabel("Position [$\AA$]")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(
        pred_density.T, origin='lower', aspect='auto',
        extent=[0, T-1, x[0], x[-1]], cmap='viridis'
    )
    axs[1].set_title("Predicted |ψ(r,t)|²")
    axs[1].set_xlabel("Time step")
    axs[1].set_ylabel("Position [$\AA$]")
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    writer.add_figure("Density Heatmap", fig, global_step=step)
    plt.close(fig)
