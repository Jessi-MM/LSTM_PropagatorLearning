# src/generate_data.py
import torch
from potentials import ProtonTransfer
import h5py
import time



def generate_dataset(n_grid=32, a=-1.5, b=1.5, seq_len=200, n_samples=10, directory='../data/'):
    """
    Generate dataset using ProtonTransfer DVR propagation and save to file.
    
    Args:
        n_grid (int): Number of grid points
        a (float): Left boundary of grid [Å]
        b (float): Right boundary of grid [Å]
        seq_len (int): Time steps to propagate
        n_samples (int): Number of trajectories
        directory (str): Directory to save the dataset
    """
    X = torch.empty((n_samples, seq_len, n_grid*3), dtype=torch.float64)
    y = torch.empty((n_samples, seq_len, n_grid*2), dtype=torch.float64)
    # The kinetic energy operator is constant, so we compute it once:
    dat = ProtonTransfer(n=n_grid, a=a, b=b, seq_len=seq_len, time=True, var_random=True)
    KE = dat.KINETIC_DVR()

    for i in range(n_samples):
        
        dat.evolution_wp(seq_len, step=1, gaussiana=True, T_DVR=KE)

        X[i] = dat.Xdat
        y[i] = dat.ydat

    # save as .h5
    timestr = time.strftime("%y%m%d-%H%M%S")
    # Build filename automatically
    filename = f"{directory}data_ngrid{n_grid}_seq{seq_len}_{timestr}.h5"

    with h5py.File(filename, "w") as f:
        f.create_dataset("dataset_X", data=X.numpy())
        f.create_dataset("dataset_y", data=y.numpy())

    print(f"✅ Dataset saved to {filename}")


if __name__ == "__main__":
    generate_dataset(n_samples=2)  # quick test run
