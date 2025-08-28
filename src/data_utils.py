import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class PropagatorDataset(Dataset):
    def __init__(self, file_path):
        """
        file_path : path to .h5 dataset
        """
        self.file_path = file_path

        # Load dataset
        with h5py.File(file_path, 'r') as f:
            self.X = torch.tensor(f['dataset_X'][:], dtype=torch.float32)
            self.y = torch.tensor(f['dataset_y'][:], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_num_grid_points(self):
        # Assuming X.shape = (n_samples, seq_len, n_grid*3)
        return self.X.shape[2] // 3

    def get_seq_len(self):
        # Assuming X.shape = (n_samples, seq_len, n_grid*3)
        return self.X.shape[1]


def create_dataloader(file_path, batch_size=32, shuffle=True, verbose=True):
    dataset = PropagatorDataset(file_path)
    
    if verbose:
        # Print dataset info
        print(f"Dataset loaded from {file_path}")
        print(f"Number of samples: {len(dataset)}")
        print(f"Sequence length: {dataset.get_seq_len()}")
        print(f"Number of grid points: {dataset.get_num_grid_points()}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Test the DataLoader
if __name__ == "__main__":
    file_path = '../data/data_ngrid32_seq200_250827-212005.h5'  # Example path
    dataloader = create_dataloader(file_path, batch_size=32)
    
    for X_batch, y_batch in dataloader:
        print(f"X batch shape: {X_batch.shape}")
        print(f"y batch shape: {y_batch.shape}")
        break  # Just to test one batch