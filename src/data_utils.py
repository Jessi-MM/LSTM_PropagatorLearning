# data_utils.py
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class PropagatorDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = h5py.File(file_path, 'r')
        self.X = self.file_path['dataset_X']
        self.y = self.file_path['dataset_y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_item, y_item

    def get_num_grid_points(self):
        return self.X.shape[2] // 3

    def get_seq_len(self):
        return self.X.shape[1]

    def summary(self, name="Dataset"):
        """Print dataset summary info (for full dataset)."""
        info = [
            "-"*50,
            f"{name} summary:",
            f"Path                : {self.file_path}",
            f"Number of samples   : {len(self)}",
            f"Sequence length     : {self.get_seq_len()}",
            f"Grid points         : {self.get_num_grid_points()}",
            f"Input tensor shape  : {self.X.shape}",
            f"Target tensor shape : {self.y.shape}",
            "-"*50
        ]
        text = "\n".join(info)
        print(text)

def summary_subset(subset: Subset, name="Subset"):
    """Print info about a subset of a PropagatorDataset."""
    dataset = subset.dataset
    num_samples = len(subset)
    info = [
        f"{name} samples   : {num_samples}"
    ]
    text = "\n".join(info)
    print(text)

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split dataset into train/val/test subsets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size],
                        generator=torch.Generator().manual_seed(seed))


def create_dataloader(file_path, batch_size=32, shuffle=True, verbose=True, split=False, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Create DataLoader(s) from .h5 dataset file.

    Args:
        file_path : path to .h5 dataset
        batch_size : batch size for DataLoader
        shuffle : whether to shuffle the data
        verbose : whether to print dataset summary
        split : if True, split into train/val/test loaders
        ratios : tuple of (train, val, test) ratios, used if split=True
        seed : random seed for splitting

    Returns:
        - If split=False: single DataLoader
        - If split=True: dict with {"train": train_loader, "val": val_loader, "test": test_loader}
    """
    dataset = PropagatorDataset(file_path)

    if not split:
        if verbose:
            print("="*20+" 🗂️ Dataset Info "+"="*20)
            dataset.summary("Full dataset")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        train_set, val_set, test_set = split_dataset(dataset, *ratios, seed=seed)
        if verbose:
            print("="*20+" 🗂️ Dataset Info "+"="*20)
            dataset.summary("Full dataset")
            summary_subset(train_set, "Train set")
            summary_subset(val_set, "Validation set")
            summary_subset(test_set, "Test set")

        loaders = {
            "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_set, batch_size=batch_size, shuffle=False)
        }
        return loaders



# Test
if __name__ == "__main__":
    file_path = '../data/data_delta.h5'  # Example path
    dataloader = create_dataloader(file_path, batch_size=8, verbose=True, split=True)
   
    
    #for X_batch, y_batch in dataloader:
     #   print("Batch loaded successfully!")
      #  print(f"X batch shape: {X_batch.shape}")
       # print(f"y batch shape: {y_batch.shape}")
        #break  # Just to test one batch}
