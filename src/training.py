# src/train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import create_dataloader  # our dataloader without normalization
from model import LSTM                       # your LSTM class

# ------------------- Hyperparameters -------------------
file_path = '../data/DataNew/datadeltav2.h5'  # change dataset path
batch_size = 16
hidden_size = 128
num_layers = 2
num_epochs = 2
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------- Load data -------------------
dataloader = create_dataloader(file_path, batch_size=batch_size, verbose=False)
dataset = dataloader.dataset  # we can access the dataset object directly

seq_len = dataset.get_seq_len()
num_grid = dataset.get_num_grid_points()
num_samples = len(dataset)

print(f"💪 Training LSTM with seq_len={seq_len}, num_grid={num_grid}")
print(f"   Number of samples: {num_samples}, Batch size: {batch_size}")
print(f"   Device: {device}")

input_size = num_grid * 3
num_output = num_grid * 2

# ------------------- Model -------------------
model = LSTM(num_output=num_output, input_size=input_size, hidden_size=hidden_size,
             num_layers=num_layers)
model = model.to(device)

# ------------------- Loss & Optimizer -------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------- Training Loop -------------------
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)

    epoch_loss /= len(dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

print("✅ Training complete!")
