import torch
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from DataProcessing import Propagator_Dataset  # from this project


from torch.utils.tensorboard import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import os

import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib import rc


path = '../../Data_Gaussian/data'  # Directory where are saving our data. (Currently there are 4700)
seq_len = 200  # How many time stamps
n_grid = 32    # Number of points on the grid
tot_data = 8000  # Number of trajectories

dataset = Propagator_Dataset(data=path, targets=path, transform=True, total_data = tot_data*200, sequence_len=seq_len, n_grid=n_grid)
dataset_size = len(dataset)

#----- Train and validation split:
test_split = 0.1
validation_split = 0.2  
shuffle_dataset = False
random_seed= 42

# Creating data indices for training and validation splits:
indices = list(range(dataset_size))
split_val = int(np.floor(validation_split * dataset_size))
split_test = int(np.floor(test_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
test_indices = indices[0:split_test] 
val_indices = indices[split_test:split_test+split_val]   
train_indices = indices[split_test+split_val:]

# Creating PT data samplers and loaders:

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

print('Total of data ', dataset_size)
print(f"Total of train samples: {len(train_sampler)}")
print(f"Total of validation samples: {len(val_sampler)}")
print(f"Total of test samples: {len(test_sampler)}")

batch_size = 10
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Shape of data
for X, y in train_loader:
    print("Train data info:")
    print("-----------------------------------")
    print(f"Shape of X in train loader: {X.shape}")
    print(f"Shape of y in train loader: {y.shape}")
    break

#----- LSTM model
#device = 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, num_output, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_output = num_output  # number of output
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        
        #self.fc_1 =  nn.Linear(hidden_size, 1024) #fully connected 1

        #self.relu = nn.ReLU()
        
        self.fc = nn.Linear(hidden_size, num_output) #fully connected last layer
    
    def forward(self,x):
        
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        #hn = hn.view(-1,self.hidden_size) #reshaping the data for Dense layer next
        #out = self.relu(hn)
        #out = self.fc_1(out) #first Dense
        #out = self.relu(output) #relu
        out = self.fc(output) #Final Output
        return out

input_size = n_grid*3  # number of features: 32 real part +32 complex part +32 potential
hidden_size = 1024  # number of features in hidden state
num_layers = 2  # number of stacked lstm layers

num_output = n_grid*2  # number of output: 32 real part + 32 complex part
sequence_len = seq_len # lenght of time steps (1 fs each one) total 5 fs
learning_rate = 1e-4

print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")

model = LSTM(num_output, input_size, hidden_size, num_layers, sequence_len) #our lstm class
# Initialize the loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  #weight_decay=0.01 <- default
print('----- Sumary of Model -----')
print(model)
print('---------------------------')


#----- Accuracy function
def S_overlap(Psi_true, Psi_ANN, X):
    """
    Input:
    Psi_true: Evolution of wavepacket from dataset test, Shape: (batch size, sequence lenght, 64)
    Psi_ANN: Evolution of wavepacket predicted with the model, Shape: (batch size, sequence lenght, 64)
    
    Output:
    S: Absolute magnitude
    angle: phase
    Characterizes the quality of the predictions. See equation (11) of Main article
    
    """
    S_tot = []
    angle_tot = []
    for j in range(batch_size):
        for l in range(seq_len):
            Psi_true_re = Psi_true[j,l,0:n_grid] + X[j,l,0:n_grid] # real part of wavepacket
            Psi_true_im = Psi_true[j,l,n_grid:n_grid*2] + X[j,l,n_grid:n_grid*2] # imaginary part of wavepacket
            Psi_t = torch.view_as_complex(torch.stack((Psi_true_re,Psi_true_im), -1))
    
    
            Psi_ANN_re = Psi_ANN[j,l,0:n_grid]+ X[j,l,0:n_grid]  # realpart of wavepacket predicted
            Psi_ANN_im = -(Psi_ANN[j,l,n_grid:n_grid*2]+ X[j,l,n_grid:n_grid*2])  # imaginary part of wavepacket predicted (- because conjugate)
            Psi_A = torch.view_as_complex(torch.stack((Psi_ANN_re,Psi_ANN_im), -1))
        
        
        
            overlap = []
            for i in range(n_grid):
                overlap.append(torch.tensor([Psi_A[i]*Psi_t[i]]))
            overl = torch.tensor(overlap)
        
            # Integrate over r (real integral + complex integral)
            # Simpson method in the grid r_n (angstroms -> au)
            r_n = np.linspace(-1.5,1.5,n_grid)*(1/0.5291775)
            overl_real = overl.real.numpy()
            overl_imag = overl.imag.numpy()
    
            real_integ = integrate.simpson(overl_real, r_n)
            imag_integ = integrate.simpson(overl_imag, r_n)
    
            # Covert to phase and magnitude of the complex result
            S_tot.append(np.sqrt(real_integ**2 + imag_integ**2))
            angle_tot.append(np.arctan(imag_integ/real_integ))
        
    S = sum(S_tot)/(batch_size*seq_len)
    angle = sum(angle_tot)/(batch_size*seq_len)
    
    return S, angle

#----- To use tensorboard
#com = input('Give a comment to SumaryWriter:')
#print('Example: Update2LSTM_1024neu_seq200_BATCH_10_LR_1E-4_4700DATA')
com = 'Model5-LSTM'
writer = SummaryWriter(comment=com)  # To use tensorboard

for X,y in train_loader:
    writer.add_graph(model,X)  # to draw diagram model
    break

#----- Train loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)#len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.squeeze().to(device)

        # Compute prediction error
        pred = model(X.float()).squeeze()
        loss = loss_fn(pred, y.float())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    writer.add_scalar("Loss/train", loss, epoch)

#----- Test loop
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correctS, correct_phase = 0, 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:                                                                                         
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            S, angle = S_overlap(y, pred, X)  
            correctS += S
            correct_phase += angle
    
    correctS /= num_batches
    correct_phase /= num_batches
    test_loss /= num_batches
    
    writer.add_scalar('Accuracy Magnitude |S| /test', 100*correctS, epoch)  # Should be 100%
    writer.add_scalar('Accuracy phase /test', correct_phase, epoch)  # Should be 0
    writer.add_scalar("Loss/validation", test_loss, epoch)

    

    print(f"Test Error: \n Accuracy Magnitude |S|: {(100*correctS):>0.1f}%")
    print(f"Test Error: \n Accuracy phase: {(correct_phase):>0.1f}\n")

#----- Training
epochs = 5
for epoch in range(0,epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_loader, model, criterion, optimizer)
    test(val_loader, model, criterion)
    
writer.flush()

PATH = '../Models'
if not os.path.exists(PATH):
    os.makedirs(PATH)
    
torch.save(model, './Models/Model5-5epochs.pth')