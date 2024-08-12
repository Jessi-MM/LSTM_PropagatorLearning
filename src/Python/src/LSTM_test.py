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
import h5py
from LSTM_mudule import LSTM



seq_len = 200  # How many time stamps
n_grid = 32    # Points on the grid

path_dat = '/home/jessica/Documentos/Codigo_Limpio/src/DataLoader/Data/data_delta.h5'  # Directory where are saving our data. (Currently there are 8000)


# Define our dataset
dataset = Propagator_Dataset(path=path_dat, data='dataset_X', targets='dataset_y')
dataset_size = len(dataset)

#----- H Parameters:
batch_size = 10
epochs = 5
learning_rate = 1e-4

#----- Train and validation split:
test_split = 0.1
validation_split = 0.2  
shuffle_dataset = False  # to train after with same data
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

#print('Total of data ', dataset_size)
#print(f"Total of train samples: {len(train_sampler)}")
#print(f"Total of validation samples: {len(val_sampler)}")
print(f"Total of test samples: {len(test_sampler)}")

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Shape of data
for X, y in test_loader:
    print("Test data info:")
    print("-----------------------------------")
    print(f"Shape of X in test loader: {X.shape}")
    print(f"Shape of y in test loader: {y.shape}")
    break

def S_overlap(Psi_true, Psi_ANN, X):
    """
    Input:
    Psi_true: Evolution of wavepacket from dataset test, Shape: (batch size, sequence lenght, 64)
    Psi_ANN: Evolution of wavepacket predicted with the model, Shape: (batch size, sequence lenght, 64)
    X : Evolution of wavepacket at time t-1
    
    Output:
    S: Absolute magnitude
    angle: phase
    Characterizes the quality of the predictions. See equation (11) of Main article

    """
    
    Psi_true_re = Psi_true[:,:,0:n_grid] + X[:,:,0:n_grid]   # realpart of wavepacket predicted
    Psi_true_im = Psi_true[:,:,n_grid:n_grid*2] + X[:,:,n_grid:n_grid*2]  # imaginary part of wavepacket predicted 
    Psi_t = torch.view_as_complex(torch.stack((Psi_true_re,Psi_true_im), -1)).to(device)
    
    Psi_ANN_re = Psi_ANN[:,:,0:n_grid]+ X[:,:,0:n_grid]  # realpart of wavepacket predicted
    Psi_ANN_im = -(Psi_ANN[:,:,n_grid:n_grid*2]+ X[:,:,n_grid:n_grid*2])  # imaginary part of wavepacket predicted (- because conjugate)
    Psi_A = torch.view_as_complex(torch.stack((Psi_ANN_re,Psi_ANN_im), -1)).to(device)
    
    overl = Psi_A*Psi_t
    
    # Integrate over r (real integral + complex integral)
    # Trapezoid method in the grid r_n (angstroms -> au)
    
    r_n = (torch.linspace(-1.5,1.5,32)*(1/0.5291775)).to(device)
    overl_real = overl.real
    overl_imag = overl.imag
    
    real_integ = torch.trapz(overl_real, r_n).to(device)
    imag_integ = torch.trapz(overl_imag, r_n).to(device)
    
    # Covert to phase and magnitude of the complex result
    S =  torch.sqrt(real_integ**2 + imag_integ**2).to(device)
    angle = torch.arctan(imag_integ/real_integ).to(device)
    
    # Mean S & angle
    S = torch.sum(S)/(batch_size*seq_len)
    angle = torch.sum(angle)/(batch_size*seq_len)
    
    
    return S, angle



device = 'cpu'
path_model = '/home/jessica/Documentos/Trained_LSTM_Models/model5b.pth'
model = torch.load(path_model,map_location=torch.device('cpu'))


def test1(dataloader, model):
    '''
    Same as test function but without writer to tensorboard
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correctS, correct_phase = 0, 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            #test_loss += loss_fn(pred, y).item()
            S, angle = S_overlap(y,pred,X)  
            correctS += S
            correct_phase += angle
    
    correctS /= num_batches
    correct_phase /= num_batches

    print(f"Test Error: \n Accuracy Magnitude |S|: {(100*correctS):>0.1f}%")
    print(f"Test Error: \n Accuracy phase: {(correct_phase):>0.1f}\n")

test1(test_loader, model)
