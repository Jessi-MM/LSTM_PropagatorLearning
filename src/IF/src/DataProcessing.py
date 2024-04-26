# class to preparing data to LSTM 
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Dataset
import h5py

#-----

class Propagator_Dataset(Dataset):
    
    def __init__(self, path, data, targets, transform=True):
        
        self.data = data  # path of data X
        self.targets = targets  # path of labels y
        self.transform = transform  # to tensor
        
        self.hf = h5py.File(path, 'r')  # reading data
        
        
    def __getitem__(self, index):
        
        X = self.hf.get(self.data)[index]
        y = self.hf.get(self.targets)[index]
        
        if self.transform:
            X = torch.tensor(X)
            y = torch.tensor(y)
        
        return X, y
    
    def __len__(self):
        
        
        tot = len(self.hf.get(self.data))
        #tot = 100
        
        return tot
