import torch
from torch import nn
from torch.autograd import Variable


device = 'cpu'

class LSTM(nn.Module):
    def __init__(self, n_grid, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.n_grid = n_grid  # points on the grid
        self.num_output = n_grid*2  # number of output: 32 real part + 32 complex part
        self.num_layers = 4  # number of layers LSTM
        self.input_size = n_grid*3  # input size: 32 real part +32 complex part +32 potential
        self.hidden_size = 2048  # hidden state
        self.seq_length = 1000  # # steps of time in trajectories

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True).to(device) #lstm

        #self.fc_1 =  nn.Linear(hidden_size, 1024) #fully connected 1

        #self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, num_output).to(device) #fully connected last layer

    def forward(self,x):

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        #hn = hn.view(-1,self.hidden_size) #reshaping the data for Dense layer next
        #out = self.relu(hn)
        #out = self.fc_1(out) #first Dense
        #out = self.relu(output) #relu
        out = self.fc(output).to(device) #Final Output
        return out