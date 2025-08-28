import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_output, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_output = num_output
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_output)
    
    def forward(self, x, h_0=None, c_0=None):
        """
        x: input sequence (batch, seq_len, input_size)
        h_0, c_0: optional hidden and cell states
        """
        # If no hidden/cell states are provided, initialize to zeros (training mode)
        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.fc(output)  # Final output for all timesteps
        return out, (hn, cn)

