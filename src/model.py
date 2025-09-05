import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LSTM_with_Attention(nn.Module):
    def __init__(self, num_output, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_output = num_output
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        # Attention weights
        self.attn = nn.Linear(hidden_size, hidden_size)
        # Final output
        self.fc = nn.Linear(hidden_size, num_output)

    def forward(self, x, h_0=None, c_0=None):
        batch_size, seq_len, _ = x.size()

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # output: (batch, seq_len, hidden_size)

        # Attention mechanism
        # Compute attention weights (batch, seq_len, seq_len)
        attn_weights = torch.bmm(output, output.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Compute context vector
        context = torch.bmm(attn_weights, output)  # (batch, seq_len, hidden_size)

        # Combine context with output (element-wise sum or concat)
        attn_out = output + context  # simple addition

        out = self.fc(attn_out)  # (batch, seq_len, num_output)
        return out, (hn, cn)