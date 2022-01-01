import torch
from torch import nn


class LSTM(nn.Module):

    def __init__(self, window_size: int = 20):
        super().__init__()

        hid_size = 20

        self.rnn = nn.LSTM(
            input_size=5, 
            hidden_size=hid_size,
            num_layers=2,
            batch_first=True,
        )
        self.reg = nn.Sequential(
            nn.Linear(hid_size, 1)
        )

    def forward(self, x: torch.Tensor, hidden = None):
        if hidden is not None:
            out, hid = self.rnn(x, hidden)
        else:
            out, hid = self.rnn(x)
        
        out = self.reg(out)
        return out, hid