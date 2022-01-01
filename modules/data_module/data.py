import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class Data(Dataset):

    def __init__(self, dt: pd.DataFrame, window_size: int = 20):
        super().__init__()
        self.values = torch.from_numpy(dt.iloc[:, 2:].values).type(dtype=torch.float32)
        self.window_size = window_size
        self.num_feature = self.values.shape[1]

    def __getitem__(self, idx: int):
        # (N_batch, seq_length, num_feature)
        x = self.values[idx:(idx+self.window_size)].reshape((self.window_size, self.num_feature))
        y = self.values[idx+1:(idx+self.window_size+1), 0].reshape((self.window_size, 1))
        return x, y

    def __len__(self):
        return self.values.shape[0] - self.window_size