import torch
from torch import nn
from modules.net_module.lstm_wrapper import LSTM
from modules.data_module.data import Data
from modules.net_module.utils import train_step, test_step
import argparse
import yaml
import os
from glob import glob
import pandas as pd
from torch.utils.data import DataLoader
from tools.plot import plot

def main():
    for c_idx in comp_idxs:
        csv = glob(os.path.join(split_data_root, f"{c_idx}.csv"))[0]
        dt = pd.read_csv(csv)
        dt[["收盘价","开盘价","最高价","最低价","成交量"]] = (dt[["收盘价","开盘价","最高价","最低价","成交量"]] - dt[["收盘价","开盘价","最高价","最低价","成交量"]].mean()) / dt[["收盘价","开盘价","最高价","最低价","成交量"]].std()
        length = len(dt)
        train_len = round(length * 0.8) 
        train_dt = dt[:train_len]
        test_dt = dt[(train_len-WINDOW_SIZE+1):]
        lstm = LSTM(window_size=WINDOW_SIZE)
        optim = torch.optim.AdamW(lstm.parameters(), lr=1e-4)
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()

        train_data = Data(train_dt, window_size=WINDOW_SIZE)
        train_data = DataLoader(train_data, batch_size=1, pin_memory=True, num_workers=4)

        test_data = Data(test_dt, window_size=WINDOW_SIZE)
        test_data = DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=4)

        ground_truth = dt[WINDOW_SIZE:len(dt)-1].iloc[:, 2]
        train_outputs = train_step(lstm, optim, train_data, criterion)
        test_outputs = test_step(lstm, test_data)
        with open("temp.txt", "w") as fp:
            print(*test_outputs, sep="\n", file=fp)

        plot(ground_truth, train_outputs + test_outputs, train_len - WINDOW_SIZE)
        
        
if __name__ == "__main__":
    comp_idxs = ["000590.SZ"]
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    data_root = config["data"]["root"]
    split_data_root = os.path.join(data_root, "split")

    WINDOW_SIZE = 40

    main()