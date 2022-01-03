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
import numpy as np
from modules.metrics.utils import AccRate
from ptflops import get_model_complexity_info
from modules.net_module.gan import GAN


def parameter_stat(net: nn.Module):
    # total_params = sum([p.numel() for p in net.parameters()])
    # trainable_params = sum([p.numel() for p in net.parameters() if p.requires_grad])
    # print(f"total: {total_params}")
    # print(f"trainable: {trainable_params}")

    macs, params = get_model_complexity_info(net, input_res=(WINDOW_SIZE, 5), as_strings=False)
    print(macs)
    print(params)


def main():
    acc_rate = AccRate()

    for c_idx in comp_idxs:
        csv = glob(os.path.join(split_data_root, f"{c_idx}.csv"))[0]
        dt = pd.read_csv(csv)
        dt[["收盘价","开盘价","最高价","最低价","成交量"]] = (dt[["收盘价","开盘价","最高价","最低价","成交量"]] - dt[["收盘价","开盘价","最高价","最低价","成交量"]].mean()) / dt[["收盘价","开盘价","最高价","最低价","成交量"]].std()
        length = len(dt)
        train_len = round(length * 0.8) 
        train_dt = dt[:train_len]
        test_dt = dt[(train_len-WINDOW_SIZE+1):]
        lstm = LSTM()
        # parameter_stat(lstm)
        gan = GAN(window_size=WINDOW_SIZE, epoch=500)
        exit(0)

        optim = torch.optim.AdamW(lstm.parameters(), lr=1e-4)
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()

        train_data = Data(train_dt, window_size=WINDOW_SIZE)
        train_data = DataLoader(train_data, batch_size=1, pin_memory=True, num_workers=4)

        test_data = Data(test_dt, window_size=WINDOW_SIZE)
        test_data = DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=4)

        ground_truth = dt[WINDOW_SIZE:len(dt)-1].iloc[:, 2]

        # lstm
        if "lstm" in methods:
            train_outputs = train_step(lstm, optim, train_data, criterion)
            test_outputs, test_x = test_step(lstm, test_data)
            with open("temp.txt", "w") as fp:
                print(*test_outputs, sep="\n", file=fp)
            er = acc_rate(torch.as_tensor(test_outputs), torch.as_tensor(test_x))
            plot(ground_truth, train_outputs + test_outputs, train_len - WINDOW_SIZE, f"LSTM_{c_idx}", error=er)


        # gan
        if "gan" in methods:
            gan = GAN(window_size=WINDOW_SIZE, epoch=500)
            train_outputs = gan.train(train_data)
            test_outputs, test_x = gan.test(test_data)
            er = acc_rate(torch.as_tensor(test_outputs), torch.as_tensor(test_x))
            plot(ground_truth, train_outputs + test_outputs, train_len - WINDOW_SIZE, f"GAN_{c_idx}", error=er)
        
        
if __name__ == "__main__":
    # comp_idxs = ["000590.SZ", "000591.SZ", "000592.SZ"]
    comp_idxs = np.loadtxt("comp_ids.txt", dtype=str).tolist()
    if isinstance(comp_idxs, str):
        comp_idxs = [comp_idxs] 
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    data_root = config["data"]["root"]
    split_data_root = os.path.join(data_root, "split")

    WINDOW_SIZE = 40

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, nargs="+")
    CONFIG = vars(parser.parse_args())
    methods = CONFIG["method"]
    methods = [v.lower() for v in methods]

    main()