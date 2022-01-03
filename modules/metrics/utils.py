import torch
from torch import nn


class AccRate:
    """
    https://flyai.com/d/StockPredict
    挑战赛上的反向误差率：
    """
    def __init__(self):
        pass

    def __call__(self, output: torch.Tensor, x: torch.Tensor):
        high = x[:, 2]
        low = x[:, 3]
        gt = x[:, 0]
        return 100 * (1 - torch.sum((output - gt) / (high - low))/x.shape[0])
