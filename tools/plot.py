import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch


def plot(ground_truth: pd.Series, pred: np.ndarray, tag: int, save_file: str, error: float):
    if isinstance(pred, list):
        pred = np.asarray(pred)
    if isinstance(error, torch.Tensor):
        error = error.item()
    if not os.path.exists("output"):
        os.mkdir("output") 
    plt.figure()
    # plt.title(f"result: {error}")
    pre = np.ones_like(pred)
    post = np.ones_like(pred)
    pre[tag:] = pred[tag:] * np.inf
    post[:tag] = post[:tag] * np.inf

    plt.plot(ground_truth.values, label="ground truth", color="green")
    plt.plot(pred * pre, label="predication(train stage)", linestyle="--", color="purple")
    plt.plot(pred * post, label="predication(test stage)", linestyle="--", color="red")
    plt.legend()
    plt.axvline(x=tag, linestyle=":", color="cyan")
    plt.savefig(f"output/{save_file}.png", bbox_inches="tight")
    plt.close()