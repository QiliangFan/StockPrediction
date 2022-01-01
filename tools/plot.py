import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(ground_truth: pd.Series, pred: np.ndarray, tag: int):
    plt.figure()
    plt.plot(ground_truth.values, label="ground truth", color="green")
    plt.plot(pred, label="predication", linestyle="--", color="purple")
    plt.axvline(x=tag, linestyle=":", color="cyan")
    plt.savefig("result.png", bbox_inches="tight")
    plt.close()