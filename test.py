from modules.data_module.data import Data
import os
import pandas as pd

simplified = pd.read_csv("analysis/simplified.csv")
Data(simplified, "000007.SZ")