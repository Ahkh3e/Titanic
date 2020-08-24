import torch
from CleanData import clean_data
import pandas as pd

train = pd.read_csv('train.csv')

clean_data(train)
train_tensor = torch.tensor(train.values)
