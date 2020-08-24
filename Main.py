from CleanData import clean_data
from Data import GenData
from Datasex import SexData
from Model import model,graph
from RFCmodel import RFC
from RFC2model import model_2
from modeling import modeling
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


clean_data(train)
clean_data(test)

train.info()
test.info()

print(train.head())
print(test.head())
#model_2(data)
#RFC(data)
#modeling(data)
#GenData(data)
#SexData(data)
model(train)
