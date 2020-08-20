from DataIn import Data_in
from Data import GenData
from Datasex import SexData
from Model import model,graph

data = Data_in()

GenData(data)

SexData(data)

model(data)
