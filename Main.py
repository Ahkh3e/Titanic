from DataIn import Data_in
from Data import GenData
from Datasex import SexData
from Model import model,graph
from RFCmodel import RFC
data = Data_in()


RFC(data)
#GenData(data)
#SexData(data)
model(data)
