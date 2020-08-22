from DataIn import Data_in
from CleanData import clean_data
from Data import GenData
from Datasex import SexData
from Model import model,graph
from RFCmodel import RFC
from RFC2model import model_2

data = Data_in()
clean_data(data)

data.info()


#RFC(data)clean_data(train)
model_2(data)
#GenData(data)
#SexData(data)
#model(data)
