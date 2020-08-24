
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def model_2 (train):
    pred_rate = []



    x = train.loc[:,["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]]
    y = train.loc[:,["Survived"]]

    x_train,x_test,y_train,y_test = train_test_split(x,y,
        test_size=0.2,
    )

    model = AdaBoostClassifier(n_estimators =50,base_estimator = RandomForestClassifier(max_depth = 5, random_state = 1), learning_rate =1)
    model.fit(x_train,np.ravel(y_train,order='C'))

    y_predict = model.predict(x_test)
    y_pred_quant = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    print("Training Accuracy :", model.score(x_train, y_train))
    print("Testing Accuracy :", metrics.accuracy_score(y_test, y_pred))
