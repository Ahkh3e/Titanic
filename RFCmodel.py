import pandas as pd
import matplotlib.pyplot as plt
from CleanData import clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os

def RFC (train):

    clean_data(train)
    x = train.loc[:,["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]]
    y = train.loc[:,["Survived"]]

    x_train,x_test,y_train,y_test = train_test_split(x,y,
        test_size=0.2,
        random_state=2
    )
    forest_classifier = RandomForestClassifier(
        max_depth=3,
        random_state=1
    )
    forest_classifier.fit(x_train, y_train)

    pred=forest_classifier.predict(x_test)
    print(forest_classifier.score(x_test,y_test))

    gradientboost_clf = GradientBoostingClassifier(
    max_depth=3,
    random_state=1)
    gradientboost_clf.fit(x_train,y_train)
    gradientboost_pred = gradientboost_clf.predict(x_test)

    print (gradientboost_clf.score(x_test,y_test))
