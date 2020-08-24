from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
def modeling (train):

    x = train.loc[:,["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]]
    y = train.loc[:,["Survived"]]

    x_train,x_test,y_train,y_test = train_test_split(x,y,
        test_size=0.2,
    )
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, np.ravel(y_train,order='C'))
    y_pred = random_forest.predict(x_test)
    random_forest.score(x_train, y_train)
    acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
    print(acc_random_forest)
