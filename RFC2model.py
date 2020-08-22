
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def model_2 (train):


    x = train.loc[:,["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]]
    y = train.loc[:,["Survived"]]

    x_train,x_test,y_train,y_test = train_test_split(x,y,
        test_size=0.2,
        random_state=0
    )

    model = RandomForestClassifier(n_estimators = 50, max_depth =5)
    model.fit(x_train,y_train)

    y_predict = model.predict(x_test)
    y_pred_quant = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    print("Training Accuracy :", model.score(x_train, y_train))
    print("Testing Accuracy :", model.score(x_test, y_test))
