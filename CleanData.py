def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    data.drop('Cabin',
    axis=1, inplace=True)
    data.Sex[data.Sex == "male"] = 0
    data.Sex[data.Sex == "female"] = 1

    data["Embarked"] = data ["Embarked"].fillna("S")
    data.Embarked[data.Embarked == "S"] = 0
    data.Embarked[data.Embarked == "C"] = 1
    data.Embarked[data.Embarked == "Q"] = 2
