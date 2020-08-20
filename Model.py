from CleanData import clean_data
from sklearn import tree,model_selection
def model (train):
    clean_data(train)

    target = train["Survived"].values
    features_names = ["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]
    features = train[features_names].values

    generalized_tree = tree.DecisionTreeClassifier(
        random_state = 1,
        max_depth = 7,
        min_samples_split = 2
    )
    generalized_tree_ = generalized_tree.fit(features, target)

    print(generalized_tree_.score(features,target))

    scores = model_selection.cross_val_score(generalized_tree, features,target, scoring = 'accuracy' , cv =50)
    print (scores)
    print (scores.mean())

def graph():
    tree.export_graphviz(generalized_tree_ , feature_names = features_names, out_file = 'tree.dot')
