import matplotlib.pyplot as plt

def GenData(data):
    fig = plt.figure(figsize = (18,6))

    plt.subplot2grid((2,3),(0,0))
    data.Survived.value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
    plt.title ("Survived")

    plt.subplot2grid((2,3),(0,1))
    plt.scatter(data.Survived, data.Age, alpha = 0.1)
    plt.title("Age Survived")

    plt.subplot2grid((2,3),(0,2))
    data.Pclass.value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
    plt.title ("Class")

    plt.subplot2grid((2,3),(1,0), colspan = 2)
    for x in [1,2,3]:
        data.Age[data.Pclass == x].plot(kind = "kde")
    plt.title("Class wrt Age")
    plt.legend(("1st","2nd","3rd"))

    plt.subplot2grid((2,3),(1,2))
    data.Embarked.value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
    plt.title ("Embarked")


    plt.show()
