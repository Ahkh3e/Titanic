import matplotlib.pyplot as plt

def SexData(data):

    fig = plt.figure(figsize = (18,6))

    female_color = "#FA0000"

    plt.subplot2grid((2,3),(0,0))
    data.Survived.value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
    plt.title ("Survived")

    plt.subplot2grid((2,3),(0,1))
    data.Survived[data.Sex == "male"].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5)
    plt.title (" Men Survived")

    plt.subplot2grid((2,3),(0,2))
    data.Survived[data.Sex == "female"].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5, color = female_color)
    plt.title ("Women Survived")

    plt.subplot2grid((2,3),(1,0))
    data.Sex[data.Survived == 1].value_counts(normalize = True).plot(kind = "bar", alpha = 0.5, color = [female_color,'b'])
    plt.title ("Women vs Men Survival")

    plt.show()
