import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split



rng = np.random


lb_Bin = LabelBinarizer()
imputer = SimpleImputer()
train = pd.read_csv("train.csv")
scaler = MinMaxScaler()



print(train.info())

train['Age'] = imputer.fit_transform(train['Age'].values.reshape(-1,1))
train['SibSp'] = imputer.fit_transform(train['SibSp'].values.reshape(-1,1))
train['Parch'] = imputer.fit_transform(train['Parch'].values.reshape(-1,1))

train["Embarked"] = train["Embarked"].fillna("S")

train.drop('Cabin',axis=1, inplace=True)
train.drop('Name',axis=1, inplace=True)
train.drop('Ticket',axis=1, inplace=True)
train.drop('PassengerId',axis=1, inplace=True)


lb_Sex = LabelEncoder()
lb_Embarked = LabelEncoder()
#lb_Ticket = LabelEncoder()

lb_Sex.fit(["male","female"])
lb_Embarked.fit(["S","Q","C"])
#lb.fit(["male","female"])

train["Sex"]=lb_Sex.transform(train["Sex"])
train["Embarked"]=lb_Embarked.transform(train["Embarked"])


train['Age'] = stats.zscore(train['Age'])

x = train.drop(["Survived"], axis=1)
y = train["Survived"]




#hyper_parameters
learning_rate = 0.01
training_epochs = 1000

#parameters
display_step = 50

train_X,test_X,train_Y,test_Y = train_test_split(x,y,
    test_size=0.2,
)

n_samples = train_X.shape[0]

X = tf.keras.Input(name="X", shape=(), dtype=tf.dtypes.float64)
Y = tf.keras.Input(name="Y", shape=(), dtype=tf.dtypes.float64)


W = tf.Variable(np.random.random(size=(1,)))
b = tf.Variable(np.random.random(size=(1,)))

# Construct a linear model (y=WX+b)
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error This is the error in the calculation to try to minimize

error = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default

optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate )

# Initialize the variables (i.e. assign their default value)


# Start training
with tf.compat.v1.Session() as sess:

    # Run the initializer

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X.ref(): x, Y.ref(): y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(error, feed_dict={X.ref(): train_X, Y.ref():train_Y})
            print("Epoch:", '%04d' % (epoch+1), "error=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_error = sess.run(error, feed_dict={X: train_X, Y: train_Y})
    print("Training error=", training_error, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)

    print("Testing... (Mean square loss Comparison)")
    testing_error = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing error=", testing_error)
    print("Absolute mean square loss difference:", abs(
        training_error - testing_error))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
