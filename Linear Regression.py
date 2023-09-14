# Importing Modules/Packages
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Loading in Our Data
data = pd.read_csv("student-mat.csv",sep=";")
# Trimming Our Data (we need to select only the ones we want to use)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Separating Our Data

predict = "G3"
X = np.array(data.drop([predict], axis=1))
Y = np.array(data[predict])
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


# after running, it cmt it out
best = 0
for _ in range(30):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # train=90% and test=10% (because it is small dataset,still can be changed based on requirement)
    linear = linear_model.LinearRegression()

    linear.fit(X_train, Y_train)
    acc = linear.score(X_test, Y_test)
    print(acc)

    # if acc > best:
    #     best = acc
    #     with open("sentiment.pickle", "wb") as f:
    #         pickle.dump(linear, f)

# load model
# load model
# with open("sentiment.pickle", "rb") as f:
#     linear = pickle.load(f)

# Predicting on Specific Students
predictions = linear.predict(X_test)

for index in range(len(predictions)):
    print(predictions[index], X_test[index], Y_test[index])

acc = linear.score(X_test, Y_test)
print(acc)
# graph plot
p = "failures"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.legend(loc=4)
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()