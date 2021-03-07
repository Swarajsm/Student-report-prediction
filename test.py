import pandas as pd
import numpy as np
import tensorflow
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0

'''for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_train, y_train)

    print(acc)

    if acc > best:
        with open("student_model.pickle", "wb") as f:
            pickle.dump(linear, f)'''


pickel_in = open("student_model.pickle", "rb")
linear = pickle.load(pickel_in)

print('Coeffient:  \n', linear.coef_)
print('Intercept: \n ', linear.intercept_)
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Change p on demand to change the representation
p = 'G1'
style .use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Grade")
pyplot.show()