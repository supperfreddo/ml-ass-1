# import libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# load data (using pandas)
data_iris = pd.read_csv('data/iris_with_species.csv', header=0)
X = data_iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data_iris['species']

# build a Naïve Bayes model
clf = GaussianNB()
clf.fit(X, y)

# use the model to predict new example
predicted = clf.predict([[4.4, 3.6, 1.7, 0.2]])
print(predicted)