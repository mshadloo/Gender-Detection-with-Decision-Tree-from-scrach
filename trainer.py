from utils import features
import pandas as pd
import numpy as np
from classifier2 import DecisionTree
from sklearn.utils import shuffle

names = pd.read_csv('names_dataset.csv')
TRAIN_SPLIT = 0.8
# Get the data out of the dataframe into a numpy matrix and keep only the name and gender columns


features = np.vectorize(features)
print(features(["Anna", "Hannah", "Paul"]))

# Extract the features for the whole dataset
X = features(list(names['name']))  # X contains the features

# Get the gender column
y= (names['sex'] == 'M').to_numpy().astype(int)


X, y = shuffle(X, y)
X_train, X_test = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
y_train, y_test = y[:int(TRAIN_SPLIT * len(y))], y[int(TRAIN_SPLIT * len(y)):]

clf = DecisionTree()
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))