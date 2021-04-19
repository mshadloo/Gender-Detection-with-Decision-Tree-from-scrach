from utils import features
import pandas as pd
import numpy as np
from classifier import DecisionTree
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--d', default=10, type=int, metavar='N',
                    help='maximum depth of decision tree')
args = parser.parse_args()
names = pd.read_csv('names_dataset.csv')
TRAIN_SPLIT = 0.8
# Get the data out of the dataframe into a numpy matrix and keep only the name and gender columns


features = np.vectorize(features)


# Extract the features for the whole dataset
X = features(list(names['Name']))  # X contains the features

# Get the gender column
y= (names['Gender'] == 'M').to_numpy().astype(int)


X, y = shuffle(X, y)
X_train, X_test = X[:int(TRAIN_SPLIT * len(X))], X[int(TRAIN_SPLIT * len(X)):]
y_train, y_test = y[:int(TRAIN_SPLIT * len(y))], y[int(TRAIN_SPLIT * len(y)):]
if __name__ == '__main__':
  clf = DecisionTree()
  st = time.time()
  clf.fit(X_train,y_train, args.d)
  test_acc= clf.score(X_test, y_test)
  
  print('accuracy of test: ',test_acc)
  train_acc = clf.score(X_train, y_train)
  print('accuracy of train: ',train_acc)

