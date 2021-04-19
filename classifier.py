from collections import defaultdict
from utils import entropy
import numpy as np
import multiprocessing as mp
import time

global X, Y, maxDepth

class TreeNode:
    def __init__(self, idx, depth):

        self.idx = idx
        self.left, self.right, self.feature = None, None, None
        self.depth = depth
        self.num_pos = np.sum(DecisionTree.train_data.y[idx])
        self.num_neg = len(idx) - self.num_pos
        self.isLeaf = self.depth >= DecisionTree.maxDepth
        self.label = 1 if self.num_pos >= self.num_neg else 0
        self.split()

    def split(self):
        if self.isLeaf:


            return

        results = [self.info_gain(feature, feature_idx) for feature, feature_idx in DecisionTree.train_data.feature_dict.items()]


        best = max(results,key=lambda p:p[1])
        best_feature = best[0]

        mask = np.isin(self.idx, DecisionTree.train_data.feature_dict[best_feature])
        left_idx = self.idx[mask]
        right_idx = self.idx[~mask]
        if len(left_idx) == 0 or len(left_idx) == len(self.idx):
            self.isLeaf = True

            return
        self.feature = best_feature

        self.left = TreeNode(left_idx, self.depth + 1)
        self.right = TreeNode(right_idx, self.depth + 1)




    def info_gain(self,feature,feature_idx):
        mask = np.isin(self.idx, feature_idx)
        left_idx = self.idx[mask]
        pos_left = np.sum(DecisionTree.train_data.y[left_idx])
        neg_left = len(left_idx) - pos_left
        pos_right, neg_right = self.num_pos - pos_left, self.num_neg - neg_left
        left_entropy = entropy(pos_left, neg_left)
        right_entropy = entropy(pos_right, neg_right)
        left_total = len(left_idx)
        right_total = len(self.idx) - left_total
        total_entropy = entropy(self.num_pos, self.num_neg)
        return (feature,total_entropy - ((left_total / len(self.idx)) * left_entropy + (
                right_total / len(self.idx)) * right_entropy))


class Train_data:
    def __init__(self, X, y):
        self.y = y
        self.feature_dict = defaultdict(list)

        for i in range(len(X)):
            for value in X[i].values():
                self.feature_dict[value].append(i)
        for feature in self.feature_dict.keys():
            self.feature_dict[feature] = np.asarray(self.feature_dict[feature])


class DecisionTree:
    maxDepth, train_data = 100, None
    def fit(self,X_train, Y_train, max_depth=10):

        if len(X_train) != len(Y_train):
            raise ValueError("length of X and Y must be the same")
        DecisionTree.maxDepth = max_depth
        DecisionTree.train_data = Train_data(X_train,Y_train)
        self.root = TreeNode(np.arange(len(Y_train)),1)
    def predict_node(self, x, node):
        if node.isLeaf :
            return node.label
        if node.feature in x.values():
            return self.predict_node(x, node.left)
        else:
            return self.predict_node(x, node.right)

    def predict(self,X_test):
        pred_test = []
        for x in X_test:
            pred_test.append(self.predict_node(x,self.root))
        return np.array(pred_test)
    def score(self,X_test,Y_test):
        pred_test = self.predict(X_test)
        return 1.0 - np.sum(np.abs(Y_test - pred_test))/len(X_test)



