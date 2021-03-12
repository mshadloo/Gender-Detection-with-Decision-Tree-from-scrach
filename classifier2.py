from collections import defaultdict
from utils import entropy
import numpy as np
class DecisionTree:
    global X, Y, maxDepth
    class TreeNode:
        def __init__(self,idx, depth):
            self.idx = idx
            self.left, self.right, self.feature = None, None, None
            self.depth = depth
            self.num_pos = np.sum(Y[idx])
            self.num_neg = len(idx) - self.num_pos
            self.isLeaf = False
            self.label = 1 if self.num_pos >= self.num_neg else 0
            self.split()
        def split(self):
            if self.depth == maxDepth:
                self.isLeaf = True
                return
            best_gain = - float('inf')
            best_feature = None
            feature_split = defaultdict(list)
            for i in self.idx:
                for value in X[i].values():
                    feature_split[value].append(i)
            for feature, left_idx in feature_split.items():
                feature_gain = self.info_gain(left_idx)
                if feature_gain > best_gain:
                    best_gain = feature_gain
                    best_feature = feature

            left_idx = feature_split[best_feature]
            if len(left_idx) == 0 or len(left_idx) == len(self.idx):
                self.isLeaf = True
                return
            right_idx = list(set(self.idx).difference(set(left_idx)))
            self.feature = best_feature
            self.left = DecisionTree.TreeNode(left_idx, self.depth + 1)
            self.right = DecisionTree.TreeNode(right_idx, self.depth + 1)
        def info_gain(self,idx_left):
            pos_left = np.sum(Y[idx_left])
            neg_left = len(idx_left) - pos_left
            pos_right, neg_right = self.num_pos - pos_left, self.num_neg - neg_left
            left_entropy = entropy(pos_left, neg_left)
            right_entropy = entropy(pos_right, neg_right)
            left_total = len(idx_left)
            right_total= len(self.idx) - left_total
            total_entropy = entropy(self.num_pos,self.num_neg)
            return total_entropy - ((left_total / len(self.idx)) * left_entropy + (
                    right_total / len(self.idx)) * right_entropy)



    def fit(self,X_train, Y_train, max_depth=100):

        if len(X_train) != len(Y_train):
            raise ValueError("length of X and Y must be the same")
        global X,Y, maxDepth
        X, Y, maxDepth = X_train, Y_train, max_depth
        self.root = DecisionTree.TreeNode(list(range(len(X_train))),1)
        print("fit is done")
    def predict_node(self, x, node):
        if node.isLeaf :
            return node.label
        if node.feature in x.values():
            return self.predict_node(x, node.left)
        else:
            return self.predict(x, node.right)

    def predict(self,X_test):
        print("predict")
        pred_test = []
        for x in X_test:
            pred_test.append(self.predict_node(x,self.root))
        return np.array(pred_test)
    def score(self,X_test,Y_test):
        pred_test = self.predict(X_test)
        print(pred_test)
        return 1.0 - np.sum(np.abs(Y_test - pred_test))/len(Y_test)



