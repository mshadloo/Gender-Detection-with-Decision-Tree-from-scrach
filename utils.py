
import numpy as np
def entropy(pos, neg):
    if (neg == 0 or pos == 0):
        return 0
    total = pos + neg
    pos_prob = pos / total
    neg_prob = 1.0 - pos_prob

    return -(pos_prob) * np.log2(pos_prob) - (neg_prob * np.log2(neg_prob))

def features(name):
    name = name.lower()
    return {
        'first-letter': name[0],
        'first2-letters': name[0:2],
        'first3-letters': name[0:3],

        'first4-letters': name[0:4],
        'first5-letters': name[0:5],
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
        'last4-letters': name[-4:],
        'last5-letters': name[-5:],
    }

def split_train_test(X, Y, train_split= 0.8):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X_train = X[idxs[:train_split*len(X)]]
    X_test = X[idxs[train_split*len(X):]]
    y_train = Y[idxs[:train_split * len(X)]]
    y_test = Y[idxs[train_split * len(X):]]
    return X_train, y_train, X_test, y_test