import numpy as np
import random


def train_test_split(x, y, test_size=0.3):
    test_indices = random.sample(list(x.index), int(test_size*len(x)))
    return x.drop(test_indices), x.loc[test_indices], y.drop(test_indices), y.loc[test_indices]


def accuracy_score(y_test, y_predict):
    return (np.array(y_test).reshape(len(y_test), 1) == np.array(y_predict).reshape(len(y_predict), 1)).mean()

