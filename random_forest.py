from decision_tree import DecisionTree
from multiprocessing import Pool
import multiprocessing
import pandas as pd
import numpy as np


class RandomForest:
    def __init__(self, n_estimators=128, n_features=None, max_depth=None, bootstrap_size=None):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.max_depth = max_depth
        self.bootstrap_size = bootstrap_size

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        # Creates random forest with using decision trees
        with Pool(int(multiprocessing.cpu_count()/2)) as pool:
            self.r_forest = pool.map(self.algorithm, range(self.n_estimators))

    def predict(self, x_test):
        data = [(self.r_forest[i], x_test.copy(deep=True)) for i in range(self.n_estimators)]
        with Pool(int(multiprocessing.cpu_count() / 2)) as pool:
            predictions = pool.map(self.predict_for_tree, data)
        return pd.concat(predictions, axis=1).mode(axis=1)[0]

    def predict_for_tree(self, params):
        tree, data = params[0], params[1]
        return tree.predict(data)

    def algorithm(self, i):
        # Prepares bootstrap data according to given bootstrap_size
        bootstrap_data_x, bootstrap_data_y = self.get_bootstrap_data()
        tree = DecisionTree(max_depth=self.max_depth, random_subspace=self.n_features)
        tree.fit(bootstrap_data_x, bootstrap_data_y)
        return tree

    def get_bootstrap_data(self):
        if self.bootstrap_size is None:
            self.bootstrap_size = len(self.x_train)
        bootstrap_indices = np.random.randint(0, len(self.x_train), size=self.bootstrap_size)
        return self.x_train.iloc[bootstrap_indices], self.y_train.iloc[bootstrap_indices]
