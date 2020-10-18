import pandas as pd
import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k

    # Fit method just store train data
    def fit(self, x_train, y_train):
        self.x_train = pd.get_dummies(x_train)
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        x_test = pd.get_dummies(x_test)
        # For each test sample calculates distance between test_sample and train_data_element
        # Similar class is selected according to the given k value
        for i in range(len(x_test)):
            distances = self.euclidean_distance(x_test.iloc[i, :], self.x_train)
            distances = np.append(distances.reshape(-1, 1), self.y_train.values.reshape(-1, 1), axis=1)
            distances = distances[distances[:, 0].argsort()]
            distances = list(distances[:self.k, 1])
            predictions.append(max(set(distances), key=distances.count))
        return pd.DataFrame(predictions)

    def euclidean_distance(self, element1, element2):
        return np.sqrt(np.sum((element1.values-element2.values)**2, axis=1))


