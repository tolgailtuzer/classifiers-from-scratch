from scipy.spatial import distance


class KNNElement:
    def __init__(self, dist, label):
        self.dist = dist
        self.label = label


class KNN:
    def __init__(self, k=3):
        self.k = k

    # Fit method just store train data
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        distances = []
        # For each test sample calculates distance between test_sample and train_data_element
        # Similar class is selected according to the given k value
        for i in range(len(X_test)):
            distances.clear()
            for j in range(len(self.X_train)):
                distances.append(KNNElement(distance.euclidean(X_test.iloc[i, :], self.X_train.iloc[j, :]), self.y_train.iloc[j]))
            distances.sort(key=lambda x: x.dist)
            distances = [distances[i].label for i in range(self.k)]
            predictions.append(max(set(distances), key=distances.count))
        return predictions
