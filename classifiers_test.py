import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn import KNN
from decision_tree import DecisionTree


if __name__ == '__main__':
    df = pd.read_csv('Datasets/iris.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # KNN Classifier
    start = time.time()
    knn = KNN(4)
    knn.fit(X_train.copy(deep=True), y_train.copy(deep=True))
    predictions = knn.predict(X_test.copy(deep=True))
    print(f"Custom KNN Accuracy: {accuracy_score(y_test, predictions)} Elapsed Time: {time.time() - start}")

    # DecisionTree Classifier
    start = time.time()
    dt = DecisionTree()
    dt.fit(X_train.copy(deep=True), y_train.copy(deep=True))
    prediction = dt.predict(X_test.copy(deep=True))
    print(f"Custom Decision Tree Accuracy: {accuracy_score(y_test, prediction)} Elapsed Time: {time.time() - start}")
