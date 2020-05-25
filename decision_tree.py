import numpy as np
import pandas as pd


class DecisionTreeElement:
    def __init__(self, question, left, right):
        self.question = question
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, min_samples=2, max_depth=None):
        self.min_samples = min_samples
        self.max_depth = max_depth

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        data = pd.concat([X_train, y_train], axis=1)
        self.feature_types = self.get_type_of_features(data)
        self.tree = self.algorithm(data, self.min_samples, self.max_depth)

    def predict(self, X_test):
        X_test.reset_index(inplace=True, drop=True)
        predictions = []
        for i in range(len(X_test)):
            predictions.append(self.predict_sample(self.tree, X_test.iloc[i, :]))
        return predictions

    def predict_sample(self, base_tree, element):
        # Question example: A <= 0.2
        question, operator, value = base_tree.question.split(' ')
        if operator == '<=':
            query = f"{element[question]}{operator}{float(value)}"
        else:
            query = f"'{element[question]}'{operator}'{value}'"

        # Simply traverse tree recursively according to the answers of questions
        if eval(query):
            chosen = base_tree.left
        else:
            chosen = base_tree.right

        if type(chosen).__name__ != "DecisionTreeElement":
            return chosen
        else:
            return self.predict_sample(chosen, element)

    def algorithm(self, data, min_samples, max_depth, depth=0):
        # If data has one class
        # If data_size less than specified sample_size
        # If tree_depth reaches the specified depth then classify data
        if self.is_pure(data) or len(data) < min_samples or depth == max_depth:
            return self.classify(data)
        else:
            depth += 1
            # Gets potential splits this is the average of two sequential data point for continuous elements
            potential_splits_ = self.potential_splits(data)
            # If there is no potential split then classify data
            if len(potential_splits_) == 0:
                return self.classify(data)
            # Determines best_split with using entropy calculation
            split_column, split_value = self.determine_best_split(data, potential_splits_)
            lower_data, upper_data = self.split_data(data, split_column, split_value)
            # Prepares question according to column type
            if self.feature_types[split_column] == 'continuous':
                question = f"{data.columns[split_column]} <= {split_value}"
            else:
                question = f"{data.columns[split_column]} == {split_value}"
            # Iterates this process recursively
            left = self.algorithm(lower_data, min_samples, max_depth, depth)
            right = self.algorithm(upper_data, min_samples, max_depth, depth)
            if left == right:
                return left
            else:
                return DecisionTreeElement(question, left, right)

    def is_pure(self, data):
        # Checks data purity. If data has one class type then we can say data is pure
        y = data.iloc[:, -1]
        return True if len(np.unique(y)) == 1 else False

    def potential_splits(self, data):
        splits = {}
        for i in range(len(data.columns) - 1):
            # Sorts unique sequential data points
            values = np.sort(np.unique(data.iloc[:, i]))
            if self.feature_types[i] == 'continuous' and len(values) > 1:
                temp = []
                # If data is continuous iteratively calculates average of sequential two item
                for j in range(1, len(values)):
                    temp.append((values[j] + values[j - 1]) / 2)
                splits[i] = temp
            elif len(values) > 1:
                # For categorical data just take unique values
                splits[i] = values
        return splits

    def split_data(self, data, split_column, split_value):
        if self.feature_types[split_column] == 'continuous':
            return data[data.iloc[:, split_column] <= split_value], data[data.iloc[:, split_column] > split_value]
        else:
            return data[data.iloc[:, split_column] == split_value], data[data.iloc[:, split_column] != split_value]

    def determine_best_split(self, data, potential_splits_):
        entropy = 9999
        split_column = -1
        split_value = -1
        # Calculates overall entropy for each split and chooses lowest one
        for key in potential_splits_:
            for value in potential_splits_[key]:
                lower, upper = self.split_data(data, key, value)
                t_entropy = self.calc_overall_entropy(lower, upper)
                if t_entropy <= entropy:
                    entropy = t_entropy
                    split_column = key
                    split_value = value
        return split_column, split_value

    def calc_entropy(self, data):
        y = data.iloc[:, -1]
        _, counts = np.unique(y, return_counts=True)
        probs = counts / sum(counts)
        return sum(probs * -np.log2(probs))

    def calc_overall_entropy(self, lower_data, upper_data):
        # Calculates entropy with formula
        n_data = len(lower_data) + len(upper_data)
        prob_lower = len(lower_data) / n_data
        prob_upper = len(upper_data) / n_data
        return prob_lower * self.calc_entropy(lower_data) + prob_upper * self.calc_entropy(upper_data)

    def classify(self, data):
        y = data.iloc[:, -1]
        labels, max_occur = np.unique(y, return_counts=True)
        return labels[np.argmax(max_occur)]

    def get_type_of_features(self, data, threshold=10):
        types = []

        for i in data.columns:
            unique_values = data[i].unique()

            if type(unique_values[0]) == 'str' or len(unique_values) <= threshold:
                types.append('categorical')
            else:
                types.append('continuous')

        return types
