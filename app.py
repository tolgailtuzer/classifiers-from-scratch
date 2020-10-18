import streamlit as st
import pandas as pd
import numpy as np
import os
from knn import KNN
from decision_tree import DecisionTree
from random_forest import RandomForest
from utils import train_test_split, accuracy_score


def file_selector(folder_path='Datasets'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def run_knn(data, target_column):
    st.sidebar.title('Choose parameters for KNN')
    ts = st.sidebar.slider('Training size', min_value=0.0, max_value=1.0, step=0.01, value=0.7)
    k = st.sidebar.number_input('k', min_value=1, max_value=int(len(data)*ts), step=1, value=3)
    run_status = st.sidebar.button('Run Algorithm')
    if run_status:
        with st.spinner('Running...'):
            x_train, x_test, y_train, y_test = train_test_split(data.drop([target_column], axis=1),
                                                                data[target_column],
                                                                test_size=1 - ts)
            clf = KNN(k=k)
            clf.fit(x_train, y_train)
            """
            ## :dart: Accuracy
            """
            st.subheader(accuracy_score(y_test, clf.predict(x_test)))


def run_decision_tree(data, target_column):
    st.sidebar.title('Choose parameters for Decision Tree')
    ts = st.sidebar.slider('Training size', min_value=0.0, max_value=1.0, step=0.01, value=0.7)
    min_samples = st.sidebar.number_input('min_samples', min_value=1, max_value=int(len(data)*ts), step=1)
    if st.sidebar.checkbox('Specify Depth'):
        max_depth = st.sidebar.number_input('max_depth', min_value=1, max_value=int(len(data)*ts), step=1)
    else:
        max_depth = None
    run_status = st.sidebar.button('Run Algorithm')
    if run_status:
        with st.spinner('Running...'):
            x_train, x_test, y_train, y_test = train_test_split(data.drop([target_column], axis=1),
                                                                data[target_column],
                                                                test_size=1 - ts)
            clf = DecisionTree(min_samples=min_samples,
                               max_depth=max_depth)
            clf.fit(x_train, y_train)
            """
            ## :dart: Accuracy
            """
            st.subheader(accuracy_score(y_test, clf.predict(x_test)))


def run_random_forest(data, target_column):
    st.sidebar.title('Choose parameters for Random Forest')
    ts = st.sidebar.slider('Training size', min_value=0.0, max_value=1.0, step=0.01, value=0.7)
    n_estimators = st.sidebar.number_input('n_estimators', min_value=1, max_value=1000, step=1)
    n_features = st.sidebar.number_input('n_features', min_value=1, max_value=len(data.columns)-1, step=1, value=len(data.columns)-1)
    bootstrap_size = st.sidebar.number_input('bootstrap_size', min_value=1, max_value=int(len(data)*ts), step=1, value=int(len(data)*ts))
    if st.sidebar.checkbox('Specify Depth'):
        max_depth = st.sidebar.number_input('max_depth', min_value=1, max_value=int(len(data)*ts), step=1)
    else:
        max_depth = None
    run_status = st.sidebar.button('Run Algorithm')
    if run_status:
        with st.spinner('Running...'):
            x_train, x_test, y_train, y_test = train_test_split(data.drop([target_column], axis=1),
                                                                data[target_column],
                                                                test_size=1 - ts)
            clf = RandomForest(n_estimators=n_estimators,
                               n_features=n_features,
                               max_depth=max_depth,
                               bootstrap_size=bootstrap_size)
            clf.fit(x_train, y_train)
            """
            ## :dart: Accuracy
            """
            st.subheader(accuracy_score(y_test, clf.predict(x_test)))


def main():
    st.title(':rocket: Classifiers From Scratch')

    selected_file = file_selector()
    data = pd.read_csv(selected_file)
    """
    ## :page_facing_up: Head of the selected file
    """
    st.write(data.head())

    """
    ## :mag_right: Data Statistics
    """
    st.write(data.describe())

    target_column = st.selectbox("Select your target column for classification",
                                 data.columns,
                                 index=len(data.columns) - 1)

    """
    ## :milky_way: Select Classification Algorithm
    """
    clf_algorithm = st.selectbox("Select algorithm for classification",
                                 ['', 'KNN', 'DecisionTree', 'RandomForest'],
                                 index=0)

    if clf_algorithm == 'KNN':
        run_knn(data, target_column)
    elif clf_algorithm == 'DecisionTree':
        run_decision_tree(data, target_column)
    elif clf_algorithm == 'RandomForest':
        run_random_forest(data, target_column)


main()
