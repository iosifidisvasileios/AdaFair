import numpy
import matplotlib.pyplot as plt

import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from AdaCost import AdaCostClassifier
from FairAdaCost import FairAdaCost
from load_kdd import load_kdd
from load_dutch_data import load_dutch_data
from load_compas_data import load_compas_data
from load_adult_gender import load_adult_gender
from load_adult_race import load_adult_race
from load_bank import load_bank
from my_useful_functions import calculate_fairness, plot_results


def run_eval(dataset=None):
    if dataset == "compass":
        X, y, sa_index, p_Group  = load_compas_data()
    elif dataset == "adult-gender":
        X, y, sa_index, p_Group = load_adult_gender()
    elif dataset == "adult-race":
        X, y, sa_index, p_Group = load_adult_race()
    elif dataset == "dutch":
        X, y, sa_index, p_Group = load_dutch_data()
    elif dataset == "kdd":
        X, y, sa_index, p_Group = load_kdd()
    elif dataset == "bank":
        X, y, sa_index, p_Group = load_bank()

    print pd.Series(X[:, sa_index]).value_counts()


    kf = StratifiedKFold(n_splits=10, random_state=int(time.time()), shuffle=True)

    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        original_classifier = DecisionTreeClassifier(max_depth=1)
        original_classifier.fit(X_train,y_train)


run_eval("compass")
# run_eval("adult-gender")
# run_eval("adult-race")
# run_eval("bank")
# run_eval("dutch")

