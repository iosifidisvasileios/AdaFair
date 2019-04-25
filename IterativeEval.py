import random
from collections import defaultdict
from multiprocessing import Process, Lock
import pickle
import os
import matplotlib
import numpy
from sklearn.model_selection import ShuffleSplit

matplotlib.use('Agg')
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from AccumFairAdaCost import AccumFairAdaCost

sys.path.insert(0, 'DataPreprocessing')
sys.path.insert(0, 'equalized_odds_and_calibration-master')

# import funcs_disp_mist as fdm

import time
from AdaCost import AdaCostClassifier

from load_dutch_data import load_dutch_data
from load_compas_data import load_compas
from load_adult import load_adult
from load_kdd import load_kdd
from load_bank import load_bank

from my_useful_functions import calculate_performance, plot_my_results, plot_per_round
import utils as ut


class serialazible_list(object):
    def __init__(self):
        self.performance = []


def create_temp_files(dataset, suffixes):
    for suffix in suffixes:
        outfile = open(dataset + suffix, 'wb')
        pickle.dump(serialazible_list(), outfile)
        outfile.close()

    if not os.path.exists("Images/"):
        os.makedirs("Images/")


def delete_temp_files(dataset, suffixes):
    for suffix in suffixes:
        os.remove(dataset + suffix)


def predict(clf, X_test, y_test, sa_index, p_Group):
    y_pred_probs = clf.predict_proba(X_test)[:, 1]
    y_pred_labels = clf.predict(X_test)
    return calculate_performance(X_test, y_test, y_pred_labels, y_pred_probs, sa_index, p_Group)


def run_eval(dataset, iterations):

    if dataset == "compass-gender":
        X, y, sa_index, p_Group, x_control = load_compas("sex")
    elif dataset == "compass-race":
        X, y, sa_index, p_Group, x_control = load_compas("race")
    elif dataset == "adult-gender":
        X, y, sa_index, p_Group, x_control = load_adult("sex")
    elif dataset == "adult-race":
        X, y, sa_index, p_Group, x_control = load_adult("race")
    elif dataset == "dutch":
        X, y, sa_index, p_Group, x_control = load_dutch_data()
    elif dataset == "bank":
        X, y, sa_index, p_Group, x_control = load_bank()
    elif dataset == "kdd":
        X, y, sa_index, p_Group, x_control = load_kdd()
        suffixes = [None,'Adaboost', 'AdaFair']

    else:
        exit(1)

    base_learners = 200
    random.seed(12345)

    # for iter in range(0, iterations):
        # start = time.time()

        # sss = ShuffleSplit(n_splits=1, test_size=0.01)

        # for train_index, test_index in sss.split(X, y):
        #
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]

    classifier = AccumFairAdaCost(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB="CSB2", c=1,
                                  debug=True)
    classifier.fit(X,y)
    # print classifier.performance
    plot_per_round(base_learners, classifier.performance, classifier.objective, "Images/Round_" + dataset + ".png")









def main(dataset, iterations=5):
    run_eval(dataset,iterations)


if __name__ == '__main__':
    # main(sys.argv[1], int(sys.argv[2]))
    main("compass-gender", 1)
    main("adult-gender", 1)
    main("bank", 1)
