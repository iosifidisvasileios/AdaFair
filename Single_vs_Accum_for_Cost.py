import random
from multiprocessing import Process, Lock
import pickle
import os
import matplotlib
from sklearn.model_selection import ShuffleSplit

matplotlib.use('Agg')
import sys

from AccumFairAdaCost import AccumFairAdaCost

sys.path.insert(0, 'DataPreprocessing')
sys.path.insert(0, 'equalized_odds_and_calibration-master')

import time
from FairAdaCost import FairAdaCost

from load_dutch_data import load_dutch_data
# from load_german import load_german
from load_compas_data import load_compas
from load_adult import load_adult
from load_kdd import load_kdd

from load_bank import load_bank
from my_useful_functions import calculate_performance, plot_my_results, plot_costs_per_round


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

def run_eval(dataset):



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
    else:
        exit(1)

    random.seed(int(time.time()))
    sss = ShuffleSplit(n_splits=1, test_size=0.5)

    for train_index, test_index in sss.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        adafairnoaccum= FairAdaCost(n_estimators=200, saIndex=sa_index, saValue=p_Group,   CSB="CSB2")
        adafairnoaccum.fit(X_train, y_train)

        adafair = AccumFairAdaCost(n_estimators=200, saIndex=sa_index, saValue=p_Group,     CSB="CSB2")
        adafair.fit(X_train, y_train)

    plot_costs_per_round("Images/" + dataset , adafairnoaccum.costs, adafair.costs)

def train_classifier(X_train, X_test, y_train, y_test, sa_index, p_Group, dataset, mutex, mode, base_learners):

    if mode == 0:
        classifier = FairAdaCost(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group,  debug=True, CSB="CSB2")
    elif mode == 1:
        classifier = AccumFairAdaCost(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, debug=True,   CSB="CSB2")

    classifier.fit(X_train, y_train)

    y_pred_probs = classifier.predict_proba(X_test)[:, 1]
    y_pred_labels = classifier.predict(X_test)

    mutex.acquire()
    infile = open(dataset, 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()
    dict_to_ram.performance.append(calculate_performance(X_test, y_test, y_pred_labels, y_pred_probs, sa_index, p_Group))
    outfile = open(dataset, 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()
    mutex.release()

def main(dataset):
    run_eval(dataset)

if __name__ == '__main__':
    # main(sys.argv[1], 10)
    main("compass-gender")
    # main("adult-gender")
    # main("bank")
    # main("kdd")
