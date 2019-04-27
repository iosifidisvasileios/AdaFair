import random
from multiprocessing import Process, Lock
import pickle
import os
import matplotlib
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB

matplotlib.use('Agg')
import sys
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from AccumFairAdaCost import AccumFairAdaCost

sys.path.insert(0, 'DataPreprocessing')
sys.path.insert(0, 'equalized_odds_and_calibration-master')

from call_eq_odds import Model as calibModel

import time
from AdaCost import AdaCostClassifier
from FairAdaCost import FairAdaCost

from load_dutch_data import load_dutch_data
# from load_german import load_german
from load_compas_data import load_compas
from load_adult import load_adult
from load_kdd import load_kdd

from load_bank import load_bank
from my_useful_functions import calculate_performance, plot_my_results


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

def run_eval(dataset, iterations):
    suffixes = ['NC AdaFair', 'AdaFair']
    create_temp_files(dataset, suffixes)

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
    create_temp_files(dataset, suffixes)



    threads = []
    mutex = []
    for lock in range(0, 2):
        mutex.append(Lock())

    random.seed(int(time.time()))

    for iter in range(0, iterations):
        start = time.time()

        sss = ShuffleSplit(n_splits=2, test_size=0.5)

        for train_index, test_index in sss.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for proc in range(0, 2):

                threads.append(Process(target=train_classifier, args=(X_train, X_test, y_train, y_test, sa_index, p_Group, dataset + suffixes[proc], mutex[proc], proc, 200)))

    for process in threads:
        process.start()

    for process in threads:
        process.join()

    threads = []

    print "elapsed time = " + str(time.time() - start)

    results = []
    for suffix in suffixes:
        infile = open(dataset + suffix, 'rb')
        temp_buffer = pickle.load(infile)
        results.append(temp_buffer.performance)
        infile.close()

    plot_my_results(results, suffixes, "Images/" + dataset + "_single_vs_accum", dataset)
    delete_temp_files(dataset, suffixes)

def train_classifier(X_train, X_test, y_train, y_test, sa_index, p_Group, dataset, mutex, mode, base_learners):

    if mode == 0:
        classifier = FairAdaCost(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group,  CSB="CSB2")
    elif mode == 1:
        classifier = AccumFairAdaCost(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group,  CSB="CSB2")

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

def main(dataset, iterations=5):
    run_eval(dataset,iterations)

if __name__ == '__main__':
    # main(sys.argv[1], 10)
    main("kdd", 1)
