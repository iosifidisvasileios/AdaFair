import random
from multiprocessing import Process, Lock
import pickle
import os
import matplotlib
import numpy
from sklearn.model_selection import ShuffleSplit

matplotlib.use('Agg')

import sys

from sklearn.model_selection import train_test_split

from AdaFair import AdaFair

sys.path.insert(0, 'DataPreprocessing')
sys.path.insert(0, 'equalized_odds_and_calibration-master')

# import funcs_disp_mist as fdm

import time
from AdaCost import AdaCostClassifier
from load_kdd import load_kdd
from load_dutch_data import load_dutch_data
# from load_german import load_german
from load_compas_data import load_compas
from load_adult import load_adult
from load_bank import load_bank
from my_useful_functions import calculate_performance, plot_results_of_c_impact


class serialazible_list():
    def __init__(self, steps):
        self.performance = {}
        for c in steps:
            self.performance[c] = []


def create_temp_files(dataset, suffixes,steps):
    for suffix in suffixes:
        outfile = open(dataset + suffix, 'wb')
        pickle.dump(serialazible_list(steps), outfile)
        outfile.close()

    if not os.path.exists("Images/"):
        os.makedirs("Images/")


def delete_temp_files(dataset, suffixes):
    for suffix in suffixes:
        os.remove(dataset + suffix)



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
    else:
        exit(1)

    suffixes = ['AdaFair NoConf.', 'AdaFair']
    random.seed(int(time.time()))

    base_learners = 200
    steps = numpy.arange(0, 1.001, step=0.2)

    create_temp_files(dataset, suffixes,steps)
    threads = []
    mutex = []
    for lock in range(0, 2):
        mutex.append(Lock())

    for iterations in range (0,iterations):
        start = time.time()

        sss = ShuffleSplit(n_splits=2, test_size=0.5)
        for train_index, test_index in sss.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for c in steps:
                threads.append(Process(target=train_classifier, args=(X_train, X_test, y_train, y_test, sa_index, p_Group, dataset + suffixes[1], mutex[1], 2, base_learners, c)))

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
    plot_results_of_c_impact(results[0], results[1], steps, "Images/", dataset)
    delete_temp_files(dataset, suffixes)



def train_classifier(X_train, X_test, y_train, y_test, sa_index, p_Group, dataset, mutex, mode, base_learners, c):
    if mode == 1:
        classifier = AdaFair(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB="CSB1", c=c)
    elif mode == 2:
        classifier = AdaFair(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB="CSB2", c=c)

    classifier.fit(X_train, y_train)

    y_pred_probs = classifier.predict_proba(X_test)[:, 1]
    y_pred_labels = classifier.predict(X_test)

    mutex.acquire()
    infile = open(dataset, 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()
    dict_to_ram.performance[c].append(calculate_performance(X_test, y_test, y_pred_labels, y_pred_probs, sa_index, p_Group))
    outfile = open(dataset, 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()
    mutex.release()


def main(dataset, iterations):
    run_eval(dataset, iterations)


if __name__ == '__main__':
    # run_eval(sys.argv[1], int(sys.argv[2]))
    main("compass-gender",10)
    # main("adult-gender", 10)
    # main("bank", 10)
    # main("kdd", 10)


