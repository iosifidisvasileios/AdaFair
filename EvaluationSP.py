import warnings
warnings.filterwarnings("ignore")

import random
from collections import defaultdict
from multiprocessing import Process, Lock
import pickle
import os
import matplotlib
from sklearn.model_selection import ShuffleSplit, StratifiedKFold

from AdaFairSP import AdaFairSP
from SMOTEBoost import SMOTEBoost
from margin import *
matplotlib.use('Agg')
import sys

sys.path.insert(0, 'DataPreprocessing')

import time

from AdaCost import AdaCostClassifier

from load_dutch_data import load_dutch_data
from load_compas_data import load_compas
from load_adult import load_adult
from load_diabetes import load_diabetes
from load_credit import load_credit
from load_kdd import load_kdd

from load_bank import load_bank
from my_useful_functions import calculate_performance_SP, plot_my_results_sp


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
    return calculate_performance_SP(X_test, y_test, y_pred_labels, y_pred_probs, sa_index, p_Group)

def run_eval(dataset, iterations):
    suffixes = [ 'Adaboost', 'AdaFair', 'SMOTEBoost' ]

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
    elif dataset == "credit":
        X, y, sa_index, p_Group, x_control = load_credit()
    elif dataset == "diabetes":
        X, y, sa_index, p_Group, x_control = load_diabetes()
    elif dataset == "kdd":
        X, y, sa_index, p_Group, x_control = load_kdd()

    else:
        exit(1)
    create_temp_files(dataset, suffixes)
    threads = []
    mutex = []
    for lock in range(0, 8):
        mutex.append(Lock())
    print (dataset)
    random.seed(int(time.time()))

    for iter in range(0, iterations):

        sss = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(time.time()))
        for train_index, test_index in sss.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for proc in range(0, 3):
                threads.append(Process(target=train_classifier, args=( X_train, X_test, y_train, y_test, sa_index, p_Group, dataset + suffixes[proc], mutex[proc],proc, 200, 1)))

    for process in threads:
        process.start()

    for process in threads:
        process.join()

    results = []
    for suffix in suffixes:
        infile = open(dataset + suffix, 'rb')
        temp_buffer = pickle.load(infile)
        results.append(temp_buffer.performance)
        infile.close()

    plot_my_results_sp(results, suffixes, "Images/StatisticalParity/" + dataset, dataset)
    delete_temp_files(dataset, suffixes)



def boostLearner(train, protectedIndex, protectedValue):
    marginAnalyzer = boostingMarginAnalyzer(train, protectedIndex, protectedValue)
    shift = marginAnalyzer.optimalShift()
    print('best shift is: %r' % (shift,))
    return marginAnalyzer.conditionalShiftClassifier(shift)



def train_classifier(X_train, X_test, y_train, y_test, sa_index, p_Group, dataset, mutex, mode, base_learners, c):
    if mode == 0:
        classifier = AdaCostClassifier(saIndex=sa_index, saValue=p_Group, n_estimators=base_learners, CSB="CSB1")
    elif mode == 1:
        classifier = AdaFairSP(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB="CSB2", c=c)
    elif mode == 2:
        classifier = SMOTEBoost(n_estimators=base_learners,saIndex=sa_index,n_samples=10, saValue=p_Group,  CSB="CSB1" )
    else:
        train=[]
        for i in range(len(X_train)):
              train.append((X_train[i],y_train[i]))
        classifier = boostLearner(train, sa_index, p_Group)
    
    if  mode in [0,1,2]:      
          classifier.fit(X_train, y_train)
          y_pred_probs = classifier.predict_proba(X_test)[:, 1]
          y_pred_labels = classifier.predict(X_test)

    else:
        y_pred_labels=[classifier(x) for x in X_test]
        y_pred_probs=[0 for i in y_pred_labels]

    mutex.acquire()
    infile = open(dataset, 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()
    dict_to_ram.performance.append(
        calculate_performance_SP(X_test, y_test, y_pred_labels, y_pred_probs, sa_index, p_Group))
    outfile = open(dataset, 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()
    mutex.release()


if __name__ == '__main__':
    run_eval("adult-gender", 2)
    # run_eval("adult-race", 10)
    # run_eval("compass-race", 10)
    # run_eval("compass-gender", 10)
    # run_eval("dutch", 10)
    # run_eval("bank", 10)
    # run_eval("diabetes", 10)
    # run_eval("kdd", 10)
