import random
from collections import defaultdict
from multiprocessing import Process, Lock
import pickle
import os
from scipy.stats import stats

import matplotlib
from sklearn.model_selection import ShuffleSplit

from SMOTEBoost import SMOTEBoost

matplotlib.use('Agg')
import sys

from AccumFairAdaCost import AccumFairAdaCost

sys.path.insert(0, 'DataPreprocessing')
sys.path.insert(0, 'equalized_odds_and_calibration-master')

import funcs_disp_mist as fdm

import time

from AdaCost import AdaCostClassifier

from load_dutch_data import load_dutch_data
from load_compas_data import load_compas
from load_adult import load_adult
from load_kdd import load_kdd

from load_bank import load_bank
from my_useful_functions import calculate_performance, plot_my_results
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


def run_eval(dataset):
    suffixes = ['Zafar et al.', 'Adaboost', 'AdaFair', 'SMOTEBoost' ]

    if dataset == "compass-gender":
        X, y, sa_index, p_Group, x_control = load_compas("sex")
    elif dataset == "compass-race":
        X, y, sa_index, p_Group, x_control = load_compas("race")
    elif dataset == "adult-gender":
        X, y, sa_index, p_Group, x_control = load_adult("sex")
    elif dataset == "adult-race":
        X, y, sa_index, p_Group, x_control = load_adult("race")
    elif dataset == "bank":
        X, y, sa_index, p_Group, x_control = load_bank()
    elif dataset == "kdd":
        X, y, sa_index, p_Group, x_control = load_kdd()

    else:
        exit(1)
    create_temp_files(dataset, suffixes)

    # init parameters for zafar method (default settings)
    tau = 5.0
    mu = 1.2
    cons_type = 4
    sensitive_attrs = x_control.keys()
    loss_function = "logreg"
    EPS = 1e-6
    sensitive_attrs_to_cov_thresh = {sensitive_attrs[0]: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}}
    cons_params = {"cons_type": cons_type, "tau": tau, "mu": mu,
                   "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

    threads = []
    mutex = []
    for lock in range(0, 8):
        mutex.append(Lock())
    base_learners = 200
    random.seed(int(time.time()))

    for iter in range(0, 1):

        sss = ShuffleSplit(n_splits=1, test_size=0.5)
        for train_index, test_index in sss.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            adaboost = AdaCostClassifier(saIndex=sa_index, saValue=p_Group, n_estimators=base_learners,CSB="CSB1")
            adafair = AccumFairAdaCost(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB="CSB2",c=1)
            smoteboost = SMOTEBoost(n_estimators=base_learners, saIndex=sa_index, n_samples=100, saValue=p_Group, CSB="CSB1")

            temp_x_control_train = defaultdict(list)
            temp_x_control_test = defaultdict(list)

            temp_x_control_train[sensitive_attrs[0]] = x_control[sensitive_attrs[0]][train_index]
            temp_x_control_test[sensitive_attrs[0]] = x_control[sensitive_attrs[0]][test_index]

            x_zafar_train, y_zafar_train, x_control_train = ut.conversion(X[train_index], y[train_index],dict(temp_x_control_train), 1)
            x_zafar_test, y_zafar_test, x_control_test = ut.conversion(X[test_index], y[test_index],dict(temp_x_control_test), 1)
            cnt = 1
            while True:
                if cnt > 41:
                    return
                try:
                    w = fdm.train_model_disp_mist(x_zafar_train, y_zafar_train, x_control_train, loss_function, EPS, cons_params)
                    _,_,_,preds = fdm.get_clf_stats(w, x_zafar_train, y_train, x_control_train, x_zafar_test, y_zafar_test, x_control_test, sensitive_attrs)
                    print 'solved!'
                    break
                except Exception, e:
                    print e
                    if cnt % 4 == 0:
                        cons_params['tau'] *= 1.10
                    cnt += 1


            adaboost.fit(X_train, y_train)
            adafair.fit(X_train, y_train)
            smoteboost.fit(X_train, y_train)

            adaboost_y_pred_labels = adaboost.predict(X_test)
            adafair_y_pred_labels = adafair.predict(X_test)
            smoteboost_y_pred_labels = smoteboost.predict(X_test)
            print "ada fair vs adaboost"
            calculate_significance(adafair_y_pred_labels, adaboost_y_pred_labels)
            print "ada fair vs smoteboost"
            calculate_significance(adafair_y_pred_labels, smoteboost_y_pred_labels)
            print "ada fair vs zafar"
            calculate_significance(adafair_y_pred_labels, preds)



def calculate_significance(a,b):
    t2, p2 = stats.ttest_rel(a, b)
    print("t = " + str(t2))
    print("p = " + str(p2))



if __name__ == '__main__':
    # run_eval(sys.argv[1], int(sys.argv[2]))
    # run_eval("compass-race", 5)
    # run_eval("compass-gender")
    run_eval("adult-gender")
    # run_eval("bank")
    # run_eval("kdd")
