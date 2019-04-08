from collections import defaultdict
from multiprocessing import Process, Lock
import pickle
import os
import matplotlib
import numpy
from sklearn.linear_model import LogisticRegression

matplotlib.use('Agg')
import sys
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from AccumFairAdaCost import AccumFairAdaCost

sys.path.insert(0, 'DataPreprocessing')
sys.path.insert(0, 'equalized_odds_and_calibration-master')

from eq_odds import Model
from call_eq_odds import Model as calibModel
import funcs_disp_mist as fdm

import time
from sklearn.model_selection import StratifiedKFold
from AdaCost import AdaCostClassifier
from FairAdaCost import FairAdaCost
from load_kdd import load_kdd
from load_dutch_data import load_dutch_data
from load_german import load_german
from load_compas_data import load_compas_data
from load_adult_gender import load_adult_gender
from load_adult_race import load_adult_race
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
    suffixes = ['adaboost', 'Sin.1', 'Sin.2', 'Cumul.1', 'Cumul.2', 'zafar', 'Hardt', 'Pleiss']
    create_temp_files(dataset, suffixes)

    if dataset == "compass":
        X, y, sa_index, p_Group, x_control = load_compas_data()
    elif dataset == "adult-gender":
        X, y, sa_index, p_Group, x_control = load_adult_gender()
    elif dataset == "adult-race":
        X, y, sa_index, p_Group, x_control = load_adult_race()
    elif dataset == "dutch":
        X, y, sa_index, p_Group, x_control = load_dutch_data()
    elif dataset == "bank":
        X, y, sa_index, p_Group, x_control = load_bank()
    # elif dataset == "german":
    #     X, y, sa_index, p_Group, x_control = load_german()
    else:
        exit(1)

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

    for iter in range(0, 5):
        start = time.time()
        kf = StratifiedKFold(n_splits=2, random_state=int(time.time()), shuffle=True)

        for train_index, test_index in kf.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for proc in range(0, 8):
                if proc < 5:
                    threads.append(Process(target=train_classifier, args=( X_train, X_test, y_train, y_test, sa_index, p_Group, dataset + suffixes[proc], mutex[proc],proc, 100)))
                elif proc == 5:
                    temp_x_control_train = defaultdict(list)
                    temp_x_control_test = defaultdict(list)

                    temp_x_control_train[sensitive_attrs[0]] = x_control[sensitive_attrs[0]][train_index]
                    temp_x_control_test[sensitive_attrs[0]] = x_control[sensitive_attrs[0]][test_index]

                    x_train, y_train, x_control_train = ut.conversion(X[train_index], y[train_index],
                                                                      dict(temp_x_control_train), 1)
                    x_test, y_test, x_control_test = ut.conversion(X[test_index], y[test_index],
                                                                   dict(temp_x_control_test), 1)

                    threads.append(Process(target=train_zafar, args=(x_train, y_train, x_control_train,
                                                                     x_test, y_test, x_control_test,
                                                                     cons_params, loss_function, EPS,
                                                                     dataset + suffixes[proc], mutex[proc],
                                                                     sensitive_attrs)))
                elif proc == 6:
                    threads.append(Process(target=train_hardt, args=(X_train, X_test, y_train, y_test, sa_index,
                                                                     dataset + suffixes[proc], mutex[proc])))
                elif proc == 7:
                    threads.append(Process(target=train_pleiss, args=(X_train, X_test, y_train, y_test, sa_index,
                                                                      dataset + suffixes[proc], mutex[proc])))

    for process in threads:
        process.start()

    for process in threads:
        process.join()

    threads = []
    print "elapsed time for k-fold iteration = " + str(time.time() - start)

    results = []
    for suffix in suffixes:
        infile = open(dataset + suffix, 'rb')
        temp_buffer = pickle.load(infile)
        results.append(temp_buffer.performance)

        infile.close()

    plot_my_results(results, suffixes, "Images/" + dataset, dataset)
    delete_temp_files(dataset, suffixes)


def train_pleiss(X_train, X_test, y_train, y_test, sa_index, dataset, mutex):
    clf = LogisticRegression().fit(X_train, y_train)

    y_pred_probs = clf.predict_proba(X_test)[:, 1]

    temp_y = y_test
    temp_y[temp_y == -1] = 0

    my_df = pd.DataFrame(columns=['label', 'group', 'prediction'])

    for line in range(0, len(X_test)):
        my_df.loc[line] = [temp_y[line], X_test[line][sa_index], y_pred_probs[line]]

    order = numpy.random.permutation(len(my_df))
    val_indices = order[0::2]
    test_indices = order[1::2]

    val_data = my_df.iloc[val_indices]
    test_data = my_df.iloc[test_indices]
    group_0_val_data = val_data[val_data['group'] == 0]
    group_1_val_data = val_data[val_data['group'] == 1]

    protected_test_data = test_data[test_data['group'] == 0]
    non_protected_test_data = test_data[test_data['group'] == 1]

    protected_val_model = calibModel(group_0_val_data['prediction'].values, group_0_val_data['label'].values)
    non_protected_val_model = calibModel(group_1_val_data['prediction'].values, group_1_val_data['label'].values)

    protected_test_model = calibModel(protected_test_data['prediction'].values, protected_test_data['label'].values)
    non_protected_test_model = calibModel(non_protected_test_data['prediction'].values,
                                          non_protected_test_data['label'].values)


    _, _, mix_rates = calibModel.calib_eq_odds(protected_val_model, non_protected_val_model, 1, 1)
    eq_odds_protected_test_model, eq_odds_non_protected_test_model = calibModel.calib_eq_odds(protected_test_model,
                                                                                              non_protected_test_model,
                                                                                              1, 1,
                                                                                              mix_rates)

    results = calibModel.results(eq_odds_protected_test_model, eq_odds_non_protected_test_model)
    mutex.acquire()
    infile = open(dataset, 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()
    dict_to_ram.performance.append(results)
    outfile = open(dataset, 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()
    mutex.release()

def train_hardt(X_train, X_test, y_train, y_test, sa_index, dataset, mutex):
    clf = LogisticRegression().fit(X_train, y_train)

    y_pred_probs = clf.predict_proba(X_test)[:, 1]

    temp_y = y_test
    temp_y[temp_y == -1] = 0

    my_df = pd.DataFrame(columns=['label', 'group', 'prediction'])

    for line in range(0, len(X_test)):
        my_df.loc[line] = [temp_y[line], X_test[line][sa_index], y_pred_probs[line]]

    order = numpy.random.permutation(len(my_df))
    val_indices = order[0::2]
    test_indices = order[1::2]

    val_data = my_df.iloc[val_indices]
    test_data = my_df.iloc[test_indices]
    group_0_val_data = val_data[val_data['group'] == 0]
    group_1_val_data = val_data[val_data['group'] == 1]

    protected_test_data = test_data[test_data['group'] == 0]
    non_protected_test_data = test_data[test_data['group'] == 1]

    protected_val_model = Model(group_0_val_data['prediction'].values, group_0_val_data['label'].values)
    non_protected_val_model = Model(group_1_val_data['prediction'].values, group_1_val_data['label'].values)
    _, _, mix_rates = Model.eq_odds(protected_val_model, non_protected_val_model)


    protected_test_model = Model(protected_test_data['prediction'].values, protected_test_data['label'].values)
    non_protected_test_model = Model(non_protected_test_data['prediction'].values, non_protected_test_data['label'].values)
    eq_odds_protected_test_model, eq_odds_non_protected_test_model = Model.eq_odds(protected_test_model, non_protected_test_model, mix_rates)

    results = Model.results(eq_odds_protected_test_model, eq_odds_non_protected_test_model)
    mutex.acquire()
    infile = open(dataset, 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()
    dict_to_ram.performance.append(results)
    outfile = open(dataset, 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()
    mutex.release()


def train_zafar(x_train, y_train, x_control_train, x_test, y_test, x_control_test,
                cons_params, loss_function, EPS, dataset, mutex, sensitive_attrs):
    def train_test_classifier():
        w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)
        rates, accuracy, balanced_acc = fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test,
                                                          x_control_test, sensitive_attrs)
        return rates, accuracy, balanced_acc

    correct = False

    cnt = 1
    while correct != True:
        try:
            rates, acc, balanced_acc = train_test_classifier()
            correct = True
        except Exception, e:
            if cnt % 5 == 0:
                cons_params['tau'] /= 1.03
            print str(e) + ", tau = " + str(cons_params['tau'])
            cnt += 1
            pass

    results = dict()

    results["balanced_accuracy"] = balanced_acc
    results["accuracy"] = acc
    results["TPR_protected"] = rates["TPR_Protected"]
    results["TPR_non_protected"] = rates["TPR_Non_Protected"]
    results["TNR_protected"] = rates["TNR_Protected"]
    results["TNR_non_protected"] = rates["TNR_Non_Protected"]
    results["fairness"] = abs(rates["TPR_Protected"] - rates["TPR_Non_Protected"]) + abs(
        rates["TNR_Protected"] - rates["TNR_Non_Protected"])

    mutex.acquire()
    infile = open(dataset, 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()
    dict_to_ram.performance.append(results)
    outfile = open(dataset, 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()
    mutex.release()


def train_classifier(X_train, X_test, y_train, y_test, sa_index, p_Group, dataset, mutex, mode, base_learners):
    if mode == 0:
        classifier = AdaCostClassifier(saIndex=sa_index, saValue=p_Group, n_estimators=base_learners, CSB="CSB2")
    elif mode == 1:
        classifier = FairAdaCost(saIndex=sa_index, saValue=p_Group, n_estimators=base_learners, CSB="CSB1")
    elif mode == 2:
        classifier = FairAdaCost(saIndex=sa_index, saValue=p_Group, n_estimators=base_learners, CSB="CSB2")
    elif mode == 3:
        classifier = AccumFairAdaCost(saIndex=sa_index, saValue=p_Group, n_estimators=base_learners, CSB="CSB1")
    elif mode == 4:
        classifier = AccumFairAdaCost(saIndex=sa_index, saValue=p_Group, n_estimators=base_learners, CSB="CSB2")

    classifier.fit(X_train, y_train)

    y_pred_probs = classifier.predict_proba(X_test)[:, 1]
    y_pred_labels = classifier.predict(X_test)

    mutex.acquire()
    infile = open(dataset, 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()
    dict_to_ram.performance.append(
        calculate_performance(X_test, y_test, y_pred_labels, y_pred_probs, sa_index, p_Group))
    outfile = open(dataset, 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()
    mutex.release()


def main(dataset):
    run_eval(dataset)


if __name__ == '__main__':
    # main(sys.argv[1])
    main("german")
