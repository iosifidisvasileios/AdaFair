from multiprocessing import Process, Lock
import pickle
import os

import sys

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'DataPreprocessing')

import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from AdaCost import AdaCostClassifier
from FairAdaCost import FairAdaCost
from load_kdd import load_kdd
from load_dutch_data import load_dutch_data
from load_compas_data import load_compas_data
from load_adult_gender import load_adult_gender
from load_adult_race import load_adult_race
from load_bank import load_bank
from my_useful_functions import calculate_fairness, plot_results, plot_calibration_curves


class my_dict(object):
    def __init__(self, increase_cost, max_cost, step):
        self.performance = dict()
        self.weights = dict()
        self.fraction_of_positives = dict()
        self.mean_predicted_value = dict()
        # self.predicted_probabilities = []
        self.init_arrays(increase_cost, max_cost, step)

    def init_arrays(self, increase_cost, max_cost, step):
        for i in range (increase_cost, max_cost+step, step):
            self.performance[i] = []
            self.weights[i] = []
            self.fraction_of_positives[i] = []
            self.mean_predicted_value[i] = []


def create_temp_files(dataset, increase_cost, max_cost, step, suffixes):
    for suffix in suffixes:
        outfile = open(dataset + suffix, 'wb')
        pickle.dump(my_dict(increase_cost, max_cost, step), outfile)
        outfile.close()

    if not os.path.exists("Images/"+dataset):
        os.makedirs("Images/"+dataset)

    if not os.path.exists("Images/" + dataset+ "/CalibrationCurves/"):
        os.makedirs("Images/"+dataset + "/CalibrationCurves/")

def delete_temp_files(dataset, suffixes):
    for suffix in suffixes:
        os.remove(dataset + suffix)

def run_eval(dataset, iterations = 15, init_cost = 0, kfold = 5, max_cost = 5, step = 1, num_base_learners = 10):

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
    else:
        exit(1)

    print pd.Series(X[:, sa_index]).value_counts()
    suffixes = ["_original","_only_costs","_costs_and_votes","_original_calibrated","_only_costs_calibrated","_costs_and_votes_calibrated",]

    create_temp_files(dataset, init_cost, max_cost, step, suffixes)

    threads = []
    mutex = []
    for lock in range (0,6):
        mutex.append(Lock())
    for cost in range(init_cost, max_cost + step, step):
    # while init_cost <= max_cost :
        costs = [1 + cost / 100., 1] # [pos , neg]
        print "cost = " + str(cost) + " out of " + str(max_cost) + " with step = " + str(step)
        for iter in range(0,iterations):
            start = time.time()
            kf = StratifiedKFold(n_splits=kfold, random_state=int(time.time()), shuffle=True)
            for train_index, test_index in kf.split(X,y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                for proc in range(0,6):
                    threads.append(Process(target=train_classifier, args=(X_train, X_test, y_train, y_test, sa_index, p_Group, costs, cost, dataset, mutex[proc], num_base_learners, proc, "CSB1")))

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
        results.append(temp_buffer)
        if suffix == "_original" or  suffix == "_only_costs":
            plot_results(init_cost, max_cost, step, temp_buffer.performance, temp_buffer.weights, "Images/" + dataset + "/" + suffix[1:],"AdaCost" + suffix)
        else:
            plot_results(init_cost, max_cost, step, temp_buffer.performance, temp_buffer.weights, "Images/" + dataset + "/" + suffix[1:],"AdaCost" + suffix, False)
        infile.close()

    plot_calibration_curves(results, suffixes,init_cost, max_cost, step, "Images/"+dataset + "/CalibrationCurves/")
    delete_temp_files(dataset,suffixes)

def train_classifier(X_train, X_test, y_train, y_test, sa_index,p_Group, costs,  increase_cost, dataset, mutex, num_base_learners, mode, CSB):
    if mode == 0 or mode == 3:
        classifier = AdaCostClassifier(saIndex=sa_index, saValue=p_Group, costs=costs, n_estimators=num_base_learners, updateAll=False, CSB=CSB)
    elif mode == 1 or mode == 4:
        classifier = FairAdaCost(saIndex=sa_index, saValue=p_Group, costs=costs, n_estimators=num_base_learners, updateAll=False, CSB=CSB)
    elif mode == 2 or mode == 5:
        classifier = FairAdaCost(saIndex=sa_index, saValue=p_Group, costs=costs, n_estimators=num_base_learners, useFairVoting=True, updateAll=False, CSB=CSB)

    if mode <= 2:
        classifier.fit(X_train, y_train)
        y_pred_probs = classifier.predict_proba(X_test)[:, 1]
        y_pred_labels = classifier.predict(X_test)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.20)
        classifier.fit(X_train, y_train)
        sigmoid = CalibratedClassifierCV(classifier, method='sigmoid', cv="prefit")
        sigmoid.fit(X_valid, y_valid)

        y_pred_probs = sigmoid.predict_proba(X_test)[:, 1]
        y_pred_labels = sigmoid.predict(X_test)

    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_probs, n_bins=10)

    if mode == 0:
        directory_string = "_original"
    elif mode == 1:
        directory_string = "_only_costs"
    elif mode == 2:
        directory_string = "_costs_and_votes"
    elif mode == 3:
        directory_string = "_original_calibrated"
    elif mode == 4:
        directory_string = "_only_costs_calibrated"
    elif mode == 5:
        directory_string = "_costs_and_votes_calibrated"

    mutex.acquire()

    infile = open(dataset + directory_string, 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()

    dict_to_ram.performance[increase_cost].append(calculate_fairness(X_test, y_test, y_pred_labels, y_pred_probs, sa_index, p_Group))
    dict_to_ram.weights[increase_cost].append(classifier.get_weights())

    dict_to_ram.fraction_of_positives[increase_cost].append(fraction_of_positives)
    dict_to_ram.mean_predicted_value[increase_cost].append(mean_predicted_value)

    outfile = open( dataset+ directory_string, 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()

    mutex.release()


def main(dataset):
    starting= time.time()
    run_eval(dataset)

    print time.time() - starting

if __name__ == '__main__':
    # main(sys.argv[1])
    main("adult-gender")