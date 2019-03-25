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

    iterations = 10
    increase_cost = 0
    max_cost = 2
    step = 1
    num_base_learners = 25


    original = my_dict(increase_cost, max_cost, step)
    only_costs = my_dict(increase_cost, max_cost, step)
    costs_and_votes = my_dict(increase_cost, max_cost, step)

    original_calibrated = my_dict(increase_cost, max_cost, step)
    only_costs_calibrated = my_dict(increase_cost, max_cost, step)
    costs_and_votes_calibrated = my_dict(increase_cost, max_cost, step)

    outfile1 = open( dataset+ "_original", 'wb')
    outfile2 = open( dataset+ "_only_costs", 'wb')
    outfile3 = open( dataset+ "_costs_and_votes", 'wb')

    outfile4 = open( dataset+ "_only_costs_calibrated", 'wb')
    outfile5 = open( dataset+ "_costs_and_votes_calibrated", 'wb')
    outfile6 = open( dataset+ "_original_calibrated", 'wb')

    pickle.dump(original, outfile1)
    pickle.dump(only_costs, outfile2)
    pickle.dump(costs_and_votes, outfile3)
    pickle.dump(only_costs_calibrated, outfile4)
    pickle.dump(costs_and_votes_calibrated, outfile5)
    pickle.dump(original_calibrated, outfile6)

    outfile1.close()
    outfile2.close()
    outfile3.close()
    outfile4.close()
    outfile5.close()
    outfile6.close()

    mutex = []
    for lock in range (0,6):
        mutex.append(Lock())

    threads = []

    while increase_cost <= max_cost :
        costs = [1 + increase_cost/100., 1] # [pos , neg]
        print "cost = " + str(increase_cost) + " out of " + str(max_cost)
        for iter in range(0,iterations):
            start = time.time()
            kf = StratifiedKFold(n_splits=10, random_state=int(time.time()), shuffle=True)
            for train_index, test_index in kf.split(X,y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                for proc in range(0,6):
                    threads.append(Process(target=train_classifier, args=(X_train, X_test, y_train, y_test, sa_index,p_Group, costs, increase_cost, dataset, mutex[proc], num_base_learners, proc)))

            for process in threads:
                process.start()

            for process in threads:
                process.join()

            threads = []
        print "elapsed time for k-fold iteration = " + str(time.time() - start)

        increase_cost += step

    infile = open(dataset + "_original", 'rb')
    original = pickle.load(infile)
    infile.close()

    infile = open(dataset + "_only_costs", 'rb')
    only_costs = pickle.load(infile)
    infile.close()

    infile = open(dataset + "_costs_and_votes", 'rb')
    costs_and_votes = pickle.load(infile)
    infile.close()

    # calibrated classifiers
    infile = open(dataset + "_original_calibrated", 'rb')
    original_calibrated = pickle.load(infile)
    infile.close()

    infile = open(dataset + "_only_costs_calibrated", 'rb')
    only_costs_calibrated = pickle.load(infile)
    infile.close()

    infile = open(dataset + "_costs_and_votes_calibrated", 'rb')
    costs_and_votes_calibrated = pickle.load(infile)
    infile.close()

    if not os.path.exists("Images/"+dataset):
        os.makedirs("Images/"+dataset)

    if not os.path.exists("Images/" + dataset+ "/CalibrationCurves/"):
        os.makedirs("Images/"+dataset + "/CalibrationCurves/")

    plot_results(0, max_cost, step, original.performance, original.weights, "Images/"+dataset + "/original", "Original AdaCost")
    plot_results(0, max_cost, step, only_costs.performance, only_costs.weights, "Images/"+dataset + "/only_costs", "AdaCost with dynamic costs")
    plot_results(0, max_cost, step, costs_and_votes.performance, costs_and_votes.weights, "Images/"+dataset + "/fully_fair", "AdaCost with dynamic costs and fair votes", False)

    # calibrated
    plot_results(0, max_cost, step, only_costs_calibrated.performance, only_costs_calibrated.weights, "Images/"+dataset + "/only_costs_calibrated", "AdaCost-Calibrated with dynamic costs", False)
    plot_results(0, max_cost, step, costs_and_votes_calibrated.performance, costs_and_votes_calibrated.weights, "Images/"+dataset + "/costs_and_votes_calibrated", "AdaCost-Calibrated with dynamic costs and fair votes", False)
    plot_results(0, max_cost, step, original_calibrated.performance, original_calibrated.weights, "Images/"+dataset + "/original_calibrated", "Original AdaCost-Calibrated", False)

    plot_calibration_curves([original,only_costs, costs_and_votes, only_costs_calibrated, costs_and_votes_calibrated, original_calibrated],
                            ["AdaCost", "+OnlyCosts" , "+Costs+Votes", "AdaCost_Calibrated", "+OnlyCosts_Calibrated" , "+Costs+Votes_Calibrated" ],
                            max_cost, step, "Images/"+dataset + "/CalibrationCurves/")


def train_classifier(X_train, X_test, y_train, y_test, sa_index,p_Group, costs,  increase_cost, dataset, mutex, num_base_learners, mode):
    if mode == 0 or mode == 3:
        classifier = AdaCostClassifier(saIndex=sa_index, saValue=p_Group, costs=costs, n_estimators=num_base_learners)
    elif mode == 1 or mode == 4:
        classifier = FairAdaCost(saIndex=sa_index, saValue=p_Group, costs=costs, n_estimators=num_base_learners)
    elif mode == 2 or mode == 5:
        classifier = FairAdaCost(saIndex=sa_index, saValue=p_Group, costs=costs, n_estimators=num_base_learners, useFairVoting=True)


    classifier.fit(X_train, y_train)

    if mode <= 2:
        y_pred_probs = classifier.decision_function(X_test)
        y_pred_labels = classifier.predict(X_test)
    else:
        sigmoid = CalibratedClassifierCV(classifier, method='isotonic', cv="prefit")
        sigmoid.fit(X_train, y_train)
        y_pred_probs = sigmoid.predict_proba(X_test)[:, 1]
        y_pred_labels = sigmoid.predict(X_test)


    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_probs, n_bins=10, normalize=True)


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
    main("compass")