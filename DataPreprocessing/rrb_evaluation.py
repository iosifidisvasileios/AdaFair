import os,sys
from random import randint

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
import time
from load_kdd import *
from load_dutch_data import *
from load_compas_data import *
from load_adult_gender import *
from load_adult_race import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut

import funcs_disp_mist as fdm

def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/float(n) # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5


def test_compas_data(dataset="compass"):
    print "dataset = " + dataset
    values = dict()
    aucScores = []
    discScores = []

    """ Generate the synthetic data """
    data_type = 1
    train_fold_size = 1
    cons_type = 1  # FPR constraint -- just change the cons_type, the rest of parameters should stay the same

    if dataset ==  "compass":
        tau = 5.0
        mu = 1.2
        X, y, x_control = load_compas_data()
    elif dataset == "adult-gender":
        tau = 5.0
        mu = 1.2
        X, y, x_control = load_adult_gender()
    elif dataset == "adult-race":
        tau = 2.0
        mu = 1.2
        X, y, x_control = load_adult_race()
    elif dataset == "dutch":
        tau = 5.0
        mu = 1.2
        X, y, x_control = load_dutch_data()
    elif dataset == "kdd":
        tau = 5.0
        mu = 1.2
        X, y, x_control = load_kdd()

    sensitive_attrs = x_control.keys()

    sensitive_attrs_to_cov_thresh = {sensitive_attrs[0]: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}}  # zero covariance threshold, means try to get the fairest solution
    cons_params = {"cons_type": cons_type,
                   "tau": tau,
                   "mu": mu,
                   "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}


    for noist in range(1,7):
        iter = 0
        print noist

        while iter < 5:
            print iter

            kf = KFold(len(X), n_folds=3, random_state=int(time.time()), shuffle=True)

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)


            """ Split the data into train and test """
            for train_index, test_index in kf:
                temp_x_control_train = defaultdict(list)
                temp_x_control_test = defaultdict(list)

                temp_x_control_train[sensitive_attrs[0]] = x_control[sensitive_attrs[0]][train_index]
                temp_x_control_test[sensitive_attrs[0]] = x_control[sensitive_attrs[0]][test_index]




                x_train, y_train, x_control_train = ut.conversion(X[train_index], y[train_index], dict(temp_x_control_train), train_fold_size)
                x_test, y_test, x_control_test = ut.conversion(X[test_index], y[test_index], dict(temp_x_control_test), train_fold_size)


                changed = set()

                for index in range(0,len(y_train)):
                    if y_train[index] != 1 and x_control_train[sensitive_attrs[0]][index] == 1 and randint(0, 10) < noist:
                        y_train[index] = 1
                        changed.add(index)

                x_modified =[]
                y_modified = []
                x_control_modified = []

                for i in changed:
                    x_modified.append(x_train[i])
                    y_modified.append(y_train[i])
                    x_control_modified.append(x_control_train[sensitive_attrs[0]][i])

                x_control_modified = {sensitive_attrs[0]: np.array(x_control_modified )}


                # cons_params = None # constraint parameters, will use them later
                loss_function = "logreg" # perform the experiments with logistic regression
                EPS = 1e-6

                def train_test_classifier():
                    w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)
                    rrb = fdm.getRRBRate(w, x_train, y_train, x_control_train, np.array(x_modified), np.array(y_modified), x_control_modified, sensitive_attrs)
                    return rrb

                try:

                    w = train_test_classifier()
                    aucScores.append(w)
                    # discScores.append(disc)
                except Exception, error:
                    print "An exception was thrown!"
                    print str(error)

                    iter -=1
                    pass

                iter += 1
                break

        # print "dataset = " + dataset
        # print "noise level = " + str(noist) + ", rrb = " +str(mean(aucScores)) + ", st.Dev = " + str(stddev(aucScores))
        values[str(noist*10) +"%"]= str(mean(aucScores))

        # print "\n-----------------------------------------------------------------------------------\n"
    print "dataset = " + dataset

    print values


def main():
    test_compas_data()
    # test_compas_data("adult-race")
    # test_compas_data("dutch")
    # test_compas_data("adult-gender")
    # test_compas_data("kdd")


if __name__ == '__main__':
    main()