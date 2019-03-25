import os,sys
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

    aucScores = []
    discScores = []

    """ Generate the synthetic data """
    data_type = 1
    train_fold_size = 1
    cons_type = 4 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same

    if dataset ==  "compass":
        tau = 5.0
        mu = 1.2
        X, y, x_control = load_compas_data()
    elif dataset == "adult-gender":
        tau = 7.0
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
        tau = 1.0
        mu = 1.2
        X, y, x_control = load_kdd()

    sensitive_attrs = x_control.keys()

    sensitive_attrs_to_cov_thresh = {sensitive_attrs[0]: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0, 1: 0}}}  # zero covariance threshold, means try to get the fairest solution
    cons_params = {"cons_type": cons_type,
                   "tau": tau,
                   "mu": mu,
                   "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}


    iter = 0

    while iter < 20:
        print "iteration = " + str(iter)
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

            # exit(1)
            # cons_params = None # constraint parameters, will use them later
            loss_function = "logreg" # perform the experiments with logistic regression
            EPS = 1e-6

            def train_test_classifier():
                w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)
                # train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)
                roc, disc = fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)
                # accuracy and FPR are for the test because we need of for plotting
                # return w, test_score, s_attr_to_fp_fn_test
                return roc,disc

            try:
                roc, disc= train_test_classifier()
                aucScores.append(roc)
                discScores.append(disc)

                one_line = "@relation compass-weka.filters.unsupervised.attribute.NumericToNominal-Rlast\n\n" + \
                           "@attribute intercept numeric\n" + \
                           "@attribute 'age_cat_25 - 45' numeric\n" + \
                           "@attribute 'age_cat_Greater than 45' numeric\n" + \
                           "@attribute 'age_cat_Less than 25' numeric\n" + \
                           "@attribute race {0,1}\n" + \
                           "@attribute sex {0,1}\n" + \
                           "@attribute priors_count numeric\n" + \
                           "@attribute c_charge_degree numeric\n" + \
                           "@attribute Class-label {-1,1}\n\n" + "@data\n"


                # print "efficiency AUC"
                print "AUC = " + str(roc) + ", dTPR = " +  str(disc[0] ) + ", TNR = " + str(disc[1] )
                np.savetxt('compass_training.arff',np.column_stack((x_train, y_train)), fmt='%d,%d,%d,%d,%d,%d,%1.6f,%d,%d', delimiter=',')
                np.savetxt('compass_testing.arff',np.column_stack((x_test, y_test)), fmt='%d,%d,%d,%d,%d,%d,%1.6f,%d,%d', delimiter=',')

                with open("compass_training.arff", 'r+') as fp:
                    lines = fp.readlines()  # lines is list of line, each element '...\n'
                    lines.insert(0, one_line)  # you can use any index if you know the line index
                    fp.seek(0)  # file pointer locates at the beginning to write the whole file again
                    fp.writelines(lines)
                    fp.close()

                with open("compass_testing.arff", 'r+') as fp:
                    lines = fp.readlines()  # lines is list of line, each element '...\n'
                    lines.insert(0, one_line)  # you can use any index if you know the line index
                    fp.seek(0)  # file pointer locates at the beginning to write the whole file again
                    fp.writelines(lines)
                    fp.close()

                raw_input("dwseee ")

            except Exception, e:
                print e

                iter -=1
                pass

            iter += 1
            break

    # print "dataset = " + dataset
    print "auc = " +str(mean(aucScores)) + ", st.Dev = " + str(stddev(aucScores))
    print "disc = "+str(mean(discScores)) + ", st.Dev = " + str(stddev(discScores))

    # print "\n-----------------------------------------------------------------------------------\n"



def main():
    # test_compas_data()
    # test_compas_data("adult-race")
    # test_compas_data("dutch")
    test_compas_data("adult-gender")
    # test_compas_data("kdd")


if __name__ == '__main__':
    main()




# dataset = compass
# auc = 64.83472648738869, st.Dev = 0.6485625790409174
# disc = -10.9599413508, st.Dev = 2.43341998452
