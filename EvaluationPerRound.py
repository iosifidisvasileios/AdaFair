import warnings
import numpy
warnings.filterwarnings("ignore")
import random
import matplotlib
from sklearn.model_selection import ShuffleSplit, train_test_split, StratifiedShuffleSplit

matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from AdaFair import AdaFair
sys.path.insert(0, 'DataPreprocessing')

import time
from load_compas_data import load_compas
from load_adult import load_adult
from load_kdd import load_kdd

from load_bank import load_bank



def run_eval(dataset):

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

    random.seed(1)
    rounds = 500
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33 )
    classifier = AdaFair(n_estimators=rounds, saIndex=sa_index, saValue=p_Group, CSB="CSB1",  use_validation=True, debug=True,X_test=X_test, y_test=y_test)
    classifier.fit(X_train,y_train)

    plot_per_round(rounds, classifier.performance, classifier.objective, classifier.theta,'Images/' + dataset + '_per_round_analysis.png')


def plot_per_round(rounds, results, objective, theta, output_dir ):
    train_error_list = []
    train_bal_error_list = []
    train_fairness = []

    test_error_list = []
    test_bal_error_list = []
    test_fairness = []
    objective_list = []
    for i in numpy.arange(1, rounds):

        line = results[i]
        objective_list.append(objective[i])

        train_bal_error_list.append(float(line[0]))
        train_error_list.append(float(line[1]))
        train_fairness.append(float(line[2]))
        test_bal_error_list.append(float(line[3]))
        test_error_list.append(float(line[4]))
        test_fairness.append(float(line[5]))

    step_list = [i for i in range(1, rounds)]

    plt.figure(figsize=(20, 6))
    plt.rcParams.update({'font.size': 16})
    plt.grid(True)

    plt.xlim([0,500])
    # plt.ylim([0,1000])
    # plt.plot(step_list, train_error_list, '--', label='Train Error rate')
    # plt.plot(step_list, test_error_list, ':', label='Test Error rate')

    plt.plot(step_list, train_bal_error_list, '-.', label='Val. Bal.Error Rate',linewidth=2)
    plt.plot(step_list, test_bal_error_list, '-', label='Test Bal.Error Rate',linewidth=2)

    plt.plot(step_list, train_fairness, ':', label='Val. Fairness',linewidth=2)
    plt.plot(step_list, test_fairness, '--', label='Test Fairness',linewidth=2)

    plt.plot(step_list, objective_list, '-x', label='Val. Objective', markersize=3.5)
    plt.axvline(x=theta, color='k', linestyle='--', label='Post-Processing', linewidth=3)
    plt.xlabel('Rounds')
    plt.legend(loc='upper center', bbox_to_anchor=(0.48, 1.075), ncol=6, shadow=False,fancybox=True, framealpha=1.0)

    plt.savefig(output_dir, bbox_inches='tight', dpi=200)




if __name__ == '__main__':
    # run_eval("compass-gender")
    # run_eval("adult-gender")
    # run_eval("bank")
    run_eval("kdd")
