import matplotlib
import numpy

from AdaFair import AdaFair
from AdaFairEQOP import AdaFairEQOP
from my_useful_functions import plot_costs_per_round_eqop, plot_costs_per_round

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, 'DataPreprocessing')

from Competitors.AdaCost import AdaCostClassifier
from load_kdd import load_kdd
# from load_german import load_german
from load_compas_data import load_compas
from load_adult import load_adult
from load_bank import load_bank


def run_eval(dataset):
    if dataset == "compass-gender":
        X, y, sa_index, p_Group, x_control = load_compas("sex")
    elif dataset == "compass-race":
        X, y, sa_index, p_Group, x_control = load_compas("race")
    elif dataset == "adult-gender":
        X, y, sa_index, p_Group, x_control = load_adult("sex")
    elif dataset == "bank":
        X, y, sa_index, p_Group, x_control = load_bank()
    elif dataset == "kdd":
        X, y, sa_index, p_Group, x_control = load_kdd()
    else:
        exit(1)

    base_learners = 200
    no_cumul = train_classifier(X, y, sa_index, p_Group, 0, base_learners)
    cumul = train_classifier(X, y, sa_index, p_Group, 1, base_learners)

    plot_costs_per_round("Images/Costs/" + dataset, no_cumul, cumul)


def train_classifier(X_train, y_train, sa_index, p_Group, mode, base_learners):
    if mode == 0:
        classifier = AdaFair(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB='CSB1', cumul=False)
    elif mode == 1:
        classifier = AdaFair(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB='CSB1', cumul=True)

    classifier.fit(X_train, y_train)
    return classifier.costs


def main(dataset):
    run_eval(dataset)


if __name__ == '__main__':
    main("compass-gender")
    # main("adult-gender")
    # main("bank")
    # main("kdd")
