import warnings
import numpy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src import AdaFairEQOP, AdaFairSP

warnings.filterwarnings("ignore")
import copy
import random
from multiprocessing import Process, Lock
import pickle
import os
import matplotlib
from sklearn.model_selection import StratifiedShuffleSplit

matplotlib.use('Agg')
import sys

from src.AdaFair import AdaFair
sys.path.insert(0, 'DataPreprocessing')
import time
from DataPreprocessing.load_adult import load_adult

def calculate_performance(data, labels, predictions, probs, saIndex, saValue):
    protected_pos = 0.
    protected_neg = 0.
    non_protected_pos = 0.
    non_protected_neg = 0.

    tp_protected = 0.
    tn_protected = 0.
    fp_protected = 0.
    fn_protected = 0.

    tp_non_protected = 0.
    tn_non_protected = 0.
    fp_non_protected = 0.
    fn_non_protected = 0.
    for idx, val in enumerate(data):
        # protrcted population
        if val[saIndex] == saValue:
            if predictions[idx] == 1:
                protected_pos += 1.
            else:
                protected_neg += 1.
            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1.
                else:
                    tn_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1.
                else:
                    fp_protected += 1.
        else:
            if predictions[idx] == 1:
                non_protected_pos += 1.
            else:
                non_protected_neg += 1.

            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1.
                else:
                    tn_non_protected += 1.
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1.
                else:
                    fp_non_protected += 1.

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)

    C_prot = (protected_pos) / (protected_pos + protected_neg)
    C_non_prot = (non_protected_pos) / (non_protected_pos + non_protected_neg)

    stat_par = C_non_prot - C_prot

    output = dict()

    # output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    output["balanced_accuracy"] =( (tp_protected + tp_non_protected)/(tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
                                   (tn_protected + tn_non_protected) / (tn_protected + tn_non_protected + fp_protected + fp_non_protected))*0.5

    output["accuracy"] = accuracy_score(labels, predictions)
    # output["dTPR"] = tpr_non_protected - tpr_protected
    # output["dTNR"] = tnr_non_protected - tnr_protected
    output["fairness"] = abs(tpr_non_protected - tpr_protected) + abs(tnr_non_protected - tnr_protected)
    # output["fairness"] = abs(stat_par)

    output["TPR_protected"] = tpr_protected
    output["TPR_non_protected"] = tpr_non_protected
    output["TNR_protected"] = tnr_protected
    output["TNR_non_protected"] = tnr_non_protected
    return output


def plot_my_results(results, names, output_dir, dataset):
    accuracy_list = []
    balanced_accuracy_list = []
    fairness_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    std_accuracy_list = []
    std_balanced_accuracy_list = []
    std_fairness_list = []
    std_tpr_protected_list = []
    std_tpr_non_protected_list = []
    std_tnr_protected_list = []
    std_tnr_non_protected_list = []

    for list_of_results in results:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in list_of_results:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        numpy.mean(accuracy)
        numpy.std(accuracy)

        accuracy_list.append(numpy.mean(accuracy))
        balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        fairness_list.append(numpy.mean(fairness))
        tpr_protected_list.append(numpy.mean(tpr_protected))
        tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        tnr_protected_list.append(numpy.mean(tnr_protected))
        tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        std_accuracy_list.append(numpy.std(accuracy))
        std_balanced_accuracy_list.append(numpy.std(balanced_accuracy))
        std_fairness_list.append(numpy.std(fairness))
        std_tpr_protected_list.append(numpy.std(tpr_protected))
        std_tpr_non_protected_list.append(numpy.std(tpr_non_protected))
        std_tnr_protected_list.append(numpy.std(tnr_protected))
        std_tnr_non_protected_list.append(numpy.std(tnr_non_protected))

    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0, 1])
    plt.yticks(numpy.arange(0, 1, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175

    plt.xticks(index + 1.5 * bar_width,
               ('Accuracy', 'Balanced Accuracy', 'Disp. Mis.', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.',
                'TNR Non-Prot.'))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'dimgray']
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i], edgecolor='black')

    plt.legend(loc='best', ncol=1, shadow=False)
    plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + "_performance.png", bbox_inches='tight', dpi=200)
    print(names)
    print("accuracy_list= " + str(accuracy_list))
    print("std_accuracy_list = " + str(std_accuracy_list))
    print("balanced_accuracy_list=  " + str(balanced_accuracy_list))
    print("std_balanced_accuracy_list = " + str(std_balanced_accuracy_list))

    print("fairness_list=  " + str(fairness_list))
    print("std_fairness_list = " + str(std_fairness_list))

    print("tpr_protected_list = " + str(tpr_protected_list))
    print("std_tpr_protected_list = " + str(std_tpr_protected_list))

    print("tpr_non_protected_list = " + str(tpr_non_protected_list))
    print("std_tpr_non_protected_list = " + str(std_tpr_non_protected_list))

    print("tnr_protected_list = " + str(tnr_protected_list))
    print("std_tnr_protected_list = " + str(std_tnr_protected_list))

    print("tnr_non_protected_list = " + str(tnr_non_protected_list))
    print("std_tnr_non_protected_list = " + str(std_tnr_non_protected_list))


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


def run_eval(dataset, iterations):
    suffixes = ['AdaFair CSB2', 'AdaFair CSB1' ]

    if dataset == "adult-gender":
        X, y, sa_index, p_Group, x_control = load_adult("sex")
    elif dataset == "adult-race":
        X, y, sa_index, p_Group, x_control = load_adult("race")
    else:
        exit(1)
    create_temp_files(dataset, suffixes)

    threads = []
    mutex = []
    for lock in range(0, 8):
        mutex.append(Lock())

    random.seed(int(time.time()))

    for iter in range(0, iterations):

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
        for train_index, test_index in sss.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for proc in range(0, 2):

                threads.append(Process(target=train_classifier, args=( copy.deepcopy(X_train),
                                                                           X_test, copy.deepcopy(y_train),
                                                                           y_test, sa_index, p_Group,
                                                                           dataset + suffixes[proc],
                                                                           mutex[proc],proc, 20, 1)))

            break

    for process in threads:
        process.start()

    for process in threads:
        process.join()

    threads = []

    results = []
    for suffix in suffixes:
        infile = open(dataset + suffix, 'rb')
        temp_buffer = pickle.load(infile)
        results.append(temp_buffer.performance)
        infile.close()

    plot_my_results(results, suffixes, "Images/" + dataset, dataset)
    delete_temp_files(dataset, suffixes)

def train_classifier(X_train, X_test, y_train, y_test, sa_index, p_Group, dataset, mutex, mode, base_learners, c):
    if mode == 0:
        classifier = AdaFair(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB="CSB2", c=c)
    elif mode == 1:
        classifier = AdaFair(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group, CSB="CSB1", c=c)
    classifier.fit(X_train, y_train)

    y_pred_probs = classifier.predict_proba(X_test)[:, 1]
    y_pred_labels = classifier.predict(X_test)

    mutex.acquire()

    infile = open(os.path.join(os.getcwd(), 'AdaFair', dataset), 'rb')
    dict_to_ram = pickle.load(infile)
    infile.close()
    dict_to_ram.performance.append(
        calculate_performance(X_test, y_test, y_pred_labels, y_pred_probs, sa_index, p_Group))
    outfile = open(os.path.join(os.getcwd(), 'AdaFair', dataset), 'wb')
    pickle.dump(dict_to_ram, outfile)
    outfile.close()
    mutex.release()


if __name__ == '__main__':
    run_eval("adult-gender", 2)
