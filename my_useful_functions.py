# encoding: utf-8

import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import accuracy_score
import numpy
import matplotlib.pyplot as plt


def calculate_performance(data, labels, predictions, probs, saIndex, saValue):
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
            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_protected += 1
                else:
                    tn_protected += 1
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_protected += 1
                else:
                    fp_protected += 1

        else:
            # correctly classified
            if labels[idx] == predictions[idx]:
                if labels[idx] == 1:
                    tp_non_protected += 1
                else:
                    tn_non_protected += 1
            # misclassified
            else:
                if labels[idx] == 1:
                    fn_non_protected += 1
                else:
                    fp_non_protected += 1

    tpr_protected = tp_protected / (tp_protected + fn_protected)
    tnr_protected = tn_protected / (tn_protected + fp_protected)

    tpr_non_protected = tp_non_protected / (tp_non_protected + fn_non_protected)
    tnr_non_protected = tn_non_protected / (tn_non_protected + fp_non_protected)
    # print "accuracy = " + str(accuracy_score(labels, predictions)) + \
    #       ",auc = " + str(roc_auc_score(labels, probs)) + \
    #       ", dTPR = " + str((tpr_non_protected - tpr_protected) * 100) + \
    #       ", dTNR = " + str((tnr_non_protected - tnr_protected) * 100) + \
    #       ", TPR_protected = " + str(tpr_protected) + \
    #       ", TPR_non_protected = " + str(tpr_non_protected) + \
    #       ", TNR_protected = " + str(tnr_protected) + \
    #       ", TNR_non_protected = " + str(tnr_non_protected)

    output = dict()

    # output["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
    output["balanced_accuracy"] =( (tp_protected + tp_non_protected)/(tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
                                   (tn_protected + tn_non_protected) / (tn_protected + tn_non_protected + fp_protected + fp_non_protected))*0.5

    output["accuracy"] = accuracy_score(labels, predictions)
    # output["dTPR"] = tpr_non_protected - tpr_protected
    # output["dTNR"] = tnr_non_protected - tnr_protected
    output["fairness"] = abs(tpr_non_protected - tpr_protected) + abs(tnr_non_protected - tnr_protected)

    output["TPR_protected"] = tpr_protected
    output["TPR_non_protected"] = tpr_non_protected
    output["TNR_protected"] = tnr_protected
    output["TNR_non_protected"] = tnr_non_protected
    return output


def plot_results_of_c_impact(csb1, csb2, steps, output_dir, dataset):
    csb1_accuracy_list = []
    csb1_balanced_accuracy_list = []
    csb1_fairness_list = []
    csb1_tpr_protected_list = []
    csb1_tpr_non_protected_list = []
    csb1_tnr_protected_list = []
    csb1_tnr_non_protected_list = []

    csb2_accuracy_list = []
    csb2_balanced_accuracy_list = []
    csb2_fairness_list = []
    csb2_tpr_protected_list = []
    csb2_tpr_non_protected_list = []
    csb2_tnr_protected_list = []
    csb2_tnr_non_protected_list = []

    for c in steps:

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in csb1[c]:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        csb1_accuracy_list.append(numpy.mean(accuracy))
        csb1_balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        csb1_fairness_list.append(numpy.mean(fairness))
        # csb1_tpr_protected_list.append(numpy.mean(tpr_protected))
        # csb1_tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        # csb1_tnr_protected_list.append(numpy.mean(tnr_protected))
        # csb1_tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

        accuracy = []
        balanced_accuracy = []
        fairness = []
        tpr_protected = []
        tpr_non_protected = []
        tnr_protected = []
        tnr_non_protected = []

        for item in csb2[c]:
            accuracy.append(item["accuracy"])
            balanced_accuracy.append(item["balanced_accuracy"])
            fairness.append(item["fairness"])
            tpr_protected.append(item["TPR_protected"])
            tpr_non_protected.append(item["TPR_non_protected"])
            tnr_protected.append(item["TNR_protected"])
            tnr_non_protected.append(item["TNR_non_protected"])

        csb2_accuracy_list.append(numpy.mean(accuracy))
        csb2_balanced_accuracy_list.append(numpy.mean(balanced_accuracy))
        csb2_fairness_list.append(numpy.mean(fairness))
        csb2_tpr_protected_list.append(numpy.mean(tpr_protected))
        csb2_tpr_non_protected_list.append(numpy.mean(tpr_non_protected))
        csb2_tnr_protected_list.append(numpy.mean(tnr_protected))
        csb2_tnr_non_protected_list.append(numpy.mean(tnr_non_protected))

    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.rcParams.update({'font.size': 13})

    # plt.plot(steps, csb1_accuracy_list, '-.', label='AFB CSB1 Accuracy')
    plt.plot(steps, csb2_accuracy_list, '--', label='Accuracy')

    # plt.plot(steps, csb1_balanced_accuracy_list, '-x', label='AFB CSB1 B.Accuracy')
    plt.plot(steps, csb2_balanced_accuracy_list, '-*', label='Balanced Accuracy')

    # plt.plot(steps, csb1_fairness_list, '-o', label='AFB CSB1 E.O.')
    plt.plot(steps, csb2_fairness_list, '-v', label='Equalized Odds')
    plt.plot(steps, csb2_tpr_protected_list, '-x', label='TPR Prot.')
    plt.plot(steps, csb2_tpr_non_protected_list, '-o', label='TPR Non-Prot.')
    plt.plot(steps, csb2_tnr_protected_list, '-^', label='TNR Prot.')
    plt.plot(steps, csb2_tnr_non_protected_list, '-<', label='TNR Non-Prot.')

    plt.legend(loc='center', bbox_to_anchor=(0.7, 0.3), shadow=False,ncol=1, framealpha=.30)

    plt.xlabel('c')
    # plt.ylabel('(%)')
    # plt.title("Impact of c for " + dataset + " dataset")

    print "csb2_accuracy_list " + str(csb2_accuracy_list)
    print "csb2_balanced_accuracy_list " + str(csb2_balanced_accuracy_list)
    print "csb2_fairness_list " + str(csb2_fairness_list)
    print "csb2_tpr_protected_list " + str(csb2_tpr_protected_list)
    print "csb2_tpr_non_protected_list " + str(csb2_tpr_non_protected_list)
    print "csb2_tnr_protected_list " + str(csb2_tnr_protected_list)
    print "csb2_tnr_non_protected_list " + str(csb2_tnr_non_protected_list)

    plt.savefig(output_dir + dataset + "_c_impact.png", bbox_inches='tight', dpi=200)


def plot_costs_per_round(output, noaccum, adafair):
    noaccum_tpr = []
    noaccum_tnr = []
    accum_tpr = []
    accum_tnr = []
    for i in noaccum:
        noaccum_tpr.append(float(i.split(",")[0]))
        noaccum_tnr.append(float(i.split(",")[1]))
    for i in adafair:
        accum_tpr.append(float(i.split(",")[0]))
        accum_tnr.append(float(i.split(",")[1]))

    steps = numpy.arange(0, len(accum_tnr), step=1)
    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.rcParams.update({'font.size': 10.5})

    plt.plot(steps, noaccum_tpr, '-.', label='AdaFair NoCumul ' + r'$\delta$FNR')
    plt.plot(steps, noaccum_tnr, ':', label='AdaFair NoCumul ' + r'$\delta$FPR')
    plt.plot(steps, accum_tpr, '-', label='AdaFair ' + r'$\delta$FNR')
    plt.plot(steps, accum_tnr, '--', label='AdaFair ' + r'$\delta$FPR')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.105), ncol=2, shadow=False,fancybox=True, framealpha=1.0)

    # plt.legend(loc='best')

    plt.xlabel('Round')

    plt.savefig(output+ "_costs.png", bbox_inches='tight', dpi=200,shadow=False,fancybox=True, framealpha=.30)


    print accum_tnr
    pass

def plot_results(init, max_cost, step, summary_performance, summary_weights, output_dir, title, plot_weights=True):
    step_list = []
    accuracy_list = []
    auc_list = []
    average_precision_list = []
    d_tpr_list = []
    d_tnr_list = []
    tpr_protected_list = []
    tpr_non_protected_list = []
    tnr_protected_list = []
    tnr_non_protected_list = []
    #
    W_pos_list = []
    W_neg_list = []
    W_dp_list = []
    W_dn_list = []
    W_fp_list = []
    W_fn_list = []

    for i in numpy.arange(init, max_cost + step, step):

        iteration_performance_summary = summary_performance[i]
        iteration_weights_summary = summary_weights[i]
        W_pos = 0.
        W_neg = 0.
        W_dp = 0.
        W_dn = 0.
        W_fp = 0.
        W_fn = 0.

        accuracy = 0
        auc = 0
        average_precision = 0
        d_tpr = 0
        d_tnr = 0
        tpr_protected = 0
        tpr_non_protected = 0
        tnr_protected = 0
        tnr_non_protected = 0

        for item in iteration_weights_summary:
            W_pos += item[0] / len(iteration_weights_summary)
            W_neg += item[1] / len(iteration_weights_summary)
            W_dp += item[2] / len(iteration_weights_summary)
            W_fp += item[3] / len(iteration_weights_summary)
            W_dn += item[4] / len(iteration_weights_summary)
            W_fn += item[5] / len(iteration_weights_summary)

        W_pos_list.append(W_pos)
        W_neg_list.append(W_neg)
        W_dp_list.append(W_dp)
        W_fp_list.append(W_fp)
        W_dn_list.append(W_dn)
        W_fn_list.append(W_fn)

        for item in iteration_performance_summary:
            accuracy += item["accuracy"] / len(iteration_performance_summary)
            auc += item["auc"] / len(iteration_performance_summary)
            d_tpr += item["dTPR"] / len(iteration_performance_summary)
            d_tnr += item["dTNR"] / len(iteration_performance_summary)
            tpr_protected += item["TPR_protected"] / len(iteration_performance_summary)
            tpr_non_protected += item["TPR_non_protected"] / len(iteration_performance_summary)
            tnr_protected += item["TNR_protected"] / len(iteration_performance_summary)
            tnr_non_protected += item["TNR_non_protected"] / len(iteration_performance_summary)
            average_precision += item["average_precision"] / len(iteration_performance_summary)

        step_list.append(i)
        accuracy_list.append(accuracy)
        auc_list.append(auc)
        average_precision_list.append(average_precision)
        d_tpr_list.append(d_tpr)
        d_tnr_list.append(d_tnr)
        tpr_protected_list.append(tpr_protected)
        tpr_non_protected_list.append(tpr_non_protected)
        tnr_protected_list.append(tnr_protected)
        tnr_non_protected_list.append(tnr_non_protected)

    plt.figure(figsize=(20, 20))
    plt.grid(True)

    plt.plot(step_list, accuracy_list, '-b', label='accuracy')
    plt.plot(step_list, auc_list, '-r', label='auc')

    plt.plot(step_list, average_precision_list, '-*', label='average precision')

    plt.plot(step_list, d_tpr_list, '--', label='dTPR')
    plt.plot(step_list, d_tnr_list, ':', label='dTNR')

    plt.plot(step_list, tpr_protected_list, '-o', label='TPR Prot.')
    plt.plot(step_list, tpr_non_protected_list, '-v', label='TPR non-Prot.')

    plt.plot(step_list, tnr_protected_list, '-.', label='TNR Prot.')
    plt.plot(step_list, tnr_non_protected_list, '-+', label='TNR non-Prot.')
    plt.legend(loc='best')

    plt.xlabel('Cost Increasement (%)')
    plt.ylabel('(%)')
    plt.title(title + " performance")

    plt.savefig(output_dir + "_performance.png")

    if not plot_weights:
        return

    plt.figure(figsize=(20, 20))
    plt.grid(True)

    plt.plot(step_list, W_pos_list, '-b', label='Positives')
    plt.plot(step_list, W_neg_list, '-r', label='Negatives')

    plt.plot(step_list, W_dp_list, '--', label='Prot. Positives')
    plt.plot(step_list, W_fp_list, ':', label='Non-Prot. Positives')

    plt.plot(step_list, W_dn_list, '-o', label='Prot. Negatives')
    plt.plot(step_list, W_fn_list, '-+', label='Non-Prot. Negatives')
    plt.legend(loc='best')

    # plt.legend(loc='best')

    plt.xlabel('Cost Increasement (%)')
    plt.ylabel('(%)')
    plt.title(title + " weights")

    plt.savefig(output_dir + "_weights.png")
    # plt.show()


def plot_per_round(rounds, results, objective, output_dir ):
    train_error_list = []
    train_bal_error_list = []
    train_fairness = []

    test_error_list = []
    test_bal_error_list = []
    test_fairness = []
    objective_list = []
    for i in numpy.arange(0, rounds):

        line = results[i].split(",")

        objective_list.append(objective[i])
        train_error_list.append(float(line[1]))
        train_bal_error_list.append(float(line[2]))
        train_fairness.append(float(line[3]))
        test_error_list.append(float(line[4]))
        test_bal_error_list.append(float(line[5]))
        test_fairness.append(float(line[6]))

    step_list = [i for i in range(0, rounds)]

    print train_fairness
    print train_error_list
    print train_bal_error_list
    print step_list
    plt.figure()
    plt.grid(True)

    plt.plot(step_list, train_error_list, '--', label='Train Error rate')
    plt.plot(step_list, test_error_list, ':', label='Test Error rate')

    plt.plot(step_list, train_bal_error_list, '-.', label='Train Bal.Error rate')
    plt.plot(step_list, test_bal_error_list, '-', label='Test Bal.Error rate')

    plt.plot(step_list, train_fairness, '-x', label='Train E.O.', markersize=3.25)
    plt.plot(step_list, objective_list, '-<', label='Objective', markersize=3.25)
    plt.plot(step_list, test_fairness, '-o', label='Test E.O.', markersize=3.25)

    # plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 10})
    # plt.ylim([0,1])
    # plt.yticks(numpy.arange(0, 1.00001, step=0.05))

    plt.xlabel('Rounds')
    plt.ylabel('(%)')
    # plt.legend(loc='best',ncol=1, shadow=False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=3, shadow=False,fancybox=True, framealpha=1.0)

    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir,bbox_inches='tight', dpi=200)



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

    # plt.figure(figsize=(11, 9))
    # plt.rcParams.update({'font.size': 14})
    # plt.ylim([0,1])
    # plt.yticks(numpy.arange(0, 1, step=0.05))
    #
    # plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')
    #
    # plt.grid(True, axis='y')
    # index = numpy.arange(0, 8, step=1.3)
    # # index = numpy.arange(7)
    # bar_width = 0.175
    #
    #
    #
    # plt.xticks(index + 1.5*bar_width ,
    #            ('Accuracy', 'Balanced Accuracy', 'Equal.Odds', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))
    #
    # colors = ['b','g','r','c','m','y','k', 'dimgray']
    # for i in range(0, len(names)):
    #     plt.bar(index + bar_width * i,
    #             [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
    #              tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
    #             yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
    #                   std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
    #                   std_tnr_non_protected_list[i]],
    #             label=names[i], color=colors[i],edgecolor='black')
    #
    # plt.legend(loc='best',ncol=2, shadow=False)
    # plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    # plt.savefig(output_dir + "_performance.png",bbox_inches='tight', dpi=400)
    #
    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 14})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.175



    plt.xticks(index + 1.5*bar_width ,
               ('Accuracy', 'Balanced Accuracy', 'Equalized Odds', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    colors = ['b','g','r','c','m','y','k', 'dimgray']
    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=1, shadow=False)
    plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir + "_performance.png",bbox_inches='tight', dpi=200)
    print names
    print "accuracy_lis t= " + str(accuracy_list)
    print "std_accuracy_list = " + str(std_accuracy_list)
    print "balanced_accuracy_list=  " + str(balanced_accuracy_list)
    print "std_balanced_accuracy_list = " + str(std_balanced_accuracy_list)

    print "fairness_list=  " + str(fairness_list)
    print "std_fairness_list = " + str(std_fairness_list)

    print "tpr_protected_list = " + str(tpr_protected_list)
    print "std_tpr_protected_list = " + str(std_tpr_protected_list)

    print "tpr_non_protected_list = " + str(tpr_non_protected_list)
    print "std_tpr_non_protected_list = " + str(std_tpr_non_protected_list)

    print "tnr_protected_list = " + str(tnr_protected_list)
    print "std_tnr_protected_list = " + str(std_tnr_protected_list)

    print "tnr_non_protected_list = " + str(tnr_non_protected_list)
    print "std_tnr_non_protected_list = " + str(std_tnr_non_protected_list)



def plot_calibration_curves(results, names, init_cost, max_cost, step, directory):
    for num in range(init_cost, max_cost + step, step):
        plt.figure(figsize=(10, 10))
        # ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        # ax2 = plt.subplot2grid((3, 1), (2, 0))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        for idx, row in enumerate(results):
            plt.plot(map(mean, zip(*row.mean_predicted_value[num])), map(mean, zip(*row.fraction_of_positives[num])),
                     "s-", label="%s" % (names[idx][1:],))

        plt.ylabel("Fraction of positives")
        plt.legend(loc="best")
        plt.title('Calibration plots  (reliability curve) for cost = ' + str(num))
        plt.savefig(directory + "calibration_cost_" + str(num) + ".png")
        plt.show()


def mean(a):
    return sum(a) / len(a)
