import matplotlib
matplotlib.use("Pdf")
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import numpy
import matplotlib.pyplot as plt

def calculate_fairness(data, labels, predictions, probs, saIndex, saValue):
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

    output["average_precision"] = average_precision_score(labels, predictions)
    output["accuracy"] = accuracy_score(labels, predictions)
    output["auc"] = roc_auc_score(labels, probs)
    output["dTPR"] = tpr_non_protected - tpr_protected
    output["dTNR"] = tnr_non_protected - tnr_protected
    output["TPR_protected"] = tpr_protected
    output["TPR_non_protected"] = tpr_non_protected
    output["TNR_protected"] = tnr_protected
    output["TNR_non_protected"] = tnr_non_protected
    return output

def plot_results(init, max_cost, step ,summary_performance, summary_weights, output_dir,title, plot_weights=True):
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
        average_precision= 0
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

    plt.figure()
    plt.grid(True)

    plt.rcParams["figure.figsize"] = (14, 12)

    plt.plot(step_list, accuracy_list, '-b', label='accuracy')
    plt.plot(step_list, auc_list, '-r', label='auc')

    plt.plot(step_list, average_precision_list, '-*', label='average precision')

    plt.plot(step_list, d_tpr_list, '--', label='dTPR')
    plt.plot(step_list, d_tnr_list, ':', label='dTNR')

    plt.plot(step_list, tpr_protected_list, '-o', label='TPR protected')
    plt.plot(step_list, tpr_non_protected_list, '-v', label='TPR non-protected')

    plt.plot(step_list, tnr_protected_list, '-.', label='TNR protected')
    plt.plot(step_list, tnr_non_protected_list, '-+', label='TNR non-protected')
    plt.legend(loc='center left', bbox_to_anchor=(0.93, 0.5))


    plt.xlabel('Cost Increasement (%)')
    plt.ylabel('(%)')
    plt.title(title + " performance")

    plt.savefig(output_dir + "_performance.png")
    # plt.show()

    if not plot_weights:
        return

    plt.figure()
    plt.grid(True)

    plt.rcParams["figure.figsize"] = (14, 12)

    plt.plot(step_list, W_pos_list, '-b', label='Positives')
    plt.plot(step_list, W_neg_list, '-r', label='Negatives')

    plt.plot(step_list, W_dp_list, '--', label='Protected Positives')
    plt.plot(step_list, W_fp_list, ':', label='Non-Protected Positives')

    plt.plot(step_list, W_dn_list, '-o', label='Protected Negatives')
    plt.plot(step_list, W_fn_list, '-+', label='Non-Protected Negatives')
    plt.legend(loc='center left', bbox_to_anchor=(0.93, 0.5))



    # plt.legend(loc='best')

    plt.xlabel('Cost Increasement (%)')
    plt.ylabel('(%)')
    plt.title(title + " weights")

    plt.savefig(output_dir + "_weights.png")
    # plt.show()
