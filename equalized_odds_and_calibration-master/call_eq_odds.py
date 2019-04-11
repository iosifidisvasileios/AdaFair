import cvxpy as cvx
import numpy as np
from collections import namedtuple

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score


class Model(namedtuple('Model', 'pred label')):
    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def tp_count(self):
        """
        True positive rate
        """
        return np.sum(np.logical_and(self.pred.round() == 1, self.label == 1)) * 1.0

    def fp_count(self):
        """
        False positive rate
        """
        return np.sum(np.logical_and(self.pred.round() == 1, self.label == 0)) * 1.0

    def tn_count(self):
        """
        True negative rate
        """
        return np.sum(np.logical_and(self.pred.round() == 0, self.label == 0)) * 1.0

    def fn_count(self):
        """
        False negative rate
        """
        return np.sum(np.logical_and(self.pred.round() == 0, self.label == 1)) * 1.0

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label


    def calib_eq_odds(self, other, fp_rate, fn_rate, mix_rates=None):
        if fn_rate == 0:
            self_cost = self.fp_cost()
            other_cost = other.fp_cost()
            print(self_cost, other_cost)
            self_trivial_cost = self.trivial().fp_cost()
            other_trivial_cost = other.trivial().fp_cost()
        elif fp_rate == 0:
            self_cost = self.fn_cost()
            other_cost = other.fn_cost()
            self_trivial_cost = self.trivial().fn_cost()
            other_trivial_cost = other.trivial().fn_cost()
        else:
            self_cost = self.weighted_cost(fp_rate, fn_rate)
            other_cost = other.weighted_cost(fp_rate, fn_rate)
            self_trivial_cost = self.trivial().weighted_cost(fp_rate, fn_rate)
            other_trivial_cost = other.trivial().weighted_cost(fp_rate, fn_rate)

        other_costs_more = other_cost > self_cost
        self_mix_rate = (other_cost - self_cost) / (self_trivial_cost - self_cost) if other_costs_more else 0
        other_mix_rate = 0 if other_costs_more else (self_cost - other_cost) / (other_trivial_cost - other_cost)

        # New classifiers
        self_indices = np.random.permutation(len(self.pred))[:int(self_mix_rate * len(self.pred))]
        self_new_pred = self.pred.copy()
        self_new_pred[self_indices] = self.base_rate()
        calib_eq_odds_self = Model(self_new_pred, self.label)

        other_indices = np.random.permutation(len(other.pred))[:int(other_mix_rate * len(other.pred))]
        other_new_pred = other.pred.copy()
        other_new_pred[other_indices] = other.base_rate()
        calib_eq_odds_other = Model(other_new_pred, other.label)

        if mix_rates is None:
            return calib_eq_odds_self, calib_eq_odds_other, (self_mix_rate, other_mix_rate)
        else:
            return calib_eq_odds_self, calib_eq_odds_other

    def trivial(self):
        """
        Given a classifier, produces the trivial classifier
        (i.e. a model that just returns the base rate for every prediction)
        """
        base_rate = self.base_rate()
        pred = np.ones(len(self.pred)) * base_rate
        return Model(pred, self.label)

    def weighted_cost(self, fp_rate, fn_rate):
        """
        Returns the weighted cost
        If fp_rate = 1 and fn_rate = 0, returns self.fp_cost
        If fp_rate = 0 and fn_rate = 1, returns self.fn_cost
        If fp_rate and fn_rate are nonzero, returns fp_rate * self.fp_cost * (1 - self.base_rate) +
            fn_rate * self.fn_cost * self.base_rate
        """
        norm_const = float(fp_rate + fn_rate) if (fp_rate != 0 and fn_rate != 0) else 1
        res = fp_rate / norm_const * self.fp_cost() * (1 - self.base_rate()) + \
            fn_rate / norm_const * self.fn_cost() * self.base_rate()
        return res

    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])

    def results(protected_model, non_protected_model):
        # accuracy = (
        #            protected_model.tn_count() + protected_model.tp_count() + non_protected_model.tn_count() + non_protected_model.tp_count()) / (
        #                protected_model.tn_count() + protected_model.tp_count() + non_protected_model.tn_count() + non_protected_model.tp_count() +
        #                protected_model.fn_count() + protected_model.fp_count() + non_protected_model.fn_count() + non_protected_model.fp_count())
        accuracy = accuracy_score(np.concatenate((protected_model.label, non_protected_model.label), axis=None),np.concatenate((protected_model.pred.round(), non_protected_model.pred.round()), axis=None))
        balanced_acc = balanced_accuracy_score(np.concatenate((protected_model.label, non_protected_model.label), axis=None),np.concatenate((protected_model.pred.round(), non_protected_model.pred.round()), axis=None))

        # balanced_acc = (
        #                    ((protected_model.tn_count() + non_protected_model.tn_count()) / (protected_model.tn_count() + non_protected_model.tn_count() + protected_model.fn_count() + non_protected_model.fn_count())) +
        #                    ((protected_model.tp_count() + non_protected_model.tp_count()) / (protected_model.tp_count() + non_protected_model.tp_count() + protected_model.fp_count() + non_protected_model.fp_count()))
        #                ) * 0.5

        fairness = abs(protected_model.tpr() - non_protected_model.tpr()) + abs(protected_model.tnr() - non_protected_model.tnr())

        return {"accuracy": accuracy, "balanced_accuracy": balanced_acc, "fairness": fairness,
                "TNR_protected": protected_model.tnr(), "TPR_protected": protected_model.tpr(),
                "TPR_non_protected": non_protected_model.tpr(), "TNR_non_protected": non_protected_model.tnr()}
