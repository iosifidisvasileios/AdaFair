"""Weight Boosting

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- ``AdaCostClassifier`` implements adaptive boosting (AdaBoost-SAMME) for
  classification problems.

- ``AdaBoostRegressor`` implements adaptive boosting (AdaBoost.R2) for
  regression problems.
"""

# Authors: Noel Dawe <noel@dawe.me>
#          Gilles Louppe <g.louppe@gmail.com>
#          Hamzeh Alsalhi <ha258@cornell.edu>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import is_classifier, ClassifierMixin, is_regressor
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble.forest import BaseForest
from sklearn.externals import six
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.tree.tree import BaseDecisionTree, DTYPE, DecisionTreeClassifier
from sklearn.utils.validation import has_fit_parameter, check_is_fitted, check_array, check_X_y, check_random_state

__all__ = [
    'SMOTEBoost'
]


from collections import Counter

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
#from sklearn.utils import shuffle


class SMOTE(object):
    """Implementation of Synthetic Minority Over-Sampling Technique (SMOTE).

    SMOTE performs oversampling of the minority class by picking target
    minority class samples and their nearest minority class neighbors and
    generating new samples that linearly combine features of each target
    sample with features of its selected minority class neighbors [1].

    Parameters
    ----------
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE:
           Synthetic Minority Over-Sampling Technique." Journal of Artificial
           Intelligence Research (JAIR), 2002.
    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.

        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.

        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1),
                                       return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

    def fit(self, X):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self


class BaseWeightBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=None):

        super(BaseWeightBoosting, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.W_pos = 0.
        self.W_neg = 0.
        self.W_dp = 0.
        self.W_fp = 0.
        self.W_dn = 0.
        self.W_fn = 0.
        self.performance = []
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        self.weight_list = []
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype, y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_alphas_ = np.zeros(self.n_estimators, dtype=np.float64)

        if self.debug:
            self.conf_scores = []

        random_state = check_random_state(self.random_state)
        # if self.debug:
        #     print  "iteration, alpha , positives , negatives , dp , fp , dn , fn"

        old_weights_sum = np.sum(sample_weight)
        pos, neg, dp, fp, dn, fn = self.calculate_weights(X, y, sample_weight)

        if self.debug:
            self.weight_list.append(
                'init' + "," + str(0) + "," + str(pos) + ", " + str(neg) + ", " + str(dp) + ", " + str(
                    fp) + ", " + str(dn) + ", " + str(fn))

        stats_c_ = Counter(y)
        maj_c_ = max(stats_c_, key=stats_c_.get)
        min_c_ = min(stats_c_, key=stats_c_.get)
        self.minority_target = min_c_

        # print "training error, training balanced accuracy, training EQ.Odds, testing error, testing balanced accuracy, testing EQ.Odds"
        for iboost in range(self.n_estimators):
            # SMOTE step.
            X_min = X[np.where(y == self.minority_target)]
            self.smote.fit(X_min)
            X_syn = self.smote.sample(self.n_samples)
            y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                            dtype=np.int64)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the original and synthetic samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))
            # Boosting step
            sample_weight, alpha, error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_alphas_[iboost] = alpha

            # Stop if error is zero
            if error == 0.5:
                break

            new_sample_weight = np.sum(sample_weight)
            multiplier = old_weights_sum/new_sample_weight

            # Stop if the sum of sample weights has become non-positive
            if new_sample_weight <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight *= multiplier

            pos, neg, dp,fp,dn,fn = self.calculate_weights(X, y, sample_weight)

            if self.debug:
                self.weight_list.append(str(iboost) + "," + str(alpha) + "," + str(pos) + ", " + str(neg) + ", " + str(dp) + ", " + str(fp) + ", " + str(dn) + ", " + str(fn))

            self.W_pos += pos/self.n_estimators
            self.W_neg += neg/self.n_estimators
            self.W_dp += dp/self.n_estimators
            self.W_fp += fp/self.n_estimators
            self.W_dn += dn/self.n_estimators
            self.W_fn += fn/self.n_estimators

            old_weights_sum = np.sum(sample_weight)

        if self.debug:
            self.get_confidence_scores(X)

        return self

    def get_weights(self,):
        return [self.W_pos, self.W_neg, self.W_dp, self.W_fp, self.W_dn, self.W_fn]

    def get_confidence_scores(self, X):
        self.conf_scores = self.decision_function(X)


    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        random_state : numpy.RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like, shape = [n_samples]
            Labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        z : float
        """
        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_alphas_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                        in zip(self.estimator_alphas_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            X = check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X

    def calculate_weights(self, data, labels, sample_weight):

        protected_positive = 0.
        non_protected_positive = 0.

        protected_negative = 0.
        non_protected_negative = 0.

        for idx, val in enumerate(data):
            # protrcted population
            if val[self.saIndex] == self.saValue:
                # protected group
                if labels[idx] == 1:
                    protected_positive += sample_weight[idx]#/len(sample_weight)
                else:
                    protected_negative += sample_weight[idx]#/len(sample_weight)
            else:
                # correctly classified
                if labels[idx] == 1:
                    non_protected_positive += sample_weight[idx]#/len(sample_weight)
                else:
                    non_protected_negative += sample_weight[idx]#/len(sample_weight)

        return [protected_positive + non_protected_positive,
                protected_negative + non_protected_negative,
                protected_positive,
                non_protected_positive,
                protected_negative,
                non_protected_negative]


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
    log_proba = np.log(proba)

    return (n_classes - 1) * (log_proba - (1. / n_classes)
                              * log_proba.sum(axis=1)[:, np.newaxis])


class SMOTEBoost(BaseWeightBoosting, ClassifierMixin):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME',
                 random_state=None,
                 saIndex=None,saValue=None,
                 costs = [1, 1], useFairVote=False,updateAll=False, debug=False, CSB="CSB1",
                 X_test=None, y_test=None,
                 n_samples=100,
                 k_neighbors=5,):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.smote = SMOTE(k_neighbors=k_neighbors,
                           random_state=random_state)

        super(SMOTEBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)


        self.cost_positive = costs[0]
        self.cost_negative = costs[1]

        self.saIndex = saIndex
        self.saValue = saValue
        self.algorithm = algorithm
        self.updateAll=updateAll
        self.debug = debug
        self.csb = CSB
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X, y, sample_weight=None):
        return super(SMOTEBoost, self).fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(SMOTEBoost, self)._validate_estimator(default=DecisionTreeClassifier(max_depth=1))

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator_, 'predict_proba'):
                raise TypeError(
                    "AdaCostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        return self._boost_discrete(iboost, X, y, sample_weight, random_state)

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)
        proba = estimator.predict_proba(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)
        n_classes = self.n_classes_

        misclassified_weights = 0.0
        #
        # for idx, row in enumerate(y_predict):
        #     if row == y [idx]:
        #         continue
        #         # misclassified_weights += (sample_weight[idx] * max(proba[idx][0], proba[idx][1]))/len(y)
        #     else:
        #         misclassified_weights += (sample_weight[idx] * max(proba[idx][0], proba[idx][1]))/len(y)
        # print misclassified_weights
        # alpha = 0.5 * np.log(( 1 - misclassified_weights)/ (misclassified_weights))

        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        alpha = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))
        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        # estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # if self.updateAll:
        # # Only boost the weights if I will fit again
        #     if not iboost == self.n_estimators - 1:
        #         # Only boost positive weights
        #         for idx, row in enumerate(sample_weight):
        #             if y[idx] == 1 and y_predict[idx] != 1:
        #                 sample_weight[idx] *= self.cost_positive * np.exp(alpha )
        #             elif y[idx] == -1 and y_predict[idx] != -1:
        #                 sample_weight[idx] *= self.cost_negative * np.exp(alpha )
        #             elif y[idx] == 1 and y_predict[idx] == 1:
        #                 sample_weight[idx] *= self.cost_positive * np.exp(-alpha )
        #             elif y[idx] == -1 and y_predict[idx] == -1:
        #                 sample_weight[idx] *= self.cost_negative * np.exp(-alpha )
        # else:
        #     if not iboost == self.n_estimators - 1:
        #         # Only boost positive weights
        #         for idx, row in enumerate(sample_weight):
        #             if y[idx] == 1 and y_predict[idx] != 1:
        #                 sample_weight[idx] *= self.cost_positive * np.exp(alpha )
        #             elif y[idx] == -1 and y_predict[idx] != -1:
        #                 sample_weight[idx] *= self.cost_negative * np.exp(alpha )
        
        
        if self.updateAll:
        # Only boost the weights if I will fit again
            if not iboost == self.n_estimators - 1:
                # Only boost positive weights
                for idx, row in enumerate(sample_weight):
                    if y[idx] == 1 and y_predict[idx] != 1:
                        if X[idx][self.saIndex] == self.saValue:
                            if self.csb == "CSB2":
                                sample_weight[idx] *=  self.cost_positive * np.exp(alpha * max(proba[idx][0], proba[idx][1]))
                            elif self.csb == "CSB1":
                                sample_weight[idx] *=  self.cost_positive * np.exp(alpha )
                        else:
                            if self.csb == "CSB2":
                                sample_weight[idx] *=  self.cost_positive * np.exp( alpha * max(proba[idx][0], proba[idx][1]))
                            elif self.csb == "CSB1":
                                sample_weight[idx] *=  self.cost_positive * np.exp( alpha )
                    elif y[idx] == -1 and y_predict[idx] != -1:
                        if X[idx][self.saIndex] == self.saValue:
                            if self.csb == "CSB2":
                                sample_weight[idx] *=  self.cost_negative *np.exp( alpha * max(proba[idx][0], proba[idx][1]))
                            elif self.csb == "CSB1":
                                sample_weight[idx] *=  self.cost_negative *np.exp( alpha )
                        else:
                            if self.csb == "CSB2":
                                sample_weight[idx] *= self.cost_negative * np.exp( alpha * max(proba[idx][0], proba[idx][1]))
                            elif self.csb == "CSB1":
                                sample_weight[idx] *= self.cost_negative * np.exp( alpha )
                    elif y[idx] == 1 and y_predict[idx] == 1:
                        if self.csb == "CSB2":
                            sample_weight[idx] *= self.cost_positive * np.exp(-alpha * max(proba[idx][0], proba[idx][1]))
                        elif self.csb == "CSB1":
                            sample_weight[idx] *= self.cost_positive * np.exp(-alpha )

                    elif y[idx] == -1 and y_predict[idx] == -1:
                        if self.csb == "CSB2":
                            sample_weight[idx] *= self.cost_negative * np.exp(-alpha * max(proba[idx][0], proba[idx][1]))
                        elif self.csb == "CSB1":
                            sample_weight[idx] *= self.cost_negative * np.exp(-alpha )
        else:
            if not iboost == self.n_estimators - 1:
                # Only boost positive weights
                for idx, row in enumerate(sample_weight):
                    if y[idx] == 1 and y_predict[idx] != 1:
                        if X[idx][self.saIndex] == self.saValue:
                            if self.csb == "CSB2":
                                sample_weight[idx] *=  self.cost_positive * np.exp(alpha * max(proba[idx][0], proba[idx][1]))
                            elif self.csb == "CSB1":
                                sample_weight[idx] *=  self.cost_positive * np.exp(alpha )
                        else:
                            if self.csb == "CSB2":
                                sample_weight[idx] *=  self.cost_positive * np.exp( alpha * max(proba[idx][0], proba[idx][1]))
                            elif self.csb == "CSB1":
                                sample_weight[idx] *=  self.cost_positive * np.exp( alpha )

                    elif y[idx] == -1 and y_predict[idx] != -1:
                        if X[idx][self.saIndex] == self.saValue:
                            if self.csb == "CSB2":
                                sample_weight[idx] *=  self.cost_negative *np.exp( alpha * max(proba[idx][0], proba[idx][1]))
                            elif self.csb == "CSB1":
                                sample_weight[idx] *=  self.cost_negative *np.exp( alpha )
                        else:
                            if self.csb == "CSB2":
                                sample_weight[idx] *= self.cost_negative * np.exp( alpha * max(proba[idx][0], proba[idx][1]))
                            elif self.csb == "CSB1":
                                sample_weight[idx] *= self.cost_negative * np.exp( alpha )
        # if self.debug:
        #     if iboost !=0:
        #         y_predict = self.predict(X)
        #         y_predict_probs = self.decision_function(X)
        #         incorrect = y_predict != y
        #         training_error = np.mean(np.average(incorrect, axis=0))
        #         train_auc = sklearn.metrics.balanced_accuracy_score(y, y_predict)
        #         train_fairness = self.calculate_fairness(X,y,y_predict)
        #
        #         y_predict = self.predict(self.X_test)
        #         y_predict_probs = self.decision_function(self.X_test)
        #         incorrect = y_predict != self.y_test
        #         test_error = np.mean(np.average(incorrect, axis=0))
        #         test_auc = sklearn.metrics.balanced_accuracy_score(self.y_test, y_predict)
        #         test_fairness = self.calculate_fairness(self.X_test,self.y_test,y_predict)
        #         self.performance.append(str(iboost) + "," + str(training_error) + ", " + str(train_auc) + ", " + str(train_fairness) + ","+ str(test_error) + ", " + str(test_auc)+ ", " + str(test_fairness))
        #         # print str(iboost) + "," + str(training_error) + ", " + str(train_auc) + ", " + str(train_fairness) + ","+ str(test_error) + ", " + str(test_auc)+ ", " + str(test_fairness)

        return sample_weight, alpha, estimator_error

    def get_performance_over_iterations(self):
        return self.performance

    def get_weights_over_iterations(self):
        return self.weight_list[-1]

    def get_initial_weights(self):
        return self.weight_list[0]



    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        pred = sum((estimator.predict(X) == classes).T * w for estimator, w in zip(self.estimators_, self.estimator_alphas_))

        pred /= self.estimator_alphas_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        X = self._validate_X_predict(X)

        if n_classes == 1:
            return np.ones((X.shape[0], 1))

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(_samme_proba(estimator, n_classes, X)
                        for estimator in self.estimators_)
        else:   # self.algorithm == "SAMME"
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_alphas_))

        proba /= self.estimator_alphas_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        return np.log(self.predict_proba(X))

    def calculate_fairness(self, data, labels, predictions):

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
            if val[self.saIndex] == self.saValue:
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

        diff_tpr = tpr_non_protected - tpr_protected
        diff_tnr = tnr_non_protected - tnr_protected

        if diff_tpr > 0:
            self.cost_protected_positive = (1 + diff_tpr) * self.cost_positive
        elif diff_tpr < 0:
            self.cost_protected_positive = (1 + abs(diff_tpr)) * self.cost_positive

        if diff_tnr > 0:
            self.cost_protected_negative = (1 + diff_tnr) * self.cost_negative
        elif diff_tpr < 0:
            self.cost_protected_negative = (1 + abs(diff_tnr)) * self.cost_negative

        # print "dTPR = " + str((tpr_non_protected - tpr_protected)*100) +", dTNR = " + str((tnr_non_protected - tnr_protected)*100)

        return 1 - (abs((tpr_non_protected - tpr_protected)) + abs((tnr_non_protected - tnr_protected)))/2
