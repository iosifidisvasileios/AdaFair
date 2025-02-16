# coding: utf-8

import pdb
import timeit
import time


from sklearn.model_selection import StratifiedShuffleSplit
from adafair import AdaFair
from adafair import load_adult
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def calculate_performance(data, labels, predictions, saIndex, saValue):
  tp_protected = 0.
  tn_protected = 0.
  fp_protected = 0.
  fn_protected = 0.

  tp_non_protected = 0.
  tn_non_protected = 0.
  fp_non_protected = 0.
  fn_non_protected = 0.
  for idx, val in enumerate(data):
    # protected population
    if val[saIndex] == saValue:
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

  output = dict()
  output["balanced_accuracy"] = ((tp_protected + tp_non_protected) / (
      tp_protected + tp_non_protected + fn_protected + fn_non_protected) +
      (tn_protected + tn_non_protected) / (
      tn_protected + tn_non_protected + fp_protected +
      fp_non_protected)) * 0.5

  output["accuracy"] = accuracy_score(labels, predictions)
  output["fairness"] = abs(tpr_non_protected - tpr_protected) + abs(
      tnr_non_protected - tnr_protected)
  output["TPR_protected"] = tpr_protected
  output["TPR_non_protected"] = tpr_non_protected
  output["TNR_protected"] = tnr_protected
  output["TNR_non_protected"] = tnr_non_protected
  return output


# since = timeit.default_timer()
since_1, since_2 = time.time(), timeit.default_timer()

X, y, sa_index, p_Group = load_adult("race")
strat_suffle = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
results = []
for train_index, test_index in strat_suffle.split(X, y):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  classifier = AdaFair(n_estimators=200, saIndex=sa_index,
                       saValue=p_Group, trade_off_c=1.0)

  classifier.fit(X_train, y_train)
  y_pred_labels = classifier.predict(X_test)
  # print(calculate_performance(
  #     X_test, y_test, y_pred_labels, sa_index, p_Group))

  ans = calculate_performance(
      X_test, y_test, y_pred_labels, sa_index, p_Group)
  results.append(ans)
  print(ans)


# time_elapsed = timeit.default_timer() - since
# print("Time cost: ", time_elapsed)
elapsed_1, elapsed_2 = (time.time() - since_1,
                        timeit.default_timer() - since_2)
print("Time cost #1: ", elapsed_1)
print("Time cost #2: ", elapsed_2)
'''
Time cost #1:  64.78533887863159
Time cost #2:  64.78498775
'''

'''
{'balanced_accuracy': 0.7860894302533374, 'accuracy': 0.8465965688987271, 'fairness': 0.08014420916703435, 'TPR_protected': 0.7277486910994765, 'TPR_non_protected': 0.6603221083455344, 'TNR_protected': 0.9168224299065421, 'TNR_non_protected': 0.9041048034934498}
{'balanced_accuracy': 0.8042520761063807, 'accuracy': 0.8280022136137244, 'fairness': 0.10026620274590581, 'TPR_protected': 0.7184466019417476, 'TPR_non_protected': 0.7610619469026548, 'TNR_protected': 0.9000951474785919, 'TNR_non_protected': 0.8424442896935933}
{'balanced_accuracy': 0.7653664721959423, 'accuracy': 0.8532374100719424, 'fairness': 0.07600321611727723, 'TPR_protected': 0.5422222222222223, 'TPR_non_protected': 0.5965260545905707, 'TNR_protected': 0.9579439252336449, 'TNR_non_protected': 0.9362445414847161}
{'balanced_accuracy': 0.7523147600651741, 'accuracy': 0.8428334255672385, 'fairness': 0.0433849287230621, 'TPR_protected': 0.5693069306930693, 'TPR_non_protected': 0.5731108930323847, 'TNR_protected': 0.9653846153846154, 'TNR_non_protected': 0.9258036490008688}
{'balanced_accuracy': 0.808674478345422, 'accuracy': 0.8339789706696181, 'fairness': 0.04630279094128442, 'TPR_protected': 0.7715736040609137, 'TPR_non_protected': 0.7572197748409202, 'TNR_protected': 0.8857142857142857, 'TNR_non_protected': 0.8537653239929948}

Time cost:  64.713330666
Time cost:  66.692567709  (after modification)
Time cost:  65.39499058300001 (after modification)
> /Users/elise/GitHubLab/AdaFair/run_example.py(104)<module>()
-> print(results)
(Pdb)

(Pdb) y
array([-1, -1, -1, ...,  1,  1,  1])
(Pdb) sa_index
44
(Pdb) p_Group
0
(Pdb) X[:, 44]
array([1., 0., 1., ..., 1., 1., 1.])
(Pdb) sum(X[:, 44] == 1)
38859
(Pdb) sum(X[:, 44] == 0)
6316
(Pdb) X.shape
(45175, 91)
(Pdb) 38859+6316
45175

(Pdb) results
[]
(Pdb) X, y, sa_index, p_Group = load_adult('sex')
(Pdb) X.shape
(45175, 91)
(Pdb) y
array([-1, -1, -1, ...,  1,  1,  1])
(Pdb) sa_index
45
(Pdb) p_Group
0
(Pdb) X[:, 45]
array([1., 0., 0., ..., 1., 1., 1.])
(Pdb) sum(X[:, 45] == 1)
30495
(Pdb) sum(X[:, 45] == 0)
14680
(Pdb) 30495 + 14680
45175
(Pdb) set(X[:, 45])
{0.0, 1.0}
'''


def check_type(num, types):
  for typ in types:
    if isinstance(num, typ):
      return True
  return False


pdb.set_trace()
print(results)


'''
>>> def func(err):
...     return 1*(np.log((1.-err)/ err)+ np.log(n_classes-1.))
...
>>> n_classes = 2
>>> func(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in func
ZeroDivisionError: float division by zero
>>> def check_zero(n):
...     return n if n!=0 else 1e-16
...
>>> def check_func(err):
...     return 1*(np.log((1.-err)/ check_zero(err))+ np.log(n_classes -1.))
...
>>> check_func(0.)
36.841361487904734
'''
