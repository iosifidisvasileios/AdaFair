import math
from utils import draw, normalize, sign
import numpy as np
#from weaklearners.decisionstump import buildDecisionStump
from AdaCost import AdaCostClassifier

# compute the weighted error of a given hypothesis on a distribution
# return all of the hypothesis results and the error

# convenience wrapper for boosting
# returns the outputted hypothesis from boosting



# call an optional diagnostic function to output round-wise intermediate results
# return more information at the end
def detailedBoost(train, weakLearner, sa_index, p_Group, numRounds=20):
   print(numRounds)
   classifier = AdaCostClassifier(saIndex=sa_index, saValue=p_Group, n_estimators=numRounds, CSB="CSB1")
   z=np.array(train)
   x=[q[0] for q in z]
   y=[q[1] for q in z]
   classifier.fit(x, y)
   return '-', classifier.estimators_, classifier.estimator_alphas_


# compute the margin of a point with the label to express whether it's correct
# alpha is the weights of the hypotheses from the boosting algorithm
def marginWithLabel(point, label, hypotheses, alpha):
    return label * sum(a * h(point) for (h, a) in zip(hypotheses, alpha)) / sum(alpha)


# compute the margin of a point
# alpha is the weights of the hypotheses from the boosting algorithm
def margin(point, hypotheses, alpha):
    #z=np.array(point)
    #X=[q[0] for q in z]
    #y=[q[1] for q in z]
    #sm=[]
    #for x in X:
    sm= sum([a * h.predict([point])[0] for (h, a) in zip(hypotheses, alpha)]) / sum(alpha)
    #print(sm)#, sum(alpha),sum([a * h.predict([point])[0] for (h, a) in zip(hypotheses, alpha)]))
    return sm


# compute the absolute value of the margin of a point
# alpha is the weights of the hypotheses from the boosting algorithm
def absMargin(point, hypotheses, alpha):
   return abs(margin(point, hypotheses, alpha))
