import math
from utils import draw, normalize, sign
import numpy as np
#from weaklearners.decisionstump import buildDecisionStump
from sklearn.tree.tree import BaseDecisionTree, DTYPE, DecisionTreeClassifier

# compute the weighted error of a given hypothesis on a distribution
# return all of the hypothesis results and the error
def weightedLabelError(h, examples, weights):
   hypothesisResults = [h.predict([x]) * y for (x, y) in examples]  # +1 if correct, else -1
   return hypothesisResults, sum(w for (z, w) in zip(hypothesisResults, weights) if z < 0)


# boost: [(list, label)], learner, int -> (list -> label)
# where a learner is (() -> (list, label)) -> (list -> label)
# boost the weak learner into a strong learner
def adaboostGenerator(examples, weakLearner, rounds, computeError=weightedLabelError):
   distr = normalize([1.0] * len(examples))
   hypotheses = [None] * rounds
   alpha = [0] * rounds

   for t in range(rounds):
      print(t)

      def drawExample():
         return examples[draw(distr)]
      #weakLearner=DecisionTreeClassifier()
      z=np.array([examples[draw(distr)] for i in range(500)])
      #print(z.shape)
      x=[q[0] for q in z]
      y=[q[1] for q in z]
      x=np.array(x)
      #print(y)
      hypotheses[t] = weakLearner.fit(x,y)
      hypothesisResults, error = computeError(hypotheses[t], examples, distr)

      alpha[t] = 0.5 * math.log((1 - error) / (.0001 + error))
      distr = normalize([d * math.exp(-alpha[t] * r)
                         for (d, r) in zip(distr, hypothesisResults)])

      def weightedMajorityVote(x):
         return sign(sum(a * h.predict([x]) for (a, h) in zip(alpha, hypotheses[:t + 1])))

      yield weightedMajorityVote, hypotheses[:t + 1], alpha[:t + 1]


# convenience wrapper for boosting
# returns the outputted hypothesis from boosting
def boost(trainingData, numRounds=20, weakLearner=DecisionTreeClassifier(), computeError=weightedLabelError):
   generator = adaboostGenerator(trainingData, weakLearner, numRounds, computeError)

   for h, _, _ in generator:
      pass

   return h


# call an optional diagnostic function to output round-wise intermediate results
# return more information at the end
def detailedBoost(trainingData, numRounds=20, weakLearner=DecisionTreeClassifier(),
                  computeError=weightedLabelError, diagnostic=None):
   generator = adaboostGenerator(trainingData, weakLearner, numRounds, computeError)

   for h, hypotheses, alphas in generator:
      if diagnostic is not None:
         diagnostic({'h': h, 'hypoheses': hypotheses, 'alphas': alphas})

   return h, hypotheses, alphas


# compute the margin of a point with the label to express whether it's correct
# alpha is the weights of the hypotheses from the boosting algorithm
def marginWithLabel(point, label, hypotheses, alpha):
    return label * sum(a * h(point) for (h, a) in zip(hypotheses, alpha)) / sum(alpha)


# compute the margin of a point
# alpha is the weights of the hypotheses from the boosting algorithm
def margin(point, hypotheses, alpha):
    sm= sum(a * h.predict([point]) for (h, a) in zip(hypotheses, alpha)) / sum(alpha)
    #print(sm, sum(alpha),sum(a * h.predict([point]) for (h, a) in zip(hypotheses, alpha)))
    return sm


# compute the absolute value of the margin of a point
# alpha is the weights of the hypotheses from the boosting algorithm
def absMargin(point, hypotheses, alpha):
   return abs(margin(point, hypotheses, alpha))
