from Competitors.errorfunctions import signedStatisticalParity, precomputedLabelError, precomputedLabelStatisticalParity

from Competitors.decisionstump import buildDecisionStump
from Competitors import boosting
import numpy
import random

try:
   import matplotlib.pyplot as plt
except (ImportError, RuntimeError):
   pass

def sign(x):
   return 1 if x >= 0 else -1

class marginAnalyzer(object):
   def __init__(self, data=None, defaultThreshold=None, marginRange=None,
                protectedIndex=None, protectedValue=None, bulkMargin=None):
      print ("print first...")
      self.defaultThreshold = defaultThreshold
      if data is not None and 'trainingData' not in dir(self):
         self.splitData(data)

      if bulkMargin is None:
         self.trainingMargins = [self.margin(x[0]) for x in self.trainingData]
         self.validationMargins = [self.margin(x[0]) for x in self.validationData]
         self.bulkMargin = None
      else:
         print("Computing bulk margins...")
         self.trainingMargins = bulkMargin([x[0] for x in self.trainingData])
         self.validationMargins = bulkMargin([x[0] for x in self.validationData])
         self.bulkMargin = bulkMargin
         print("Done")

      self.margins = numpy.concatenate([self.trainingMargins, self.validationMargins], axis=0)
      self.minMargin, self.maxMargin = min(self.margins), max(self.margins)
      self.minShift, self.maxShift = (self.minMargin - self.defaultThreshold,
                                      self.maxMargin - self.defaultThreshold)
      if marginRange is not None:
         self.marginRange = marginRange
      else:
         self.marginRange = (self.minMargin, self.maxMargin)
      self.setProtected(protectedIndex, protectedValue)

   def margin(self, x):
      raise NotImplementedError()

   def splitData(self, data):
      self.data = random.sample(data, len(data))
      self.trainingData = data[:len(data) // 2]
      self.validationData = data[len(data) // 2:]

   def setProtected(self, protectedIndex, protectedValue):
      assert protectedIndex is not None
      assert protectedValue is not None
      self.protectedIndex = protectedIndex
      self.protectedValue = protectedValue

   # returns True if x is protected, False otherwise;
   # can be used as a condition for conditionalShiftClassifier
   def protected(self, x):
      assert self.protectedIndex is not None
      assert self.protectedValue is not None
      return x[self.protectedIndex] == self.protectedValue

   # returns a classifier which takes a data point as an input
   # and returns 1 if margin is above threshold, -1 otherwise
   def classifier(self, threshold=None):
      if threshold is None:
         threshold = lambda x: self.defaultThreshold
      return lambda x: 1 if self.margin(x) >= threshold(x) else -1

   # returns a classifier with shifted threshold for data points satisfying condition
   def conditionalShiftClassifier(self, shift, condition=None):
      if condition is None:
         condition = self.protected
      return self.classifier(
        lambda x: self.defaultThreshold + shift if condition(x) else self.defaultThreshold)

   def conditionalMarginShiftedLabels(self, data, margins, shift, condition):
      # condition is margin >= threshold + shift, so that if shift is negative
      # the threshold is lower.
      shiftedMargins = [(m - shift if condition(x[0]) else m) for (m, x) in zip(margins, data)]
      labels = [1 if m >= self.defaultThreshold else -1 for m in shiftedMargins]
      return labels

   # finds the shift which achieves goal=0 under condition
   # goal takes two arguments, data and h
   def optimalShift(self, goal=None, condition=None, rounds=20):
      if goal is None:
         goal = lambda d, h: signedStatisticalParity(d, self.protectedIndex, self.protectedValue, h)
      if condition is None:
         condition = self.protected

      low = self.minShift
      high = self.maxShift
      dataToUse = self.validationData

      minGoalValue = goal(dataToUse, self.conditionalShiftClassifier(low, condition))
      maxGoalValue = goal(dataToUse, self.conditionalShiftClassifier(high, condition))
      # print((low, minGoalValue))
      # print((high, maxGoalValue))

      if sign(minGoalValue) != sign(maxGoalValue):
         # a binary search for zero
         for _ in range(rounds):
            midpoint = (low + high) / 2
            if (sign(goal(dataToUse, self.conditionalShiftClassifier(low, condition))) ==
                    sign(goal(dataToUse, self.conditionalShiftClassifier(midpoint, condition)))):
               low = midpoint
            else:
               high = midpoint
         return midpoint
      else:
         print("Warning: bisection method not applicable")
         bestShift = None
         bestVal = float('inf')
         step = (high - low) / rounds
         for newShift in numpy.arange(low, high, step):
            newVal = goal(dataToUse, self.conditionalShiftClassifier(newShift, condition))
            print(newVal)
            newVal = abs(newVal)
            if newVal < bestVal:
               bestShift = newShift
               bestVal = newVal
         return bestShift

   def optimalShiftClassifier(self, goal=None, condition=None, rounds=20):
      if goal is None:
         goal = lambda d, h: signedStatisticalParity(d, self.protectedIndex, self.protectedValue, h)
      if condition is None:
         condition = self.protected
      return self.conditionalShiftClassifier(self.optimalShift(goal, condition, rounds), condition)

   # plots margins and incorrect margins for entire population and class satisfying condition
   def plotMarginHistogram(self, condition=None, bins=40, plotLabels=('population', 'protected'), filename=None):
      if condition is None:
         condition = self.protected
      marginList = self.margins
      protectedMargins = [v for ((x, y), v) in zip(self.data, self.margins) if condition(x)]
      incorrectMargins = [v for ((x, y), v) in zip(self.data, self.margins)
                          if (v - self.defaultThreshold) * y < 0]
      incorrectProtectedMargins = [
        v for ((x, y), v) in zip(self.data, self.margins)
        if (v - self.defaultThreshold) * y < 0 and condition(x)]

      f, (ax1, ax2) = plt.subplots(2, 1)

      # distribution of signed margins on test data
      ax1.hist(marginList, bins=bins, label=plotLabels[0])
      ax1.hist(protectedMargins, bins=bins, label=plotLabels[1], color='y')
      ax1.set_xlim([self.marginRange[0], self.marginRange[1]])
      ax1.set_title("Confidence values")

      ax2.hist(incorrectMargins, bins=bins, label=plotLabels[0])
      ax2.hist(incorrectProtectedMargins, bins=bins, label=plotLabels[1], color='y')
      ax2.set_xlim([self.marginRange[0], self.marginRange[1]])
      ax2.set_title("Confidence values of incorrect examples")

      plt.subplots_adjust(hspace=.75)
      plt.legend()

      if filename is None:
         plt.show()
      else:
         plt.savefig(filename)
         plt.clf()

   def plotTradeoff(self, data=None, n=100, filename=None):
      plotTitle = 'Shifted Decision Boundary Bias vs Error'
      plotLabels = ('Label error', 'Bias')
      condition = self.protected

      if data is None:
         data = self.validationData
         pts, labels = zip(*data)
         precomputedMargins = self.validationMargins
      else:
         pts, labels = zip(*data)
         if self.bulkMargin is not None:
            precomputedMargins = self.bulkMargin(pts)
         else:
            precomputedMargins = [self.margin(x[1]) for x in data]

      xs = numpy.arange(self.minShift, self.maxShift, (self.maxShift - self.minShift) / n)
      shiftedLabels = lambda shift: self.conditionalMarginShiftedLabels(data, precomputedMargins, shift, condition)

      srError = [0] * len(xs)
      srBias = [0] * len(xs)

      for i, shift in enumerate(xs):
         newLabels = shiftedLabels(shift)
         srError[i] = precomputedLabelError(data, newLabels)
         srBias[i] = precomputedLabelStatisticalParity(pts, newLabels, self.protectedIndex, self.protectedValue)
         print("%.4f,%.4f,%.4f" % (xs[i], srError[i], srBias[i]))

      width = 3

      plt.plot(xs, srError, label=plotLabels[0], linewidth=width)
      plt.plot(xs, srBias, label=plotLabels[1], linewidth=width)
      plt.title(plotTitle)
      plt.gca().invert_xaxis()
      plt.axhline(0, color='black')
      plt.figaspect(10.0)
      plt.legend(loc='center left')

      if filename is None:
         plt.show()
      else:
         plt.savefig(filename)
         plt.clf()

class boostingMarginAnalyzer(marginAnalyzer):
   def __init__(self, data, protectedIndex, protectedValue, numRounds, weakLearner=buildDecisionStump, computeError=boosting.weightedLabelError):

      print ("print second...")
      print (protectedIndex,protectedValue)


      self.splitData(data)
      _, self.hypotheses, self.alphas = boosting.detailedBoost(
         self.trainingData, numRounds, weakLearner, computeError)
      super().__init__(defaultThreshold=0, marginRange=(-1, 1), protectedIndex=protectedIndex,
                       protectedValue=protectedValue)

   def margin(self, x):
      return boosting.margin(x, self.hypotheses, self.alphas)