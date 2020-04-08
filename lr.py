def lrDetailedSKL(data):
   from sklearn import linear_model
   points, labels = zip(*data)
   clf = linear_model.LogisticRegression()
   lrClassifier = clf.fit(points, labels)
   return (
      lambda x: lrClassifier.predict_proba([x])[0][1],
      lambda x: 1 if lrClassifier.predict_proba([x])[0][1] >= 0.5 else -1
   )


def lrSKL(data):
   return lrDetailedSKL(data)[1]
