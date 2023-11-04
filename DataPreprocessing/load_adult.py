from __future__ import division
# import urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

# sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
# import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)

os.chdir("..")

def load_adult(sensitive_attribute):

	FEATURES_CLASSIFICATION = ["workclass", "education", "Maritial-status", "occupation", "race", "sex", "country","age", "Capital-gain","Capital-loss","Hours-per-week"] #features to be used for classification
	CONT_VARIABLES = ["age", "Capital-gain","Capital-loss","Hours-per-week"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
	CLASS_FEATURE = "Class-label" # the decision variable
	SENSITIVE_ATTRS = [sensitive_attribute]

	# load the data and get some stats
	df = pd.read_csv("DataPreprocessing/adult.csv")

	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])

	# convert class label 0 to -1
	y = data[CLASS_FEATURE]
	y[y==0] = -1

	X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
	x_control = defaultdict(list)

	feature_names = []
	for attr in FEATURES_CLASSIFICATION:
		vals = data[attr]
		if attr in CONT_VARIABLES:
			vals = [float(v) for v in vals]
			vals = preprocessing.scale(vals) # 0 mean and 1 variance
			vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

		else: # for binary categorical variables, the label binarizer uses just one var instead of two
			lb = preprocessing.LabelBinarizer()
			lb.fit(vals)
			vals = lb.transform(vals)

		# add to sensitive features dict
		if attr in SENSITIVE_ATTRS:
			x_control[attr] = vals

		# add to learnable features
		X = np.hstack((X, vals))

		if attr in CONT_VARIABLES: # continuous feature, just append the name
			feature_names.append(attr)
		else: # categorical features
			if vals.shape[1] == 1: # binary features that passed through lib binarizer
				feature_names.append(attr)
			else:
				for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
					feature_names.append(attr + "_" + str(k))

	# convert the sensitive feature to 1-d array
	x_control = dict(x_control)
	for k in x_control.keys():
		assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
		x_control[k] = np.array(x_control[k]).flatten()

	return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, x_control


if __name__ == "__main__":
	load_adult("race")
