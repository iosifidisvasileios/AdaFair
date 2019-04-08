from __future__ import division
import urllib2
import os,sys

import collections
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
# import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)



def load_dutch_data():

	FEATURES_CLASSIFICATION = ["sex","age","household_position","household_size","prev_residence_place","citizenship","country_birth","edu_level", "economic_status","cur_eco_activity","Marital_status"]
	CONT_VARIABLES = ["age","household_position","household_size","prev_residence_place","citizenship","country_birth","edu_level", "economic_status","cur_eco_activity","Marital_status"]

	CLASS_FEATURE = "occupation" # the decision variable
	SENSITIVE_ATTRS = ["sex"]

	COMPAS_INPUT_FILE = "DataPreprocessing/dutch.csv"


	# load the data and get some stats
	df = pd.read_csv(COMPAS_INPUT_FILE)

	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])


	""" Feature normalization and one hot encoding """
	y = data[CLASS_FEATURE]
	y[y==0] = -1

	print collections.Counter(data["sex"])

	print "\nNumber of people with prestigious professions"
	print pd.Series(y).value_counts()
	print "\n"
	# convert class label 0 to -1




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

	# sys.exit(1)

	"""permute the date randomly"""
	perm = range(0,X.shape[0])
	shuffle(perm)
	X = X[perm]
	y = y[perm]
	for k in x_control.keys():
		x_control[k] = x_control[k][perm]


	# X = ut.add_intercept(X)

	# feature_names = ["intercept"] + feature_names
	# assert(len(feature_names) == X.shape[1])
	# print "Features we will be using for classification are:", feature_names, "\n"
	# print x_control

	return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 0, x_control