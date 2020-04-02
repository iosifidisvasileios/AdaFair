from __future__ import division

import collections

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import preprocessing
from random import seed


SEED = 1234
seed(SEED)
np.random.seed(SEED)


def load_diabetes():
    FEATURES_CLASSIFICATION = ["race", "gender", "age", "weight", "admission_type_id",
                               "discharge_disposition_id", "admission_source_id", "time_in_hospital", "payer_code",
                               "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications",
                               "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2",
                               "diag_3", "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin", "repaglinide",
                               "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
                               "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
                               "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
                               "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
                               "metformin-pioglitazone", "change", "readmitted"]
    CONT_VARIABLES = ["admission_type_id",
                      "discharge_disposition_id", "admission_source_id", "time_in_hospital", "num_lab_procedures",
                      "num_procedures", "num_medications",
                      "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    CLASS_FEATURE = "diabetesMed"  # the decision variable
    SENSITIVE_ATTRS = ["gender"]

    COMPAS_INPUT_FILE = "DataPreprocessing/diabetic_data.csv"


    df = pd.read_csv(COMPAS_INPUT_FILE)

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])
    print(collections.Counter(data["gender"]))

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == 'No'] = "1"
    y[y == "Yes"] = '-1'
    y = np.array([int(k) for k in y])
    print(collections.Counter(y))

    X = np.array([]).reshape(len(y),
                             0)  # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col

        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals

        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES:  # continuous feature, just append the name
            feature_names.append(attr)
        else:  # categorical features
            if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    feature_names.append('target')
    print(np.sum(X[:, feature_names.index(SENSITIVE_ATTRS[0])]))
    print(len(X[:, feature_names.index(SENSITIVE_ATTRS[0])]))
    return X, y, feature_names.index(SENSITIVE_ATTRS[0]), 1, x_control
