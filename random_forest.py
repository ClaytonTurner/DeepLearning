"""
This file is for algorithmic comparison's sake for BOW and CUIs
in relation to the deep learning algorithms
"""

import sys
import os
import gzip
import cPickle as pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from logistic_sgd import load_data

dataset = "sle.pkl.gz"
#datasets = load_data(dataset)
f = gzip.open(dataset,"rb")
datasets = pickle.load(f)

(train_set_x, train_set_y) = datasets[0]
(valid_set_x, valid_set_y) = datasets[1] # We just need to merge with train here
(test_set_x, test_set_y) = datasets[2]

train_set_x = np.concatenate([train_set_x,valid_set_x])
train_set_y = np.concatenate([train_set_y,valid_set_y])

clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(train_set_x,train_set_y) # maybe try fit_transform

probas = clf.predict_proba(test_set_x)

fold = int(sys.argv[1])
if fold < 10:
	fold = "0"+str(fold)
else:
	fold = str(fold)
fname = os.path.expanduser("~/DeepLearning/results/"+fold)
np.savetxt(fname+"_labels.txt",test_set_y,fmt="%s")
np.savetxt(fname+"_p_values.txt",probas,fmt="%s")
print "Completed Random Forest Classifier"
