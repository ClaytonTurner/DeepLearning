"""
This file is for algorithmic comparison's sake for BOW and CUIs
in relation to the deep learning algorithms
"""

import sys
import os
import gzip
import cPickle as pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from logistic_sgd import load_data

dataset = "sle.pkl.gz"
#datasets = load_data(dataset)
f = gzip.open(dataset,"rb")
datasets = pickle.load(f)

(train_set_x, train_set_y) = datasets[0]
(valid_set_x, valid_set_y) = datasets[1] # We just need to merge with train here
(test_set_x, test_set_y) = datasets[2]

test_set_y = np.asarray(test_set_y,dtype=np.float32)

train_set_x = np.concatenate([train_set_x,valid_set_x])
train_set_y = np.concatenate([train_set_y,valid_set_y])

clf = GaussianNB()
clf = clf.fit(train_set_x,train_set_y)

probas = clf.predict_proba(test_set_x)

save_probas = []
for proba in probas:
	save_probas.append(proba[1])

#external_set = "external_test.pkl.gz"
#f = gzip.open(external_set,"rb")
#ext_sets = pickle.load(f)
#(ext_set_x,ext_set_y) = ext_sets
#f.close()

#ext_set_x = np.asarray(ext_set_x,dtype=np.float32)
#ext_set_y = np.asarray(ext_set_y,dtype=np.float32)

#ext_probas = clf.predict_proba(ext_set_x)
#save_ext_probas = []
#for proba in ext_probas:
#    save_ext_probas.append(proba[1])

#fold = int(sys.argv[1])
#if fold < 10:
#	fold = "0"+str(fold)
#else:
#	fold = str(fold)

#fname = os.path.expanduser("~/DeepLearning/results_run_nb_bow/")
fname = os.path.expanduser("~/DeepLearning/results_run_nb_cui/")
iteration = sys.argv[1]
np.savetxt(fname+iteration+"_labels.txt",test_set_y,fmt="%s")
np.savetxt(fname+iteration+"_p_values.txt",save_probas,fmt="%s")
#np.savetxt(fname+"_external_labels.txt",ext_set_y,fmt="%s")
#np.savetxt(fname+"_external_p_values.txt",save_ext_probas,fmt="%s")
print "Completed Naive Bayes Classifier"
