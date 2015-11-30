from __future__ import division
import cPickle as pickle
import gzip
import numpy as np
import data_tweaking as dt
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import normalize
from sklearn.ensemble import ExtraTreesClassifier
import sys
import random


attrfile = "sle_data/rheumatol/attributes.txt"
goldDataString = "sle_data/rheumatol/gold_data.txt"
goldInstancesString = "sle_data/rheumatol/gold_fixed_instance.txt"
golddata_matrix = dt.readSparse(attributesString=attrfile,dataString=goldDataString,instancesString=goldInstancesString)
gold_labels = dt.get_labels_according_to_data_order(dataString=goldDataString,instancesString=goldInstancesString)

for i in range(len(gold_labels)):
	if i == 0:
		if gold_labels[i] == "100":
			gold_labels[i] = "1"
		else:
			gold_labels[i] = "0"
	elif gold_labels[i] == "100":# for fixing the labels AND bootstrapping a negative
		gold_labels[i] = "1"
	elif gold_labels[i] == "-100":# for fixing the labels
		gold_labels[i] = "0"

golddata_matrix = golddata_matrix.todense()

clf = ExtraTreesClassifier(max_features=100) # "auto" is default max_features (sqrt(n))
golddata_matrix = clf.fit(golddata_matrix,gold_labels).transform(golddata_matrix)


rows_in_gold = golddata_matrix.shape[0] # == len(gold_labels)


def normalize(m):
	m = m.T
	m = (m - m.min())/np.ptp(m)
	return m.T
golddata_matrix = normalize(golddata_matrix)
#golddata_matrix = clf.fit(golddata_matrix,gold_labels).transform(golddata_matrix)

pickleArray = [[golddata_matrix,gold_labels],
		[golddata_matrix,gold_labels],
		[golddata_matrix,gold_labels],
		#[pretrain_matrix]]
		[golddata_matrix]]
f = gzip.open("sle.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()
