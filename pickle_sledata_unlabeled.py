'''
This file is for pickling the data so we can predict new patients from our unlabeled data
This will allow a physician to confirm our predictions to increase the size of and
	balance our dataset
'''

from __future__ import division
import cPickle as pickle
import gzip
import numpy as np
import data_tweaking as dt
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import normalize
import sys
import random
'''
pickleArray[0] = train_set
	pickleArray[0][0] = x 
	pickleArray[0][1] = y
pickleArray[1] = valid_set
	pickleArray[1][0] = x
	pickleArray[1][1] = y 
pickleArray[2] = test_set
	pickleArray[2][0] = x
	pickleArray[2][1] = y
(new part for us)
pickleArray[3] = pretrain_set

Pretraining just sends in train_set[0] (aka, the x's of the training set)
We need to redefine these variables since what we want to pre-train on 
	doesn't actually have labels

sys.argv:
[0] = filename
'''
td_amt = .85

attrfile = "sle_data/final/full_attributes.txt" # equivalent to ".../goldattributes.txt"
goldDataString = "sle_data/final/golddata.txt"
goldInstancesString = "sle_data/final/full_instance_modified.txt"
golddata_matrix = dt.readSparse(attributesString=attrfile,dataString=goldDataString,instancesString=goldInstancesString)
gold_labels = dt.get_labels_according_to_data_order(dataString=goldDataString,instancesString=goldInstancesString)
golddata_matrix = dt.readSparseFewCuis(golddata_matrix)


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

rows_in_gold = golddata_matrix.shape[0] # == len(gold_labels)
start = int(9*rows_in_gold/10)
end = rows_in_gold
valid_matrix = golddata_matrix[start:end]
valid_labels = gold_labels[start:end]
golddata_matrix = np.delete(golddata_matrix,[x for x in range(start,end)],0)
gold_labels = np.delete(gold_labels,[x for x in range(start,end)],0)


def normalize(m):
	m = m.T
	m = (m - m.min())/np.ptp(m)
	return m.T
golddata_matrix = normalize(golddata_matrix)

rows_in_gold = golddata_matrix.shape[0] ## redefine since we removed test set
train_matrix = golddata_matrix
train_labels = gold_labels

test_matrix = dt.readSparse(attributesString=attrfile,dataString="sle_data/final/alldata_goldcuisonly.txt",instancesString="sle_data/final/allinstance.txt")

test_matrix = dt.readSparseFewCuis(test_matrix)
test_labels = np.asarray([0 for x in range(len(test_matrix))])

print type(train_matrix)
print "size of test:",len(test_matrix)

pickleArray = [[train_matrix,train_labels],
		[valid_matrix,valid_labels],
		[test_matrix,test_labels],
		[train_matrix]]
f = gzip.open("sle.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()
