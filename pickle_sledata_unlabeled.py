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
from sklearn.ensemble import ExtraTreesClassifier
import sys
import random


td_amt = .85

def normalize(m):
	m = m.T
	m = (m - m.min())/np.ptp(m)
	return m.T

#attrfile = "sle_data/final/full_attributes.txt" # equivalent to ".../goldattributes.txt"
attrfile = "sle_data/rheumatol/attributes.txt"
#goldDataString = "sle_data/final/golddata.txt"
goldDataString = "sle_data/rheumatol/gold_data.txt"
#goldInstancesString = "sle_data/final/full_instance_modified.txt"
goldInstancesString = "sle_data/rheumatol/gold_fixed_instance.txt"
golddata_matrix = dt.readSparse(attributesString=attrfile,dataString=goldDataString,instancesString=goldInstancesString)
gold_labels = dt.get_labels_according_to_data_order(dataString=goldDataString,instancesString=goldInstancesString)
#golddata_matrix = dt.readSparseFewCuis(golddata_matrix)


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

# Find out what's training now, length-wise so we can split back off after normalization
training_rows = golddata_matrix.shape[0] 
# Define here so we can do normalization
test_matrix = dt.readSparse(attributesString=attrfile,dataString="sle_data/rheumatol/for_creating_test_data/final_data.txt",instancesString="sle_data/rheumatol/for_creating_test_data/final_instance.txt")
test_matrix = test_matrix.todense()

norm_matrix = normalize(np.concatenate((golddata_matrix,test_matrix),axis=0))
golddata_matrix = norm_matrix
test_matrix = norm_matrix[training_rows:]
golddata_matrix = np.delete(golddata_matrix,
	[x for x in range(training_rows,golddata_matrix.shape[0])],0)

rows_in_gold = golddata_matrix.shape[0] # == len(gold_labels)
start = int(9*rows_in_gold/10)
end = rows_in_gold
valid_matrix = golddata_matrix[start:end]
valid_labels = gold_labels[start:end]
golddata_matrix = np.delete(golddata_matrix,[x for x in range(start,end)],0)
gold_labels = np.delete(gold_labels,[x for x in range(start,end)],0)

# Fit to training
clf = ExtraTreesClassifier(max_features="auto")
golddata_matrix = clf.fit(golddata_matrix,gold_labels).transform(golddata_matrix)

# Transform validation
valid_matrix = clf.transform(valid_matrix)


rows_in_gold = golddata_matrix.shape[0] ## redefine since we removed test set
train_matrix = golddata_matrix
train_labels = gold_labels

test_labels = np.asarray([0 for x in range(len(test_matrix))])
test_matrix = clf.transform(test_matrix)


# ALTER THIS FOR GETTING NEW PATIENTS. OUT OF MEMORY ERRORS OTHERWISE. ugh
test_matrix = np.delete(test_matrix,[x for x in range(2000,test_matrix.shape[0])],0)
test_labels = np.delete(test_labels,[x for x in range(2000,test_matrix.shape[0])],0)

#print "size of test:",len(test_matrix)
print "Shapes for: train, valid, test"
print train_matrix.shape
print valid_matrix.shape
print test_matrix.shape

pickleArray = [[train_matrix,train_labels],
		[valid_matrix,valid_labels],
		[test_matrix,test_labels],
		[train_matrix]]

f = gzip.open("sle.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()
'''
pickleArray = [train_matrix,train_labels]
f = gzip.open("newsle1.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()

pickleArray = [valid_matrix,valid_labels]
f = gzip.open("newsle2.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()

pickleArray = [test_matrix,test_labels]
f = gzip.open("newsle3.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()
'''

#pickleArray2 = [test_matrix,test_labels]
#f = gzip.open("sle.pkl2.gz","wb")
#pickle.dump(pickleArray2,f)
#f.close()
