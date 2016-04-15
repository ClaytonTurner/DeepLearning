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
'''
'''
sys.argv:
[0] = filename
[1] = where to cut off training set
[2] = which tenth of data to use for testing
[3] = pca components
'''
#do_pca = len(sys.argv) > 3
#if do_pca:
#	n_components = int(sys.argv[3])
#else:
#	n_components = 20

rseed = 31212 # only comment out when generating confidence intervals
total_folds = 10

#rseed = rseed + sys.argv[3] - 1 # so we start with the original and just shift upward

attrfile = "sle_data/rheumatol_extended/attributes.txt"
goldDataString = "sle_data/rheumatol_extended/gold_data.txt"
goldInstancesString = "sle_data/rheumatol_extended/gold_fixed_instance.txt"
golddata_matrix = dt.readSparse(attributesString=attrfile,dataString=goldDataString,instancesString=goldInstancesString)
gold_labels = dt.get_labels_according_to_data_order(dataString=goldDataString,instancesString=goldInstancesString)
golddata_matrix = golddata_matrix.todense()

# This is for incorporating data we initially used as an external set - but now we want to randomly create a balanced external test set - we will bootstrap training samples as needed from the non-external set. The name "for_creating_test_data" is an artifact but let's leave it for sanity's sake until we play cleanup
obeidDataString = "sle_data/rheumatol_extended/for_creating_test_data/final_data_obeid_labeled.txt"
obeidInstancesString = "sle_data/rheumatol_extended/for_creating_test_data/final_instance_obeid_labeled.txt"
obeid_matrix = dt.readSparse(attributesString=attrfile,dataString=obeidDataString,instancesString=obeidInstancesString)
obeid_labels = dt.get_labels_according_to_data_order(dataString=obeidDataString,instancesString=obeidInstancesString)
obeid_matrix = obeid_matrix.todense() # Though this matrix is small, we still used csr
golddata_matrix = np.concatenate((golddata_matrix,obeid_matrix),axis=0)
temp_list = list(gold_labels)
temp_list.extend(list(obeid_labels))
gold_labels = np.array(temp_list) # Add our new labels - order is preserved

# Let's shuffle since our data isn't in any kind of random order
np.random.seed(rseed) # random number - we need this for bootstrapping consistency
def my_shuffle(m,l):
    rng_state = np.random.get_state()
    np.random.shuffle(m)
    np.random.set_state(rng_state)
    np.random.shuffle(l)

my_shuffle(golddata_matrix,gold_labels)

# I know we need to bootstrap 18 positive samples so let's just hardcode for now - generalize later if needed
counter = 0
counter_limit = 18
# We shuffled before this so we'll just take the first 18 that fit
for i in range(len(gold_labels)):
    if gold_labels[i] == "100":
        counter += 1
        golddata_matrix = np.concatenate((golddata_matrix,golddata_matrix[i]),axis=0)
    if counter == counter_limit: break

temp_list = list(gold_labels)
temp_list.extend(["100"]*counter_limit)
gold_labels = np.array(temp_list)

for i in range(len(gold_labels)):
	if i == 0: # Old bug fix - if it ain't broke...
		if gold_labels[i] == "100":
			gold_labels[i] = "1"
		else:
			gold_labels[i] = "0"
	elif gold_labels[i] == "100":# for fixing the labels AND bootstrapping a negative
		gold_labels[i] = "1"
	elif gold_labels[i] == "-100":# for fixing the labels
		gold_labels[i] = "0"

np.random.seed(rseed+1) # Keep external set consistent
# We added bootstraps to the end of the matrix so we need to shuffle to get rid of bias
my_shuffle(golddata_matrix,gold_labels)

np.savetxt("golddata_matrix.csv",golddata_matrix)
np.savetxt("gold_labels.csv",np.asarray(gold_labels),fmt="%s")

from copy import deepcopy
n_times = 20
test_set_size = 100
internal_folds = 5
for i in range(n_times):
    indices = np.random.permutation(len(gold_labels))
    test_indices = indices[:100]
    train_indices = indices[100:]
    fold_indices = np.split(train_indices,internal_folds)
    np.savetxt("indices/test_indices_"+str(i)+".csv",test_indices,fmt="%d")
    np.savetxt("indices/train_indices_"+str(i)+".csv",train_indices,fmt="%d")
    for j in range(internal_folds):
        validation_indices = fold_indices[j]
        fold_indices_copy = deepcopy(fold_indices)
        fold_indices_copy.pop(j)
        internal_train_indices = list(np.asarray(fold_indices_copy).flatten())
        np.savetxt("indices/validation_indices_"+str(i)+"_"+str(j)+".csv",validation_indices,fmt="%d")
        np.savetxt("indices/internal_train_indices_"+str(i)+"_"+str(j)+".csv",internal_train_indices,fmt="%d")


