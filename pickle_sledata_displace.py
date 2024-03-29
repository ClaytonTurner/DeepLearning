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
#total_folds = 10
iteration = int(sys.argv[1])
fold = int(sys.argv[2])
BOWs = False
if len(sys.argv) > 3:
    # Then we're doing BOWs
    BOWs = True

#import pdb
#pdb.set_trace()

if BOWs:
    #f = gzip.open("sle.bows_full.pkl.gz","rb")
    f = gzip.open("sle.bows_bootstrapped.pkl.gz","rb")
    golddata_matrix,gold_labels = pickle.load(f)
    f.close()
    golddata_matrix = np.asarray(golddata_matrix)
    print golddata_matrix.shape
    print len(gold_labels)
else: # So we're doing CUIs
    golddata_matrix=np.loadtxt("golddata_matrix.csv")
    gold_labels=np.loadtxt("gold_labels.csv")

def normalize(m):
    m = m.T
    m = (m - m.min())/np.ptp(m)
    return m.T

golddata_matrix = normalize(golddata_matrix)



external_test_indices = list(np.loadtxt("indices/test_indices_"+str(iteration)+".csv",dtype="int"))
train_indices = list(np.loadtxt("indices/internal_train_indices_"+str(iteration)+"_"+str(fold)+".csv",dtype="int"))
test_indices = list(np.loadtxt("indices/validation_indices_"+str(iteration)+"_"+str(fold)+".csv",dtype="int"))
test_matrix = golddata_matrix[test_indices]
external_test_matrix = golddata_matrix[external_test_indices]
test_labels = gold_labels[test_indices]
external_test_labels = gold_labels[external_test_indices]

td_amt = .85
train_matrix = golddata_matrix[train_indices[0:int(td_amt*len(train_indices))]]
train_labels = gold_labels[train_indices[0:int(td_amt*len(train_indices))]]
valid_matrix = golddata_matrix[train_indices[int(td_amt*len(train_indices)):]]
valid_labels = gold_labels[train_indices[int(td_amt*len(train_indices)):]]

clf = ExtraTreesClassifier(max_features=100)#"auto")#100) # "auto" is default max_features (sqrt(n))
train_matrix = clf.fit(train_matrix,train_labels).transform(train_matrix)
external_test_matrix = clf.transform(external_test_matrix)
valid_matrix = clf.transform(valid_matrix)
test_matrix = clf.transform(test_matrix)

print type(train_matrix)
print train_matrix.shape
print train_labels.shape
print valid_matrix.shape
print valid_labels.shape
print test_matrix.shape
print test_labels.shape

pickleArray = [[train_matrix,train_labels],
		[valid_matrix,valid_labels],
		[test_matrix,test_labels],
		#[pretrain_matrix]]
		[train_matrix]]
f = gzip.open("sle.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()

f = gzip.open("external_test.pkl.gz","wb")
pickle.dump((external_test_matrix,np.array(external_test_labels)),f)
f.close()

'''
f = gzip.open("sle.pkl.gz","rb")
a,b,c,d = pickle.load(f)
out = open("data.out","w")
out.write(d)
out.close()
f.close()
'''
