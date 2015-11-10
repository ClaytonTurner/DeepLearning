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
[3] = number_of_clusters
'''

td_amt = float(sys.argv[1]) # amount of data to use as training; inverse is validation
test_tenth = int(sys.argv[2]) # assuming a value 1-10
cluster_count = sys.argv[3]
if cluster_count not in ["100","1000","9121"]:
    print "Cluster count (argv[3]) must be 100, 1000, or 9121. Exiting..."
    sys.exit(0)

# Set up data matrix
data_filename = "../Word2VecSLE/results/occurrenceMatrix_"+cluster_count+"_classes.csv"
data_file = open(data_filename, "r")
datalines = data_file.readlines()
data_file.close()
datalines = datalines[1:] # Don't need the header
for i in range(len(datalines)):
    datalines[i] = datalines[i].split(",")
    datalines[i][-1] = datalines[i][-1].strip()

# Set up labels
goldInstancesString = "sle_data/rheumatol/gold_fixed_instance.txt"
label_file = open(goldInstancesString,"r")
labels = label_file.readlines()
label_file.close()
sid_diagnosis = {}
for label in labels:
    sid,diag = label.split("\t")
    diag = diag.strip()
    if diag == "100":
        diag = "1"
    else:
        diag = "0"
    sid_diagnosis[sid] = str(diag)

golddata = list()
gold_labels = list()
for i in range(len(datalines)):
    d_line = datalines[i]
    gold_labels.append(sid_diagnosis[d_line[0]])
    #d_line.append(sid_diagnosis[d_line.split(",")[0]])
    d_line = d_line[1:] # Don't need the subject id in the data matrix
    golddata.append(d_line)

golddata_matrix = np.array(golddata, dtype=np.float32)
print golddata_matrix
def normalize(m):
	m = m.T
	m = (m - m.min())/np.ptp(m)
	return m.T
golddata_matrix = normalize(golddata_matrix)
gold_labels = np.array(gold_labels)

clf = ExtraTreesClassifier(max_features=100) # "auto" is default max_features (sqrt(n))
golddata_matrix = clf.fit(golddata_matrix,gold_labels).transform(golddata_matrix)

# Let's shuffle since our data isn't in any kind of random order
#np.random.seed(32401) # random number
rng_state = np.random.get_state()
np.random.shuffle(golddata_matrix)
np.random.set_state(rng_state)
np.random.shuffle(gold_labels)


rows_in_gold = golddata_matrix.shape[0] # == len(gold_labels)
start = int((test_tenth-1)*rows_in_gold/10)
end = int(test_tenth*rows_in_gold/10)
test_matrix = golddata_matrix[start:end]
test_labels = gold_labels[start:end]
golddata_matrix = np.delete(golddata_matrix,[x for x in range(start,end)],0)
gold_labels = np.delete(gold_labels,[x for x in range(start,end)],0)


rows_in_gold = golddata_matrix.shape[0] ## redefine since we removed test set
train_matrix = golddata_matrix[0:(td_amt*rows_in_gold)]
train_labels = gold_labels[0:(td_amt*rows_in_gold)]
valid_matrix = golddata_matrix[(td_amt*rows_in_gold):]
valid_labels = gold_labels[(td_amt*rows_in_gold):]

print type(train_matrix)

pickleArray = [[train_matrix,train_labels],
		[valid_matrix,valid_labels],
		[test_matrix,test_labels],
		#[pretrain_matrix]]
		[train_matrix]]
f = gzip.open("sle.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()
