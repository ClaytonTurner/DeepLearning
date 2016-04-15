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

td_amt = float(sys.argv[1]) # amount of data to use as training; inverse is validation
test_tenth = int(sys.argv[2]) # assuming a value 1-10
if len(sys.argv) > 3:
    rseed = rseed + sys.argv[3] - 1 # so we start with the original and just shift upward

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

# We need to cut out 25 positives and 25 negatives for the external test set
n = 50
pos_count = 0
neg_count = 0
external_test_labels = []
temp_labels = list(gold_labels)
for i in range(len(gold_labels)):
    label = gold_labels[i]
    if i == 0: # This is how we'll initialize the numpy matrix - no shaping/init issues
        external_test_matrix = golddata_matrix[0]
    if pos_count == n and neg_count == n: break # We're done
        # Increment or continue logic
    if label == "1":
        if pos_count == n: continue
        pos_count += 1
    else:
        if neg_count == n: continue
        neg_count += 1
    # If we didn't break or continue, we know we need this row in the external test
    if i == 0:
        external_test_matrix = golddata_matrix[0]
    else:
        external_test_matrix = np.concatenate((external_test_matrix,golddata_matrix[i-len(external_test_labels)]),axis=0)
    golddata_matrix = np.delete(golddata_matrix,i-len(external_test_labels),0)
    temp_labels.pop(i-len(external_test_labels))
    external_test_labels.append(label)

gold_labels = np.array(temp_labels)

print "Length of external test matrix (exp. "+str(n*2)+"):"+str(external_test_matrix.shape)
print "Length of external test labels (exp. "+str(n*2)+"):"+str(len(external_test_labels))

def normalize(m):
	m = m.T
	m = (m - m.min())/np.ptp(m)
	return m.T
golddata_matrix = normalize(golddata_matrix)


rows_in_gold = golddata_matrix.shape[0] # == len(gold_labels)
start = int((test_tenth-1)*rows_in_gold/total_folds)
end = int(test_tenth*rows_in_gold/total_folds)
test_matrix = golddata_matrix[start:end]
test_labels = gold_labels[start:end]
golddata_matrix = np.delete(golddata_matrix,[x for x in range(start,end)],0)
gold_labels = np.delete(gold_labels,[x for x in range(start,end)],0)

rows_in_gold = golddata_matrix.shape[0] ## redefine since we removed test set
train_matrix = golddata_matrix[0:(td_amt*rows_in_gold)]
train_labels = gold_labels[0:(td_amt*rows_in_gold)]
valid_matrix = golddata_matrix[(td_amt*rows_in_gold):]
valid_labels = gold_labels[(td_amt*rows_in_gold):]

clf = ExtraTreesClassifier(max_features=100)#"auto")#100) # "auto" is default max_features (sqrt(n))
train_matrix = clf.fit(train_matrix,train_labels).transform(train_matrix)
external_test_matrix = clf.transform(external_test_matrix)
valid_matrix = clf.transform(valid_matrix)
test_matrix = clf.transform(test_matrix)

print type(train_matrix)

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
