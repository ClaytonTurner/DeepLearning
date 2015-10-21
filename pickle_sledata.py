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
do_pca = len(sys.argv) > 3
if do_pca:
	n_components = int(sys.argv[3])
else:
	n_components = 20

td_amt = float(sys.argv[1]) # amount of data to use as training; inverse is validation
test_tenth = int(sys.argv[2]) # assuming a value 1-10



#attrfile = "sle_data/goldattributes.txt"
#attrfile = "sle_data/final/goldattributes.txt" # equivalent to ".../goldattributes.txt"
attrfile = "sle_data/rheumatol/attributes.txt"
#goldDataString = "sle_data/sorted_golddata.txt"
#goldDataString = "sle_data/final/golddata.txt"
#goldDataString = "sle_data/final/golddata_with_new_labels.txt"
goldDataString = "sle_data/rheumatol/gold_data.txt"
#goldInstancesString = "sle_data/goldinstance.txt"
#goldInstancesString = "sle_data/final/full_instance_modified.txt"
#goldInstancesString = "sle_data/final/more_labels_instance.txt"
goldInstancesString = "sle_data/rheumatol/gold_fixed_instance.txt"
golddata_matrix = dt.readSparse(attributesString=attrfile,dataString=goldDataString,instancesString=goldInstancesString)
gold_labels = dt.get_labels_according_to_data_order(dataString=goldDataString,instancesString=goldInstancesString)
#golddata_matrix = dt.readSparseFewCuis(golddata_matrix)

#pretrain_matrix_list = []
'''
# This code is for bootstrapping a pretraining matrix from the training data rather than from a dedicated pretrain set
switch = False
for i in range(len(gold_labels)):
	if not switch:
		if gold_labels[i] == "-100":
			bootstrap_pretrain = golddata_matrix[i]
			switch = True
	elif gold_labels[i] == "-100":
		bootstrap_pretrain = np.concatenate([bootstrap_pretrain,golddata_matrix[i]])
'''
for i in range(len(gold_labels)):
	if i == 0:
		#pretrain_matrix = golddata_matrix[i]
		if gold_labels[i] == "100":
			gold_labels[i] = "1"
		else:
			gold_labels[i] = "0"
	elif gold_labels[i] == "100":# for fixing the labels AND bootstrapping a negative
		#pretrain_matrix = np.concatenate([pretrain_matrix,golddata_matrix[i]])
		#pretrain_matrix = np.concatenate([pretrain_matrix,bootstrap_pretrain[random.randint(0,len(bootstrap_pretrain)-1)]]) # add a random negative pretrain row
		gold_labels[i] = "1"
	elif gold_labels[i] == "-100":# for fixing the labels
		gold_labels[i] = "0"

#pretrain_matrix = np.asmatrix(pretrain_matrix_list)
# comment out when using FewCuis method
golddata_matrix = golddata_matrix.todense()
#pca = RandomizedPCA(n_components)
#pca.fit(golddata_matrix)
#golddata_matrix = pca.transform(golddata_matrix)
clf = ExtraTreesClassifier(max_features=100) # "auto" is default max_features (sqrt(n))
golddata_matrix = clf.fit(golddata_matrix,gold_labels).transform(golddata_matrix)

# Let's shuffle since our data isn't in any kind of random order
# Doing this here so we can alternate with PCA and FewCUIs consistently
np.random.seed(32401) # random number
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


'''
variance_explained = []
for i in range(len(pca.explained_variance_ratio_)):
	if i == 0:
		variance_explained.append(pca.explained_variance_ratio_[i])
	else:
		variance_explained.append(pca.explained_variance_ratio_[i]+variance_explained[i-1])

print "pca explained variance: "
print variance_explained
'''
def normalize(m):
	m = m.T
	m = (m - m.min())/np.ptp(m)
	return m.T
golddata_matrix = normalize(golddata_matrix)

rows_in_gold = golddata_matrix.shape[0] ## redefine since we removed test set
train_matrix = golddata_matrix[0:(td_amt*rows_in_gold)]
train_labels = gold_labels[0:(td_amt*rows_in_gold)]
#valid_matrix = golddata_matrix[(rows_in_gold/3):(2*rows_in_gold/3)]
valid_matrix = golddata_matrix[(td_amt*rows_in_gold):]
#valid_labels = gold_labels[(rows_in_gold/3):(2*rows_in_gold/3)]
valid_labels = gold_labels[(td_amt*rows_in_gold):]
#test_matrix = golddata_matrix[(2*rows_in_gold/3):rows_in_gold]
#test_matrix = valid_matrix
#test_labels = gold_labels[(2*rows_in_gold/3):rows_in_gold]
#test_labels = valid_labels

#pretrain_matrix = dt.readSparse(attributesString=attrfile,dataString="sle_data/alldata_gold_cuis_only.txt",instancesString="sle_data/allinstance_corrected.txt")

# Not using pretraing matrix currently so let's comment it out
#pretrain_matrix = dt.readSparse(attributesString=attrfile,dataString="sle_data/final/alldata_goldcuisonly.txt",instancesString="sle_data/final/allinstance.txt")

#pretrain_matrix = dt.readSparseFewCuis(pretrain_matrix)

# comment out when use FewCuis method
#pretrain_matrix = pretrain_matrix.todense()
#pretrain_matrix = pca.transform(pretrain_matrix)
#pretrain_matrix = normalize(pretrain_matrix)

#Add our training data to the pretraining
##pretrain_matrix = np.concatenate((pretrain_matrix,train_matrix))

#print type(pretrain_matrix)
print type(train_matrix)


pickleArray = [[train_matrix,train_labels],
		[valid_matrix,valid_labels],
		[test_matrix,test_labels],
		#[pretrain_matrix]]
		[train_matrix]]
f = gzip.open("sle.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()

'''
f = gzip.open("sle.pkl.gz","rb")
a,b,c,d = pickle.load(f)
out = open("data.out","w")
out.write(d)
out.close()
f.close()
'''
