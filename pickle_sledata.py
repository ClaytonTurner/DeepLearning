import cPickle as pickle
import gzip
import numpy as np
import data_tweaking as dt
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import normalize
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

#unlabeled_data_file = open("sle_data/alldata_gold_cuis_only.txt","r")
#unlabeled_data_lines = unlabeled_data_file.readlines()
#unlabeled_data_file.close()
#labeled_data_file = open("sle_data/sorted_golddata.txt","r")
#labeled_data_lines = labeled_data_file.readlines()
#labeled_data_file.close()

attrfile = "sle_data/goldattributes.txt"
goldDataString = "sle_data/sorted_golddata.txt"
goldInstancesString = "sle_data/goldinstance.txt"
golddata_matrix = dt.readSparse(attributesString=attrfile,dataString=goldDataString,instancesString=goldInstancesString)
gold_labels = dt.get_labels_according_to_data_order(dataString=goldDataString,instancesString=goldInstancesString)

for i in range(len(gold_labels)):
	if gold_labels[i] == "100":
		gold_labels[i] = "1"
	elif gold_labels[i] == "-100":
		gold_labels[i] = "0"

golddata_matrix = golddata_matrix.todense()

pca = RandomizedPCA(n_components=3)
pca.fit(golddata_matrix)
golddata_matrix = pca.transform(golddata_matrix)

variance_explained = []
for i in range(len(pca.explained_variance_ratio_)):
	if i == 0:
		variance_explained.append(pca.explained_variance_ratio_[i])
	else:
		variance_explained.append(pca.explained_variance_ratio_[i]+variance_explained[i-1])

print "pca explained variance: "
print variance_explained

def normalize(m):
	m = m.T
	m = (m - m.min())/np.ptp(m)
	return m.T
#golddata_matrix = normalize(golddata_matrix, axis=0).ravel()
golddata_matrix = normalize(golddata_matrix)

rows_in_gold = golddata_matrix.shape[0] ## == len(gold_labels)
train_matrix = golddata_matrix[0:(rows_in_gold/3)]
train_labels = gold_labels[0:(rows_in_gold/3)]
valid_matrix = golddata_matrix[(rows_in_gold/3):(2*rows_in_gold/3)]
valid_labels = gold_labels[(rows_in_gold/3):(2*rows_in_gold/3)]
test_matrix = golddata_matrix[(2*rows_in_gold/3):rows_in_gold]
test_labels = gold_labels[(2*rows_in_gold/3):rows_in_gold]
pretrain_matrix = dt.readSparse(attributesString=attrfile,dataString="sle_data/alldata_gold_cuis_only.txt",instancesString="sle_data/allinstance_corrected.txt")

pretrain_matrix = pretrain_matrix.todense()
pretrain_matrix = pca.transform(pretrain_matrix)
pretrain_matrix = normalize(pretrain_matrix)

pickleArray = [[train_matrix,train_labels],
		[valid_matrix,valid_labels],
		[test_matrix,test_labels],
		[pretrain_matrix]]
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
