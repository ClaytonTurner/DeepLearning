import cPickle as pickle
import gzip
import numpy as np
import data_tweaking as dt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import sys

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

pca = PCA()
pca.fit(golddata_matrix.T)

variance_explained = []
for i in range(len(pca.explained_variance_ratio_)):
	if i == 0:
		variance_explained.append(pca.explained_variance_ratio_[i])
	else:
		variance_explained.append(pca.explained_variance_ratio_[i]+variance_explained[i-1])

print "pca explained variance: "
print variance_explained
print "\n" # ensure buffer printout
