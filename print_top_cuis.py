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

np.random.seed(32401)

np.set_printoptions(threshold=np.nan) # For printing entire array

attrfile = "sle_data/rheumatol/attributes.txt"
goldDataString = "sle_data/rheumatol/gold_data.txt"
goldInstancesString = "sle_data/rheumatol/gold_fixed_instance.txt"
golddata_matrix = dt.readSparse(attributesString=attrfile,dataString=goldDataString,instancesString=goldInstancesString)
gold_labels = dt.get_labels_according_to_data_order(dataString=goldDataString,instancesString=goldInstancesString)

infile = open(attrfile,"r")
attrs = infile.readlines()
infile.close()
attr_string = str()
for i in range(len(attrs)):
    if i == 0:
        attr_string += "0.0 "
    else:
        attr_string += attrs[i][1:].strip()+" "
attr_string = attr_string[:-2]+"\n"

golddata_matrix = golddata_matrix.todense()

clf = ExtraTreesClassifier()
golddata_matrix = clf.fit(golddata_matrix,gold_labels).transform(golddata_matrix)
#attrs = clf.transform(attr_string.split(" "))
attrs = attr_string.split(" ")

import copy
feature_imp = clf.feature_importances_.copy()

for i in range(10):
    item_index = np.where(feature_imp==max(feature_imp))
    for item in item_index:
        print item
        print attrs[item]
        feature_imp = np.delete(feature_imp,item)
    for item in item_index:
        attrs.pop(item)
