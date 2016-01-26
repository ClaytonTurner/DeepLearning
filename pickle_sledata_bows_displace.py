from __future__ import division
import cPickle as pickle
import gzip
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.ensemble import ExtraTreesClassifier
import sys

rseed = 31212

td_amt = float(sys.argv[1])
test_tenth = int(sys.argv[2])

f = gzip.open("sle.bows_full.pkl.gz","rb")
data,labels = pickle.load(f)
f.close()

def my_shuffle(m,l):
    rng_state = np.random.get_state()
    np.random.shuffle(m)
    np.random.set_state(rng_state)
    np.random.shuffle(l)

data = np.asarray(data)
labels = np.array(labels)
my_shuffle(data,labels)
print "data shape: "+ str(data.shape)
print "data[0] shape: "+ str(data[0].shape)
print "labels shape: "+ str(labels.shape)

# Bootstrap data
counter = 0
counter_limit = 18
for i in range(len(labels)):
    if labels[i] == "1":
        counter += 1
        print "counter: "+str(counter)
        data = np.concatenate((data,np.array([data[i]])),axis=0)
    if counter == counter_limit: break
temp_list = list(labels)
temp_list.extend(["1"]*counter_limit)
labels = np.array(temp_list)

np.random.seed(rseed+1) # Consistency with other pickle file
my_shuffle(data,labels)

n = 25
pos_count = 0
neg_count = 0
external_test_labels = []
temp_labels = list(labels)
for i in range(len(labels)):
    label = labels[i]
    print "i: "+str(i)+" | pos: "+str(pos_count)+" | neg: "+str(neg_count)
    if i == 0:
        external_test_matrix = np.array([data[0]])
    if pos_count == n and neg_count == n: break
    if label == "1":
        if pos_count == n: continue
        pos_count += 1
    else:
        if neg_count == n: continue
        neg_count += 1
    if i > 0:
        external_test_matrix = np.concatenate((external_test_matrix,np.array([data[i-len(external_test_labels)]])),axis=0)
    data = np.delete(data,i-len(external_test_labels),0)
    temp_labels.pop(i-len(external_test_labels))
    external_test_labels.append(label)

labels = np.array(temp_labels)

print "Length of external test matrix: "+str(external_test_matrix.shape)
print "Length of external test labels: "+str(len(external_test_labels))

def normalize(m):
    m = m.T
    m = (m - m.min())/np.ptp(m)
    return m.T

data = normalize(data)

rows_in_data = data.shape[0]
start = int((test_tenth-1)*rows_in_data/10)
end = int(test_tenth*rows_in_data/10)
test_matrix = data[start:end]
test_labels = labels[start:end]
data = np.delete(data,[x for x in range(start,end)],0)
labels = np.delete(labels,[x for x in range(start,end)],0)

rows_in_data = data.shape[0]
train_matrix = data[0:(td_amt*rows_in_data)]
train_labels = labels[0:(td_amt*rows_in_data)]
valid_matrix = data[(td_amt*rows_in_data):]
valid_labels = labels[(td_amt*rows_in_data):]

clf = ExtraTreesClassifier(max_features="auto")
train_matrix = clf.fit(train_matrix,train_labels).transform(train_matrix)
external_test_matrix = clf.transform(external_test_matrix)
valid_matrix = clf.transform(valid_matrix)
test_matrix = clf.transform(test_matrix)

print type(train_matrix)

pickleArray = [[train_matrix,train_labels],
        [valid_matrix,valid_labels],
        [test_matrix,test_labels],
        [train_matrix]]

f = gzip.open("sle.pkl.gz","wb")
pickle.dump(pickleArray,f)
f.close()

f = gzip.open("external_test.pkl.gz","wb")
pickle.dump((external_test_matrix,np.array(external_test_labels)),f)
f.close()
