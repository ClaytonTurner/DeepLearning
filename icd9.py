import numpy as np

f = open("sle_data/icd9_preds.csv","r")
lines = f.readlines()
f.close()

data = []
for line in lines:
    new_line = line.split(",")
    new_line[-1] = new_line[-1].strip()
    data.append(new_line)

matrix = np.asarray(data)

rseed = 31212

np.random.seed(rseed)

np.random.shuffle(matrix)

# Doing this counter really only so I can keep track of what
# ids would be in the external set and which aren't
# Clearly we don't need to repeat the data for it's own sake
counter = 0
counter_limit = 18
for i in range(len(matrix)):
    if matrix[i][2] == "100":
        counter += 1
        matrix = np.concatenate((matrix,np.array([matrix[i]])),axis=0)
    if counter == counter_limit: break

np.random.seed(rseed+1)

np.random.shuffle(matrix)

n = 25
pos_count = 0
neg_count = 0
external_data = []
external_test_labels = [] # Only using this to mimic our indexing on concat/delete
for i in range(len(matrix)):
    label = matrix[i][2]
    if i == 0:
        external_test_matrix = matrix[0]
    if pos_count == n and neg_count == n: break
    if label == "100":
        if pos_count == n: continue
        pos_count += 1
    else:
        if neg_count == n: continue
        neg_count += 1
    if i == 0:
        external_test_matrix = matrix[0]
    else:
        external_test_matrix = np.concatenate((external_test_matrix,matrix[i-len(external_test_labels)]),axis=0)
    matrix = np.delete(matrix,i-len(external_test_labels),0)
    external_test_labels.append(label)
print external_test_matrix
# Now let's go ahead and do the predictions since our data is in the correct order
# We aren't really doing CV since the result would be the same as it is icd9 based
correct = 0
incorrect = 0
for i in range(len(matrix)):
    icd9 = matrix[i][1]
    label = matrix[i][2]
    if int(icd9) > 0:
        if label == "100":
            correct += 1
        else:
            incorrect += 1
    else:
        if label == "100":
            incorrect += 1
        else:
            correct += 1
print "Accuracy (CV) for icd9 is: "+str((float(correct))/(float(correct)+float(incorrect)))

correct = 0
incorrect = 0
for i in range(0,len(external_test_matrix),3):
    # Something weird is happening where the matrix gets flattened
    # It's an easy pattern so I just ran with it
    #icd9 = external_test_matrix[i][1]
    #label = external_test_matrix[i][2]
    icd9 = external_test_matrix[i+1]
    label = external_test_matrix[i+2]
    #print icd9,label # Manual check if desired
    if int(icd9) >0:
        if label == "100":
            correct += 1
        else:
            incorrect += 1
    else:
        if label == "100":
            incorrect += 1
        else:
            correct += 1
print "Accuracy (Ext) for icd9 is: "+str((float(correct))/(float(correct)+float(incorrect)))
