import cPickle as pickle
import gzip
import numpy as np

pickleArray [] #going to be 3 items
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

Pretraining just sends in train_set[0] (aka, the x's of the training set)
We need to redefine these variables since what we want to pre-train on 
	doesn't actually have labels
'''


# Pickle and zip to binary data
f = open("sle.pkl","w")
pickle.dump(pickleArray,f)
f.close()
f = open("sle.pkl","rb")
f_out = gzip.open("sle.pkl.gz","wb")
f_out.writelines(f)
f.close()
f_out.close()

