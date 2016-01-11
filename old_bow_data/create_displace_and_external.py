import cPickle as p
import gzip
import numpy as np

base = "sle.pkl.gz"

# We're leveraging the test data from the old BOWs data in order
#   to create our new data which incorporates an external test set
for i in range(10):
    f = gzip.open(base+i,"rb")
    a,b,c,d = p.load(f)
    f.close()
    if i == 0: # Init the array since it's numpy - much cleaner this way
        data = c[0]
        labels = c[1]
    else: # We already have a numpy array, so we need to concatenate instead
        data = np.concatenate((data,c[0]),axis=0)
        labels = np.concatenate((data,c[1]),axis=0)

# So now data and labels contains all the rheumatol data
# We need to add the obeid data


# Now we need to set up the bootstrapping to balance out the data
