
# coding: utf-8

# In[5]:

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier


# In[6]:

import sys
sys.path.append('/home/andersonp/GitHub/DeepLearningMovies')
from KaggleWord2VecUtility import KaggleWord2VecUtility
#DATADIR='/home/andersonp/GitHub/DeepLearningMovies/'
DATADIR='/home/cat200/DeepLearning/word2vec/'

if len(sys.argv) < 1:
    print "Proper usage: python Kaggle_Python_Word2Vec.py <fold>"
    print "<fold> specifies which tenth of the training data is used for testing"


# In[7]:

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


# In[8]:

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,            num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


# In[9]:

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews


# In[10]:

# Read data from files
#traindata_string = "labeledTrainData.tsv"
traindata_string = "word2vec_training_without_external_labeled.csv"
#unlabeled_string = "unlabeledTrainData.tsv"
unlabeled_string = "word2vec_training_without_external_unlabeled.csv"
external_string = "word2vec_external_set.csv"
train = pd.read_csv( os.path.join(DATADIR, traindata_string), header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( os.path.join(DATADIR, unlabeled_string), header=0,  delimiter="\t", quoting=3 )
external = pd.read_csv( os.path.join(DATADIR, external_string), header=0, delimiter="\t", quoting=3 )

# In[11]:

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews "      "and %d unlabeled reviews\n" % (train["review"].size, unlabeled_train["review"].size )


# In[12]:

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[13]:

train.ix[0,]


# In[14]:
np.random.seed(31212)
def shuffle(df):
    index = list(df.index)
    np.random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
    return df

#fracTrain = 0.5
#fracTrain = 0.9
fold = int(sys.argv[1])
nSamples = train.shape[0]
#order = np.random.permutation(nSamples) # come up with a random ordering
#train1 = shuffle(train) # The saved files are already shuffled once - consistency

# This was done before this entire script so let's comment it out
'''# We need to extract the external set before the test set for CV
n = 50
pos_count = 0
neg_count = 0
external_data = []
for i in range(1,len(train1.index)+1):
    subject = train1.ix[i,]
    diagnosis = subject["sentiment"]
    print len(subject)
    if diagnosis == "0":
        if neg_count < n:
            #item = train1.pop(subject)
            del train1[subject]
            external_data.append(subject)
        else:
            continue
    else:
        if pos_count < n:
            #item = train1.pop(subject)
            del train1[subject]
            external_data.append(subject)
        else:
            continue
    if pos_count + neg_count == 2*n:
        break # because this means we have our external set
external_test = pd.Series(external_data)
print external_test
'''
# Now let's extract the test set for CV
np.random.seed(31212) # consistency with other results
order = np.arange(nSamples) # numpy handled shuffling already
#splitIndex = int(np.round(nSamples*fracTrain))
splitStart = round((float(fold-1))*(1./10.)*nSamples)
splitEnd = round((float(fold))*(1./10.)*nSamples)
#train1 = train.ix[order[:splitIndex],:] 
train1 = train.ix[:,:] # Full dataset
#test1 = train.ix[order[splitIndex:],:] # Test set
test1 = train.ix[order[int(splitStart):int(splitEnd)],:]
train1.drop(train1.index[range(int(splitStart),int(splitEnd))]) # Now let's remove the test set from training


# In[15]:

print train1.shape
print test1.shape


# # Traditional Word Averaging

# In[16]:

# ****** Split the labeled and unlabeled training sets into clean sentences
#
sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in train1["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)


# In[17]:

# ****** Set parameters and train the word2vec model
#
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
model = Word2Vec(sentences, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling, seed=1)


# In[18]:

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

#model.doesnt_match("man woman child kitchen".split())
#model.doesnt_match("france england germany berlin".split())
#model.doesnt_match("paris berlin london austria".split())
#model.most_similar("man")
#model.most_similar("queen")
#model.most_similar("awful")


# In[19]:

# ****** Create average vectors for the training and test sets
#
print "Creating average feature vecs for training reviews"

trainDataVecs = getAvgFeatureVecs( getCleanReviews(train1), model, num_features )


# In[20]:

# ****** Fit a random forest to the training set, then make predictions
#
# Fit a random forest to the training data, using 100 trees
#forest = RandomForestClassifier( n_estimators = 100 , oob_score=True)

#print "Fitting a random forest to labeled training data..."
#forest = forest.fit( trainDataVecs, train1["sentiment"] )


# In[21]:

#forest.oob_score_


# In[22]:

testDataVecs = getAvgFeatureVecs( getCleanReviews(test1), model, num_features )
externalDataVecs = getAvgFeatureVecs ( getCleanReviews(external), model, num_features )

# Test & extract results
#result = forest.predict( testDataVecs )

# Accuracy?


# In[23]:

#print np.size(np.where(result == test1["sentiment"]))*1./np.size(result)


# # Word2Vec inversion - This is what we are actually focusing on

# In[24]:

# ****** Split the labeled and unlabeled training sets into clean sentences
#
sentences_pos = []  # Initialize an empty list of sentences
sentences_neg = []  # Initialize an empty list of sentences
sentences_unlabelled = []  # Initialize an empty list of sentences

inxs_pos = np.where(train1["sentiment"] == 1)[0].tolist()
inxs_neg = np.where(train1["sentiment"] == 0)[0].tolist()

print "Parsing sentences from training set"
for inx in inxs_pos:
    review = train1["review"].iloc[inx]
    sentences_pos += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

for inx in inxs_neg:
    review = train1["review"].iloc[inx]
    sentences_neg += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)    
    
print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences_unlabelled += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)


# In[ ]:




# In[ ]:




# In[ ]:




# In[25]:

# ****** Set parameters and train the word2vec model
#
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
#print "Training Word2Vec model..."
#model_pos = Word2Vec(sentences_pos + sentences_unlabelled, workers=num_workers, \
#            size=num_features, min_count = min_word_count, \
#            window = context, sample = downsampling, seed=1, hs=1, negative=0)
#
#model_neg = Word2Vec(sentences_neg + sentences_unlabelled, workers=num_workers, \
#            size=num_features, min_count = min_word_count, \
#            window = context, sample = downsampling, seed=1,hs=1, negative=0)

from gensim.models import Word2Vec
import multiprocessing

## create a w2v learner 
basemodel = Word2Vec(
    workers=multiprocessing.cpu_count(), # use your cores
    iter=3, # iter = sweeps of SGD through the data; more is better
    hs=1, negative=0 # we only have scoring for the hierarchical softmax setup
    )
print basemodel


# In[26]:

basemodel.build_vocab(sentences) 


# In[27]:

from copy import deepcopy
models = [deepcopy(basemodel) for i in range(2)]
slist = list(sentences_pos)
models[0].train(  slist, total_examples=len(slist) )
slist = list(sentences_neg)
models[1].train(  slist, total_examples=len(slist) )


# In[28]:

"""
docprob takes two lists
* docs: a list of documents, each of which is a list of sentences
* models: the candidate word2vec models (each potential class)

it returns the array of class probabilities.  Everything is done in-memory.
"""

import pandas as pd # for quick summing within doc

def docprob(docs, mods):
    # score() takes a list [s] of sentences here; could also be a sentence generator
    sentlist = [s for d in docs for s in d]
    # the log likelihood of each sentence in this review under each w2v representation
    llhd = np.array( [ m.score(sentlist, len(sentlist)) for m in mods ] )
    # now exponentiate to get likelihoods, 
    lhd = np.exp(llhd - llhd.max(axis=0)) # subtract row max to avoid numeric overload
    # normalize across models (stars) to get sentence-star probabilities
    prob = pd.DataFrame( (lhd/lhd.sum(axis=0)).transpose() )
    # and finally average the sentence probabilities to get the review probability
    prob["doc"] = [i for i,d in enumerate(docs) for s in d]
    prob = prob.groupby("doc").mean()
    return prob


# In[32]:

probs = docprob(sentences_pos[0:2],models)


# In[33]:

probs


# In[57]:

docs = []
for review in test1["review"]:
    docs.append(KaggleWord2VecUtility.review_to_sentences(review, tokenizer))


# In[ ]:




# In[59]:
if str(fold) == "10":
    fold = "10"
else:
    fold = "0"+str(fold)

probs = docprob(docs,models)
probs.to_csv("/home/cat200/DeepLearning/results/"+str(fold)+"_p_values.txt",index=False,header=False)

# In[66]:

predictions = np.ones((probs.shape[0]))


# In[67]:

np.shape(predictions)


# In[68]:

predictions[np.where(probs.iloc[:,1] > 0.5)] = 0 # The second column is actually the negative model


# In[69]:

print predictions.shape
print test1["sentiment"].shape


# In[70]:
print "CV Accuracy"
cv_acc = np.size(np.where(predictions == test1["sentiment"]))*1./np.size(predictions)
print cv_acc

# In[ ]:

#predictions

# External set prediction
docs = []
for review in external["review"]:
    docs.append(KaggleWord2VecUtility.review_to_sentences(review,tokenizer))
probs = docprob(docs,models)
predictions = np.ones((probs.shape[0]))
np.shape(predictions)
predictions[np.where(probs.iloc[:,1] > 0.5)] = 0
print predictions.shape
print external["sentiment"].shape

print "External Accuracy"
ext_acc = np.size(np.where(predictions == external["sentiment"]))*1./np.size(predictions)
print ext_acc

f = open("fold"+str(fold)+"_accuracy.txt","w")
f.write("CV: "+str(cv_acc)+"\nExt: "+str(ext_acc))
f.close()
probs.to_csv("/home/cat200/DeepLearning/results/"+str(fold)+"_external_p_values.txt",index=False,header=False)
test1["sentiment"].to_csv("/home/cat200/DeepLearning/results/"+str(fold)+"_labels.txt",index=False,header=False)
external["sentiment"].to_csv("/home/cat200/DeepLearning/results/"+str(fold)+"_external_labels.txt",index=False,header=False)


