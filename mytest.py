import theano
import numpy as np
import scipy.sparse as sp
from theano import sparse
import denoisingAutoencoder as dA
import SdA
import os
import sys

def run():
    #os.chdir("C:\\Users\\Clayton\\DeepLearning\\datasets")
    print os.getcwd()
    
    '''
    defaults for denoising autoencoder test
    learning_rate=0.1
    training_epochs=15
    dataset='mnist.pk1.gz'
    batch_size=20
    output_folder=dA_plots
    '''
    dA.test_dA()
    
    '''
    defaults for Stacked denoising auto-encoder class
    finetune_lr=0.1
    pretraining_epochs=15
    pretrain_lr=0.001
    training_epochs=1000
    dataset='mnist.pkl.gz'
    batch_size=1
    '''
    SdA.test_SdA()

def readSparse():
    # Ref: http://www.deeplearning.net/software/theano/library/sparse/index.html#libdoc-sparse
    
    # Names of input files output by cTAKES
    attributesString = "attributes.txt"
    instancesString = "instance.txt"
    dataString = "data.txt"
    
    # Start at repo name: here it's DeepLearning
    # Change directory to DeepLearning/datasets/example_notes
    os.chdir(os.path.join("datasets","example_notes"))
    
    # Read in data files
    attrFile = open(attributesString,"r")
    attributes = attrFile.readlines()
    attrFile.close()
    instanceFile = open(instancesString,"r")
    instance = instanceFile.readlines()
    instanceFile.close()
    dataFile = open(dataString,"r")
    dataLines = dataFile.readlines()
    dataFile.close()
    
    # 1 to 1 relationship with data and indices
    # data = [a,b,c]
    # indices = [x,y,z]
    # datapoint a is on row x
    dataArray = []
    data = np.asarray(dataArray)
    indicesArray = [] 
    indices = np.asarray(indicesArray)
    indptrArray = []
    indptr = np.asarray(indptrArray)
    
    # For shape of array
    rows = len(instance)
    cols = len(attributes)
    
    m = sp.csc_matrix((data,indices,indptr), shape=(rows,cols))
    print m.toarray()
    
readSparse()