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

def sortSparseData():
    os.chdir(os.path.join("datasets","example_notes"))
    dataString = "data.txt"
    outfile = "outdata.txt"
    dataFile = open(dataString,"r")
    dataLines = dataFile.readlines()
    dataFile.close()
    dataString = ''
    indicesArray = []
    sorthelper = []

    for line in dataLines:
        subjectid,cui,cuicount = line.split("\t")
        cuicount = cuicount.strip()
        if subjectid not in indicesArray:
            sorthelper.sort(key=lambda tup: tup[1])
            string = "\n".join(str(s) for s in sorthelper)
            string = string+'\n'
            dataString += string
            #print sorthelper

            sorthelper = []
        sorthelper.append([subjectid,cui,cuicount])
        indicesArray.append(subjectid)
    import re
    dataString = re.sub("\[","",dataString)
    dataString = re.sub("\]","",dataString)
    dataString = re.sub("\'","",dataString)
    dataString = re.sub(", ","\t",dataString)
    out = open(outfile,"w")
    # indexing gets rid of new line at beginning
    # strip gets rid of new line at end
    out.write(dataString[1:].strip())
    out.close()

def readSparse():
    # Ref: http://www.deeplearning.net/software/theano/library/sparse/index.html#libdoc-sparse
    
    # Names of input files output by cTAKES
    attributesString = "attributes.txt"
    instancesString = "instance.txt"
    dataString = "outdata.txt" # this should be the output of sortSparseData
    
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
    # indptr is a slice array for columns

    # Row = Patient

    # For shape of array
    rows = len(instance)
    cols = len(attributes)

    dataArray = []
    indicesArray = [] 
    for line in dataLines:
        subjectid,cui,cuicount = line.split("\t")
        cuicount = cuicount.strip()
        if subjectid not in indicesArray:
            # so we have finished a patient
            indicesArray.append(subjectid)
        dataArray.append(cuicount)
    data = np.asarray(dataArray)
    indices = np.asarray(indicesArray)

    # we need to sort by cuis so we can figure out indptr
    # since it deals with columns 
    # not efficient but it's necessary
    indptrArray = []
    import operator
    newDataList = []
    for line in dataLines:
        tup = line.split("\t")
        tup[2] = tup[2].strip()
        newDataList.append(tup)
    newDataList.sort(key = operator.itemgetter(1,0))
    print newDataList

    indptr = np.asarray(indptrArray)


    
    
    
    m = sp.csc_matrix((data,indices,indptr), shape=(rows,cols))
    print m.toarray()
   
#sortSparseData() 
readSparse()