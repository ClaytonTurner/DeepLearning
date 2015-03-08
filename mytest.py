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

def sortSparseData(dataString):
    #os.chdir(os.path.join("datasets","example_notes")) for testing
    os.chdir(os.path.join("sle_data")) # for sle data
    #dataString = "data.txt"
    outfile = "sorted_"+dataString
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
            sorthelper.sort(key=lambda tup: int(tup[1]))
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

def merge_all_and_gold():
    # Names of input files output by cTAKES
    attributesString = "attributes.txt"
    instancesString = "instance.txt"
    dataString = "sorted_alldata.txt" # this should be the output of sortSparseData
    g_attributesString = "goldattributes.txt"
    g_instancesString = "goldinstance.txt"
    g_dataString = "sorted_golddata.txt"
    # Start at repo name: here it's DeepLearning
    # Change directory to DeepLearning/datasets/example_notes
    os.chdir(os.path.join("datasets","example_notes"))# for example data
    #os.chdir(os.path.join("sle_data")) # for sle data
    
    # Read in data files
    attrFile = open(attributesString,"r")
    attributes = attrFile.read().splitlines()
    attrFile.close()
    instanceFile = open(instancesString,"r")
    instance = instanceFile.readlines()
    instanceFile.close()
    dataFile = open(dataString,"r")
    dataLines = dataFile.readlines()
    dataFile.close()
    g_attrFile = open(g_attributesString,"r")
    g_attributes = g_attrFile.read().splitlines()
    attrFile.close()
    g_instanceFile = open(g_instancesString,"r")
    g_instance = g_instanceFile.readlines()
    g_instanceFile.close()
    g_dataFile = open(g_dataString,"r")
    g_dataLines = g_dataFile.readlines()
    g_dataFile.close()
    final_attr = []

    cuis_in_gold_only = []
    import subprocess
    proc = subprocess.Popen(["comm", "-13", "allattributes.txt","goldattributes.txt"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    cuis_in_gold_only = out.split("\n")
    cuis_in_gold_only.append("D99999999999999") #no cui can come before this - clean way to circumvent conditional
    gold_check_key = 0

    for i in range(len(attributes)):
	curr = attributes[i]
	
	while(curr > cuis_in_gold_only[mykey]):#check if we need to insert a goldattribute first. While in case we have multiple in a row we need to do
		for j in range(len(dataLines)):#let's increment every cui that comes after that cui	
			a,b,c = dataLines[j].split("\t")
			if int(b) > i: #if we have a datapoint whom's cui index comes after curr
				b = str(int(b)+1)
			dataLines[j] = "\t".join([a,b,c])
		final_attr.append(cuis_in_gold_only[mykey])	
		gold_check_key += 1 # eases conditioal use
	if curr in g_attributes:
		cui_row_in_gold = g_attributes.index(curr)
		# so let's go in data and update the reference to the cui
		for j in range(len(g_dataLines)):
			rec = g_dataLines[j]
			gold_data_cui_num = rec.split("\t")[1]
			cui = g_attributes[gold_data_cui_num]
			if cui == curr:
				a,b,c = g_dataLines.split("\t")
				g_dataLines[j] = "\t".join([a,i,c])

		#g_attributes.remove(curr)
	else: # curr not in g_attributes	
		for j in range(len(g_dataLines)):
			a,b,c = g_dataLines[j].split("\t")
			if int(b) > i: #if we have a datapoint whom's cui index comes after curr
				b = str(int(b)+1)
			g_dataLines[j] = "\t".join([a,b,c])

	final_attr.append(curr)
    m_attr_out = open("merged_attributes.txt","w")
    m_attr_out.write("\n".join(final_attr))
    m_attr_out.close()
    m_data_out = open("","w")
    m_data_out.write("\n".join(dataLines))
    m_data_out.close()
    m_gdata_out = open("","w")
    m_gdata_out.write("\n".join(g_dataLines)
    m_gdata_out.close()
		

def readSparse():
    # Ref: http://www.deeplearning.net/software/theano/library/sparse/index.html#libdoc-sparse
    
     # Names of input files output by cTAKES
    attributesString = "attributes.txt"
    instancesString = "instance.txt"
    dataString = "outdata.txt" # this should be the output of sortSparseData
    
    # Start at repo name: here it's DeepLearning
    # Change directory to DeepLearning/datasets/example_notes
    os.chdir(os.path.join("datasets","example_notes"))# for example data
    #os.chdir(os.path.join("sle_data")) # for sle data
    
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
    rows = len(instance)-1
    cols = len(attributes)

    dataArray = []
    indicesArray = [] 
    subjectidArray = []
    subjectctr = -1
    indptrctr = 0
    indptrArray = []
    for line in dataLines:
        subjectid,cui,cuicount = line.split("\t")
        cuicount = cuicount.strip()
        if subjectid not in subjectidArray:
            # so we have finished a patient
            subjectidArray.append(subjectid)
            subjectctr += 1
            indptrArray.append(indptrctr)
        dataArray.append(float(cuicount))
        #indicesArray.append(subjectctr)
        indicesArray.append(int(cui)-1)
        indptrctr += 1
    indptrArray.append(indptrctr)

        
    data = np.asarray(dataArray)
    indices = np.asarray(indicesArray)

    # we need to sort by cuis so we can figure out indptr
    # since it deals with columns 
    # not efficient but it's necessary

    indptr = np.asarray(indptrArray)
    print "data: "
    print data
    print "indices: "
    print indices
    print "indptr: "
    print indptr

    
    
    
    m = sp.csr_matrix((data,indices,indptr), shape=(rows,cols))
    print m.toarray()
   
sortSparseData("alldata.txt")
sortSparseData("golddata.txt") 
#readSparse()
