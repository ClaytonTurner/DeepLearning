__author__ = "caturner3"

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interp

#infile = "clean_rheumatol_labeled.csv"
infile = "clean_rheumatol_labeled_patient_centric.csv"
def delimit():
	# Only run this once for each csv
	# Afterward, append a header for the pandas dataframe
	f = open(infile,"r")
	lines = f.readlines()
	f.close()
	new_lines = []
	for i in xrange(len(lines)):
		sid_note = lines[i].split('\t', 1)#.rsplit('\t', 1)
		note_diag = sid_note[1].rsplit('\t', 1)
		sid_note_diag = [sid_note[0], note_diag[0], note_diag[1]]
		new_lines.append("\tdelimiter\t".join(sid_note_diag))
	f = open("clean_"+infile,"w")
	f.writelines(new_lines)
	f.close()

def delimit_patient_centric():
	f = open(infile,"r")
	lines = f.readlines()
	f.close()
	# So we want to combine subject id's notes together
	# Dictionary sounds like a dope way to do this
	sid_note_dict = {}
	sid_diag_dict = {}
	header = lines[0]
	for i in range(1, len(lines)):
		line = lines[i]
		sid,note,diagnosis = line.split("\tdelimiter\t")
		if sid not in sid_note_dict:
			sid_note_dict[sid] = note
			sid_diag_dict[sid] = diagnosis
		else:
			sid_note_dict[sid] += note
	outstring = header
	for subjid in sid_diag_dict:
		outstring += "\tdelimiter\t".join([subjid,sid_note_dict[subjid],sid_diag_dict[subjid]])
	outfile = open("clean_rheumatol_labeled_patient_centric.csv","w")
	outfile.write(outstring)
	outfile.close()

def bow(is_sda=True):
	if is_sda:
		infile = "C:\\SLENLP\\csv_data\\clean_rheumatol_labeled_patient_centric.csv"
	rocs = []
	test_datas = []
	testval = []
	#probas_list = np.array()
	mean_tpr = 0.0
	mean_fpr = np.linspace(0,1,100)
	all_tpr = []
	for fold in range(10):
		data = pd.read_csv(infile, header=0, delimiter="\tdelimiter\t", quoting=3)
		r = np.random.RandomState()
		saved_state = r.get_state()
		data.reindex(np.random.permutation(data.index))
		r.set_state(saved_state)
		#train = data[:int(9./10.*float(len(data["note"])))]
		#test = data[int(9./10.*float(len(data["note"]))):]
		total_length = float(len(data["note"]))
		test = copy.copy(data[int(float(fold)/10.*total_length):int(float(fold+1)/10.*total_length)])
		remove_from_training = range(int(float(fold)/10.*total_length),int(float(fold+1)/10.*total_length))
		train = copy.copy(data)
		train = train.drop(remove_from_training)
		'''if is_sda:
			# We want this to happen after the testing is removed, hence the placement
			# We want our validation set to be 15% of the training so let's chop
			#	it off of the end (end is arbitrary, but easy to keep constant)
			valid = copy.copy(train[int(.85*len(train)):])
			print len(valid["diagnosis"].values)
			print valid["diagnosis"].values
			remove_from_training_again = range(int(.85*len(train)),len(train))
			train = train.drop(remove_from_training_again)'''
		#print total_length
		#print len(train)
		#print len(train["note"])
		#print len(train["diagnosis"])
		#print len(train)
		#print len(test)
		#print len(train["note"])
		#print train["note"][24351]

		print "Data loaded...\n"
		# Split up into training and testing

		# Loop over each note; create an index i that goes from
		# 	0 to the length of the note list
		###
		# THIS PART ONLY WORKS WITH SDA RIGHT NOW. THE int(i) CHECKS
		#  ARE THE CULPRIT. CHANGE IT IF YOU WANT TO USE IT IN THE FUTURE
		###
		print "Cleaning and parsing the training set notes...\n"
		clean_train_notes = []
		for i in xrange( 0, len(data) ): # update data to training when division happens
			if int(i) not in remove_from_training:# and int(i) not in remove_from_training_again:
				clean_train_notes.append(
					" ".join(KaggleWord2VecUtility.
						review_to_wordlist(train["note"][i], True)))
		#print "here",clean_train_notes[0]
		'''clean_valid_notes = []
		for i in xrange( 0, len(data) ): # update data to training when division happens
			if int(i) in remove_from_training_again:
				clean_valid_notes.append(
					" ".join(KaggleWord2VecUtility.
						review_to_wordlist(valid["note"][i], True)))
		'''

		# Create bag of words from the training set
		print "Creating the bag of words...\n"

		# Initialize the "CountVectorizer" object, which is scikit-learn's
		# bag of words tool.
		vectorizer = CountVectorizer(analyzer = "word",   \
		                             tokenizer = None,    \
		                             preprocessor = None, \
		                             stop_words = None,   \
		                             max_features = 5000)

		# fit_transform() does two functions: First, it fits the model
		# and learns the vocabulary; second, it transforms our training data
		# into feature vectors. The input to fit_transform should be a list of
		# strings.
		train_data_features = vectorizer.fit_transform(clean_train_notes)

		# Numpy arrays are easy to work with, so convert the result to an
		# array
		#train_data_features = train_data_features.toarray()
		train_data_features = train_data_features.todense()
		#print train_data_features.shape
		#train_data_features = np.asarray(train_data_features)
		val = copy.copy(int(.85*train_data_features.shape[0]))
		valid_data_features = train_data_features[val:]
		#print val,train_data_features.shape[0]
		train_data_features = np.delete(train_data_features,range(val,train_data_features.shape[0]),0)
		#print train_data_features
		#print valid_data_features
		#break
		'''if is_sda:
			valid_data_features = vectorizer.fit_transform(clean_valid_notes)
			valid_data_features = valid_data_features.todense()
		'''
		if not is_sda:
			# ******* Train a random forest using the bag of words
			#
			print "Training the random forest (this may take a while)..."


			# Initialize a Random Forest classifier with 100 trees
			forest = RandomForestClassifier(n_estimators = 100)

			# Fit the forest to the training set, using the bag of words as
			# features and the sentiment labels as the response variable
			#
			# This may take a few minutes to run
			forest = forest.fit( train_data_features, train["diagnosis"].values )

			print "Forest trained. Preparing testing data..."

		clean_test_notes = []
		for i in xrange( 0, len(data) ): # update data to training when division happens
			if i in remove_from_training:
				clean_test_notes.append(
					" ".join(KaggleWord2VecUtility.
						review_to_wordlist(test["note"][i], True)))

		test_data_features = vectorizer.transform(clean_test_notes)
		if not is_sda:
			result = forest.predict(test_data_features.todense())
			probas = forest.predict_proba(test_data_features.todense())
			if fold == 0:
				probas_list = probas
			else:
				probas_list = np.concatenate((probas_list, probas), axis=0)
			testval = []
			for val in test["diagnosis"].values:
				if val.astype(np.int32) == 100:
					testval.append(1.)
				else:
					testval.append(0.)
			fpr,tpr,thresholds = metrics.roc_curve(testval, probas[:, 1], pos_label=1)
			roc = metrics.auc(fpr,tpr)
			mean_tpr += interp(mean_fpr,fpr,tpr)
			mean_tpr[0] = 0.0
			plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area = %0.2f)' % (fold,roc))
			#print "ROC:",roc
			#rocs.append(roc)
			#test_datas += probas
			#testval = []
			output = pd.DataFrame( data={"subjectid":test["subjectid"], "diagnosis":result})
			output.to_csv("bag_of_words"+str(fold)+".csv", index=False, quoting=3)

			print "Complete: Wrote results to bag_of_words"+str(fold)+".csv"

			plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label='Luck')
			mean_tpr /= 10
			mean_tpr[-1] = 1.0
			mean_auc = metrics.auc(mean_fpr,mean_tpr)
			plt.plot(mean_fpr,mean_tpr,'k--',label='Mean ROC (area = %0.2f)' % mean_auc,lw=2)
			plt.xlim([-0.05, 1.05])
			plt.ylim([-0.05, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver operating characteristic example')
			plt.legend(loc="lower right")
			plt.show()
			#fpr,tpr,thresholds = metrics.roc_curve(testval, probas_list[:, 1], pos_label=1)
			#roc = metrics.auc(fpr,tpr)
			#print "len",len(testval)
			#print "ROC:",roc
			#print "fpr:",fpr
			#print "tpr:",tpr
			#print "Final ROC:",sum(rocs)/10.
		else:
			'''from sklearn.ensemble import ExtraTreesClassifier
			clf = ExtraTreesClassifier(max_features=100)
			print 'len',len(train_data_features)
			print train_data_features
			train_data_features = np.asarray(train_data_features)
			valid_data_features = np.asarray(valid_data_features)
			test_data_features = np.asarray(test_data_features)
			train_data_features = clf.fit(train_data_features,train["diagnosis"].values).transform(train_data_features)
			valid_data_features = clf.transform(valid_data_features)
			test_data_features = clf.transform(test_data_features)'''
			# first, create pickled object
			# second, call run_sda
			import gzip
			import cPickle as pickle
			import pdb
			#pdb.set_trace()
			# Create validation from training
			print type(np.asarray(train_data_features,dtype=np.float64))
			print type(train["diagnosis"].values[:val])
			#print "HERE",train_data_features.shape
			test_data_features = test_data_features.todense()
			#pdb.set_trace()
			train_labels = ["1" if x==100 else "0" for x in train["diagnosis"].values[:val]]
			valid_labels = ["1" if x==100 else "0" for x in train["diagnosis"].values[val:]]
			test_labels = ["1" if x==100 else "0" for x in test["diagnosis"].values]
			pickleArray = [[train_data_features,train_labels],
							[valid_data_features,valid_labels],
							[test_data_features,test_labels],
							[train_data_features]]
			f = gzip.open("sle.pkl.gz"+str(fold),"wb")
			pickle.dump(pickleArray,f)
			f.close()
			#import subprocess as sp 
			#sp.call("python sle_SdA.py 1 "+str(fold),shell=True)

def create_labels():
	diag_dict = {}
	infile = open("diagnoses.csv","r")
	diagnoses = infile.readlines()
	infile.close()
	for d in diagnoses:
		sid, diag = d.split("\t")
		diag = diag.strip()
		if sid not in diag_dict:
			diag_dict[sid] = diag
	return diag_dict

def patient_centric_predict():
	for fold in range(10):
		infile = open("bag_of_words"+str(fold)+".csv","r")
		if fold == 0:
			lines = infile.readlines()
		else:
			lines += infile.readlines()
		infile.close()
	# We have all the predicted labels
	diag_dict = create_labels()
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for line in lines:
		diag,sid = line.split(",")
		sid = sid.strip()
		if diag == "diagnosis":
			continue
		if diag == diag_dict[sid]:
			if diag == "100":
				TP += 1
			else:
				TN += 1
		else:
			if diag == "100":
				FP += 1
			else:
				FN += 1
	correct = TP + TN
	incorrect = FP + FN
	print "Accuracy of "+str(float(correct)/float(int(incorrect)+int(correct)))
	print "TP FP TN FN ",TP,FP,TN,FN

def ensemble_predict():
	for fold in range(10):
		infile = open("bag_of_words"+str(fold)+".csv","r")
		if fold == 0:
			lines = infile.readlines()
		else:
			lines += infile.readlines()
		infile.close()
	# We have all the predicted labels
	
	pred_diag = {}

	# Let's figure out what the ensemble predicts
	prev_sid = "None yet"
	neg = 0
	pos = 0
	for line in lines:
		diag, sid = line.split(",")
		sid = sid.strip()
		if diag == "diagnosis":
			continue
		curr_sid = sid
		if prev_sid == "None yet":
			prev_sid = curr_sid
		elif curr_sid != prev_sid:
			if pos > neg:
				pred_diag[prev_sid] = "100"
			elif neg > pos:
				pred_diag[prev_sid] = "-100"
			else:
				pred_diag[prev_sid] = "Tied"
			pos = 0
			neg = 0
			prev_sid = curr_sid
		else: # curr_sid == prev_sid
			if diag == "-100":
				neg += 1
			elif diag == "100":
				pos += 1
			else:
				print "diagnosis of 100 or -100 got through"
				break

	diag_dict = create_labels()
	# diag_dict now contains labels

	tied = 0
	correct = 0
	incorrect = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for key in pred_diag:
		guess = pred_diag[key]
		answer = diag_dict[key]
		if guess == "Tied":
			tied += 1
		elif guess == answer:
			correct += 1
			if guess == "100":
				TP += 1
			else:
				TN += 1
		else:
			incorrect += 1
			if guess == "100":
				FP += 1
			else:
				FN += 1
	print "Accuracy of "+str(float(correct)/float(int(incorrect)+int(correct)))+" with "+str(tied)+" ties"
	print "TP FP TN FN ",TP,FP,TN,FN

def deidentify():
	infile = "C:\\SLENLP\\csv_data\\clean_rheumatol_labeled_patient_centric.csv"
	f = open(infile,"r")
	lines = f.readlines()
	f.close()
	w = open("words.txt","r") # copy of the words file from Ubuntu
	words = w.readlines()
	w.close()
	words_index = 0
	out = str(lines[0])
	append_extra = ''
	# Create a dictionary for mapping words
	print "Creating mapper..."
	mapper = {}
	for i in range(1,len(lines)):
		sid,note,diagnosis = lines[i].split("\tdelimiter\t")
		note_word_list = note.split(" ")
		for j in range(len(note_word_list)):
			if note_word_list[j] not in mapper:
				if words_index >= len(words):
					words_index = 0
					append_extra += 'a'
				mapper[note_word_list[j]] = words[words_index].strip()+append_extra
				words_index += 1
	print "Mapper created..."
	# De-identify notes with the mapping
	print "Deidentifying notes"
	for i in range(1,len(lines)):
		if i%50 == 0:
			print str(i)+" patients deidentified\n"
		new_note = ''
		sid,note,diagnosis = lines[i].split("\tdelimiter\t")
		note_word_list = note.split(" ")
		for j in range(len(note_word_list)):
			new_note += mapper[note_word_list[j]]+" "

		out += sid + "\tdelimiter\t" + new_note + "\tdelimiter\t" + diagnosis
	print "Deidentification done."
	o = open("deidentified_labeled_patients.csv","w")
	o.write(out)
	o.close()
#ensemble_predict()
#patient_centric_predict()
#bow()
deidentify()