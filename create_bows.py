'''  
This file was created when we needed to recreate BOWs for the new data
that was curated.
This is for creating BOWs.
'''
from collections import OrderedDict as OD

def make_patient_centric():
	'''
	Currently the data is note-centric (see obeid_data.csv)
	We want to make it patient-centric
	'''
	f = open("obeid_data.csv","r")
	lines = f.readlines()
	f.close()

	patient_note_dict = OD()
	patient_diagnosis_dict = OD()
	for line in lines:
		sid,note,diagnosis = line.split("\tdelimiter\t")
		note = note.replace("\n"," ")
		note = note.replace("\r"," ") # Blame Windows
		# Add to the patient's note(s)
		if sid not in patient_note_dict:
			patient_note_dict[sid] = note
		else:
			patient_note_dict[sid] += " "+note
		# Record the diagnosis
		if sid not in patient_diagnosis_dict:
			sid = sid.strip()
			if diagnosis == "100":
				patient_diagnosis_dict[sid] = "1"
			else: # Covers "0" and "-100" cases
				patient_diagnosis_dict[sid] = "0"

	write_string = ""
	i = 0
	for key in patient_note_dict:
		i += 1
	print i
	for sid in patient_note_dict:
		write_string += sid+"\tdelimiter\t"+patient_note_dict[sid]+"\tdelimiter\t"+patient_diagnosis_dict[sid]+"\n"
	write_string = write_string#.strip() removed because of the reo
	f = open("obeid_data_patient_centric.csv","w")
	f.write(write_string)
	f.close()

def reorder_data_to_match_cuis():
	'''
	Our data currently isn't ordered like the CUI data is
	This is a problem because of how we extract our external test set
	So we need to take the 20 patients we added and put them at the end

	Our data is already ordered on both ends so we don't have to worry
	about ordering outside of this
	'''
	sids_to_move = ["70757","71736","72243","74396","78579","78645","80286","81025","81228","81711","84973","87810","87897","89001","89121","92001","92656","96583","96954","98329"]
	f = open("obeid_data_patient_centric.csv","r")
	lines = f.readlines()
	f.close()

	write_string = ""
	sids_to_move_string = ""
	for line in lines:
		sid,note,diagnosis = line.split("\tdelimiter\t")
		if sid in sids_to_move:
			sids_to_move_string += line
		else:
			write_string += line
	write_string += sids_to_move_string
	f = open("obeid_data_patient_centric_correct_order.csv","w")
	f.write(write_string.strip())
	f.close()

def preprocessing():
	from nltk.corpus import stopwords
	import string
	f = open("obeid_data_patient_centric_correct_order.csv","r")
	lines = f.readlines()
	f.close()

	# Before we do any stopword removal, we need to kill punctuation
	exclude = set(string.punctuation)
	for i in range(len(lines)): 
		line = lines[i]
		new_line = ''.join(ch for ch in line if ch not in exclude)
		lines[i] = new_line

	# Before we do any data processing, we need to remove stopwords, etc.
	for i in range(len(lines)):
		line = lines[i]
		sid,data,diagnosis = line.split("\tdelimiter\t")
		data = " ".join(data.split())
		stops = set(stopwords.words("english"))
		words = data.split(" ")
		new_words = []
		for word in words:
			if word not in stops:
				new_words.append(word)
		lines[i] = " ".join(new_words)
		lines[i] = sid+"\tdelimiter\t"+lines[i]+"\tdelimiter\t"+diagnosis
	f = open("obeid_data_patient_centric_correct_order_preprocessed.csv","w")
	f.writelines(lines)
	f.close()

def create_bow_np_array():
	import numpy as np
	'''
	This function will create a numpy array for our bag of words
	In doing this we will leave out headers, creating an effective
	deidentification
	This allows us to use our high performance cluster
	'''
	f = open("obeid_data_patient_centric_correct_order_preprocessed.csv","r")
	lines = f.readlines()
	f.close()

	# We want to make one pass in order to create a word-column dictionary
	# so our notes know which column to increment according to their words
	print "Creating wordlist. This takes a while."
	wordlist = []
	for line in lines:
		sid,data,diagnosis = line.split("\tdelimiter\t")
		data = " ".join(data.split()) # Removes extra spaces, tabs, etc.
		words = data.split(" ")
		for word in words:
			if word not in wordlist:
				wordlist.append(word)
	# We can leverage list's indexing to figure out where words should increment

	np_array = []
	labels = [] # We'll numpy-ify this later
	i = 0
	for line in lines:
		print "Processing patient number "+str(i)
		sid,data,diagnosis = line.split("\tdelimiter\t")
		labels.append(diagnosis.strip()) # Label is simple
		bow_counts = [0 for x in range(len(wordlist))] # init all counts to 0
		data = data.split(" ")
		for word in data:
			column_to_increment = wordlist.index(word) # ValueError should never arise
			bow_counts[column_to_increment] += 1
		np_array.append(bow_counts)
		i += 1
	#np_array = np.asarray(np_array)
	
	# Now let's save this without headers so we can move to the server
	import gzip
	import cPickle as pickle 
	f = gzip.open("sle.bows_full.pkl.gz","wb")
	pickle.dump([np_array,labels],f)
	f.close()

make_patient_centric()
reorder_data_to_match_cuis()
preprocessing()
create_bow_np_array()