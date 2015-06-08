def create_subject_note_hash():
	f = open("patient_and_note_ids.txt","r")
	lines = f.readlines()
	f.close()
	myhash = {}
	for line in lines:
		mylist = line.split('\t')
		noteid = mylist[0].strip()
		subjectid = mylist[1].strip()
		myhash[noteid]=subjectid
	return myhash

def collapse_instance_ids(arfffile):
	f = open(arfffile,"r")
	lines = f.readlines()
	f.close()
	h = create_subject_note_hash()
	out = ''
	prev_inst = -1
	prev_row = ''
	for line in lines:
		if line[0] != "{":
			out += line
		else:
			noteid = line.split(',')[0].split(' ')[1]
			if "\n" not in noteid: # for case with no cuis for a note
				if h[noteid] == prev_inst:
				# combine the rows
					append = line[1:].strip()[:-1]
					append = ','.join(append.split(',')[1:])
					prev_row += ", "+append
				else:
					# add old row to out
					out += prev_row+"}\n"

					# start a new row set
					prev_inst = h[noteid]
					prev_row = line.strip()[:-1]
	#print out
	f = open(arfffile,"w")
	f.write(out)
	f.close()

def reorder_arff(arfffile):
	def getKey(item):
		return int(item[0])
	f = open(arfffile,"r")
	lines = f.readlines()
	f.close()
	out = ''
	for line in lines:
		if line[0] != "{":
			out += line
		else:
			mylist = line[1:-1].split(',')
			for i in range(len(mylist)):
				mylist[i] = mylist[i].split(' ')
				if len(mylist[i]) == 3:
					mylist[i].pop(0) # empty string sometimes gets passed
			sorted_list_of_lists = sorted(mylist, key=getKey)
			for i in range(len(sorted_list_of_lists)):
				sorted_list_of_lists[i] = ' '.join(sorted_list_of_lists[i])
			line = '{' + ','.join(sorted_list_of_lists) + '}\n'
			out += line
	f = open(arfffile,"w")
	f.write(out)
	f.close()

def finish_arff(arfffile):
	f = open(arfffile,"r")
	lines = f.readlines()
	f.close()
	out = ''
	for line in lines:
		if line[0] != "{":
			out += line
		else:
			mylist = line.split(',')
			newlist = []
			prev_cui = '0'
			prev_val = 0
			skip = True
			for item in mylist:
				if skip:
					newlist.append(item)
					skip = False
				else:
					cui,val = item.split(' ')
					val = val.replace("}","")
					#if val == '100' or '-100'
					if cui == prev_cui:# and val.strip()[-1] != "}"
						prev_val += int(val)


					else: # cui != prev_cui
						if not(prev_cui == "0" and prev_val == 0):
							newlist.append(prev_cui+' '+str(prev_val))
						prev_val = int(val)

					prev_cui = cui
			if prev_val > 100:
				prev_val = 100
			newlist.append(prev_cui+' '+str(prev_val))
			out += ','.join(newlist)+"}\n"
	f = open(arfffile,"w")
	f.write(out)
	f.close()

def add_controls(arfffile):
	f = open(arfffile,"r")
	lines = f.readlines()
	f.close()
	out = ''
	index = 0
	for line in lines:
		if line[0] != "{":
			out += line
		else:
			if line[-5:].strip() == '100}':
				index += 1
			else:
				line = line[:-1]+'16599 -100}\n'
			out += line
	print 395-index
	f = open(arfffile,"w")
	f.write(out)
	f.close()

def icd9_additions():
	f = open("sparse_data_finished_with_icd9.arff","r")
	lines = f.readlines()
	f.close()
	icd9 = open("icd9_codes.txt","r")
	icd9lines = icd9.readlines()
	icd9.close()
	icd9lines.pop(0)
	p_n_ids = open("patient_and_note_ids.txt","r")
	pnids = p_n_ids.readlines()
	p_n_ids.close()
	out = ''
	index = 0
	for line in lines:
		if line[0] != "{":
			out += line
		else:
			for pnid in pnids:
				if pnid.split('\t')[0].strip() == line.split(',')[0].split(' ')[1].strip():
					subjectid = pnid.split('\t')[1].strip()
					for icd9 in icd9lines:
						if subjectid == icd9.split('\t')[0].strip():
							icd9code = icd9.split('\t')[1].strip()
							myline = line.replace("?",icd9code)
							out += myline
							break
					#break
			index += 1
	f = open("sparse_data_finished_with_icd9_added.arff","w")
	f.write(out)
	f.close()
f = "sle_trial_top10.arff"
collapse_instance_ids(f)
reorder_arff(f)
finish_arff(f)
add_controls(f)
