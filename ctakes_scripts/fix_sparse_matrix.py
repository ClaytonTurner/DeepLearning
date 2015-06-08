def create_subject_note_hash():
	f = open("patient_and_note_ids.txt","r")
	lines = f.readlines()
	f.close()
	myhash = {}
	for line in lines:
		mylist = line.split('\t')
		noteid = mylist[0].strip()
		subjectid = mylist[1].strip()
		#if subjectid in myhash:
		#	myhash[subjectid].append(noteid)
		#else:
		#	myhash[subjectid] = [noteid]
		myhash[noteid]=subjectid
	return myhash
def data():
	#f = open("attributes.txt","r")
	#attributes = f.readlines()
	#f.close()
	f = open("data.txt","r")
	data = f.readlines()
	f.close()
	subject_note_hash = create_subject_note_hash()

	new_data = ''
	index = 0
	for line in data:
		mylist = line.split("\t")
		assert len(mylist)==3
		#print index
		#print mylist[0]
		#print index == mylist[0]
		if str(index) == mylist[0]:
			# we're in a cui row so let's grab it
			mylist[0] = subjectid
			line = '\t'.join(mylist)
			new_data += line
		else:
			# we don't want the row specifying the noteid so just increment index
			index += 1
			# rewrite the noteid row
			subjectid = subject_note_hash[str(mylist[2]).strip()]

	# Now we must consider if two notes each said the same cui
	new_data_list = new_data.split('\n')
	new_data2 = ''
	ids = []
	values = []
	for i in range(len(new_data_list)):
		subjectid = new_data_list[i][0]
		cui = new_data_list[i][1]
		value = new_data_list[i][2]
		for j in range(i+1,len(new_data_list)):
			if subjectid==new_data_list[j][0] and cui==new_data_list[j][1] and subjectid != '0':
				# So we have found a matching subject id and cui
				ids.append(j)
				values.append(new_data_list[j][2])
		tempval = 0.0
		for val in values:
			tempval += float(val)
		for i in ids:
			new_data_list[i] = ['0','0','0.0']
		if len(values) > 0:
			new_data_list[i][2] = str(float(value) + tempval)
	new_data2 = '\n'.join(new_data_list)
	#out = open("modified_data2.txt","w")
	out = open("modified_data.txt","w")
	out.write(new_data2)
	out.close()

def instance():
	f = open("instance.txt","r")
	instances = f.readlines()
	f.close()
	subject_note_hash = create_subject_note_hash()
	new_instances = ''
	for instance in instances:
		mylist = instance.split("\t")
		assert len(mylist) == 2
		subjectid = subject_note_hash[mylist[0]]
		mylist[0] = subjectid 
		instance = '\t'.join(mylist)
		new_instances += instance
	temp = new_instances.split('\n')
	temp2 = set(temp)
	instances = list(temp2)
	new_instances = '\n'.join(instances)
	out = open("modified_instance.txt","w")
	out.write(new_instances)
	out.close()
data()
