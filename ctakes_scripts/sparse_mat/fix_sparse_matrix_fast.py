from collections import OrderedDict as OD

# creates a hash
# hash[noteid] = subjectid
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

def data():
        #f = open("attributes.txt","r")
        #attributes = f.readlines()
        #f.close()
        f = open("full_data.txt","r")
        data = f.readlines()
        f.close()
        subject_note_hash = create_subject_note_hash()

        new_data = ''
        index = 0
        for line in data:
                #print "index: "+str(index)
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
        f = open("temp_data.txt","w")
        f.write(new_data)
        f.close()


def fix_data():
	# Temp_data has already been run through data() 
	# all_data and temp_data differ in that temp_data has matched subjectid's (or really 
	#	just a unique id as we use the first noteid for that patient)
	# We need to collapse the CUIs together
	f = open("temp_data.txt","r")
	data = f.readlines()
	f.close()
	subject_note_hash = create_subject_note_hash()
	
	current_sid = ''
	cuis_counts_for_sid = {}
	final_data = list()
	for line in data:
		sid,cui,count = line.split("\t")
		count = count.strip()
		if sid == current_sid: 
			if cui in cuis_counts_for_sid:
				cuis_counts_for_sid[cui] += float(count)
			else:
				cuis_counts_for_sid[cui] = float(count)
		else:
			# Add to the final list
			for c in OD(sorted(cuis_counts_for_sid.items(),key=lambda t: int(t[0]))):
				final_data.append(str(current_sid)+"\t"+str(c)+"\t"+str(cuis_counts_for_sid[c]))
				#final_data.append([str(current_sid),str(c),str(cuis_counts_for_sid[c])])
			cuis_counts_for_sid = {}
			current_sid = sid

	# Join the final_data together
	f = open("final_data_all.txt","w")
	#final_data = sorted(final_data)
	f.write("\n".join(final_data))
	f.close()		

def instance():
        #f = open("instance.txt","r")
        f = open("full_instance.txt","r")
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
        #out = open("modified_instance.txt","w")
        out = open("full_instance_modified.txt","w")
        out.write(new_instances)
        out.close()

