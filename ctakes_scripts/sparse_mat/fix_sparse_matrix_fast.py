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
	print "subject note hash length",len(myhash)
        return myhash

def data():
        #f = open("attributes.txt","r")
        #attributes = f.readlines()
        #f.close()
        f = open("alldata.txt","r")
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


def fix_data(data_file="temp_data.txt"):
	# Temp_data has already been run through data() 
	# all_data and temp_data differ in that temp_data has matched subjectid's (or really 
	#	just a unique id as we use the first noteid for that patient)
	# We need to collapse the CUIs together
	f = open(data_file,"r")
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
        f = open("allinstance.txt","r")
        instances = f.readlines()
        f.close()
        subject_note_hash = create_subject_note_hash()
        new_instances = ''
	for instance in instances:
		mylist = instance.split("\t")
		if len(mylist) == 2:
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

# This is for merging gold attributes and regular attributes into just the gold
def fix_attributes():
	all_attr_f = open("allattributes.txt","r")
	all_attr_lines = all_attr_f.readlines()
	all_attr_f.close()
	gold_attr_f = open("goldattributes.txt","r")
	gold_attr_lines = gold_attr_f.readlines()
	gold_attr_f.close()
	all_data_f = open("alldata.txt","r")
	all_data_lines = all_data_f.readlines()
	all_data_f.close()

	gold_cui_hash = {} # gold_cui_hash[cui] = line
	all_cui_hash = {} # all_cui_hash[line] = cui
	for i in range(len(all_attr_lines)):
		cui = all_attr_lines[i].strip()
		all_cui_hash[str(i+1)] = cui 
	for i in range(len(gold_attr_lines)):
		cui = gold_attr_lines[i].strip()
		gold_cui_hash[cui] = str(i+1)
	out = ''
	i = 0
	for i in range(len(all_data_lines)):
		sid,cui_line,val = all_data_lines[i].split("\t")
		CUI = all_cui_hash[cui_line]
		if CUI in gold_cui_hash:
			new_cui = gold_cui_hash[CUI]
			out += "\t".join([sid,new_cui,val])
		
	#for attr in all_attr_lines:
	#	if attr in gold_attr_lines:
	#		attribute = attr.strip()
	#		
	#	i += 1


	outfile = open("alldata_goldcuisonly.txt","w")
	outfile.write(out)
	outfile.close()

fix_attributes()



















