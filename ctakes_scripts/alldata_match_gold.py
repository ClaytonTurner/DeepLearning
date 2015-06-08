def remove_gold_data(arff):
		f = open(arff,"r")
		lines = f.readlines()
		f.close()
		out = str()
		for line in lines:
				if line[0] != "{":
					out += line
				else:
					diag = line.split(" ")[-1]
					#if diag != "100}\n" and diag != "-100}\n":
					#		print diag
					if diag == "777}\n": # No else so we get rid of actual diagnoses
						out += line
		f = open(arff,"w")
		f.write(out)
		f.close()	

def align_attributes(arff):
	f = open(arff,"r")
	alllines = f.readlines()
	f.close()
	f = open("golddata.arff","r")
	goldlines = f.readlines()
	f.close()
	goldattrs = []
	for line in goldlines:
		if line[:10] == "@attribute":
			goldattrs.append(line)
	i = 0
	for line in goldattrs: # add gold lines that aren't in alldata to all data (don't understand how this happens...)
			if line[:10] == "@attribute":
				if line not in alllines:
					
				i += 1

	i = 0
	allattrsupdate = []
	for line in alllines:
		if line[:10] == "@attribute":
			# Now let's see if the same attribute is in goldattrs
			if line in goldattrs:
				#So we want to keep it
				allattrsupdate.append(i) # so we know the index so we can update the data section later	
			i += 1	
		#elif line[0] = "{": # So we're in the data section


f = "alldata.arff"
#remove_gold_data(f)
align_attributes(f)
