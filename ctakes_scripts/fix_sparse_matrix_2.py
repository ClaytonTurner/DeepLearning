f = open("modified_data.txt","r")
lines = f.readlines()
f.close()
string = ''
for i in range(len(lines)-1):
	mylist = lines[i].split('\t')
	if mylist[0] == '0':
		print("copy found")
	else:
		mylist2 = lines[i+1].split('\t')
		if mylist[0] == mylist2[0] and mylist[1] == mylist2[1]:
			newval = str(int(mylist[2].strip())+int(mylist2[2].strip()))
			string += mylist[0]+'\t'+mylist[1]+'\t'+newval+'\n'
			lines[i+1] = '0\t0\t0'
		else:
			string += lines[i]
outfile = open("modified_data_out.txt","w")
outfile.write(string)
outfile.close()
