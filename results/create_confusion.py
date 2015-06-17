
import os
init = os.getcwd()
#successes = ['success'+str(x) for x in range(1,14)]
successes = ['success13']#['success1','success2','success3','success4','success5','success6','success7','success8']

for success in successes:
	os.chdir(os.path.join(init,success))
	f = open("p_values.txt","r")
	p_lines = f.readlines()
	f.close()
	f = open("labels.txt","r")
	labels = f.readlines()
	f.close()
	acc = 0
	for l in labels:
		acc += round(float(l.strip()))
	print "acc: ",acc
	t_n = 0
	t_p = 0
	f_n = 0
	f_p = 0

	for i in range(len(p_lines)):
		label = labels[i]
		p = p_lines[i]
		if float(label.strip()) == round(float(p.strip())):
			if float(label.strip()) > 0.5:
				t_p += 1
			else:
				print label.strip(),round(float(p.strip()))
				t_n += 1
		else:
			if float(label.strip()) > 0.5:
				f_n += 1
			else:
				f_p += 1

	if (t_p+f_n) == 0:
		sensitivity = "DNE"
	else:
		sensitivity = float(t_p)/float(t_p+f_n)
	if (f_p+t_n) == 0:
		specificity = "DNE"
	else:
		specificity = float(t_n)/float(f_p+t_n)
	if (t_p+f_p) == 0:
		precision = "DNE"
	else:
		precision = float(t_p)/float(t_p+f_p)

	string = '\n'.join(["TP: "+str(t_p),"FP: "+str(f_p),"TN: "+str(t_n),"FN: "+str(f_n),"Sensitivity: "+str(sensitivity),"Specificity: "+str(specificity),"Precision: "+str(precision)])
	f = open("confusion.txt","w")
	f.write(string)
	f.close()

