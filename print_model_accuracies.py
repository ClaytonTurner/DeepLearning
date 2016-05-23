#models = [0,1,2] # For RF and w2v
models = [1,2,3] # For NN

for mo_no in models:
    f = open(str(mo_no)+"_p_values.txt","r")
    p_values = f.readlines()
    f.close()
    f = open(str(mo_no)+"_labels.txt","r")
    labels = f.readlines()
    f.close()
    correct = 0
    incorrect = 0
    for i in range(len(p_values)):
            p = str(round(float(p_values[i].strip())))
            l = str(float(labels[i].strip()))
            if p == l:
                correct += 1
            else:
                incorrect += 1
    print "Model "+str(mo_no)+"~ correct: "+str(correct)+" | incorrect: "+str(incorrect)
    print "Model "+str(mo_no)+" Accuracy: "+str(float(correct)/(float(correct)+float(incorrect)))
