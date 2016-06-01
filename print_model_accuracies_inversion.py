#models = [0,1,2] # For RF and w2v
#models = [1,2,3] # For NN
models = [0,1,2,3] # for w2v

for mo_no in models:
    for ii in range(3):
        for j in range(3):
            m_string = str(mo_no)+","+str(ii)+","+str(j)
            f = open(m_string+"_p_values.txt","r")
            p_values = f.readlines()
            f.close()
            f = open(m_string+"_labels.txt","r")
            labels = f.readlines()
            f.close()
            correct = 0
            incorrect = 0
            for i in range(len(p_values)):
                #p = str(round(float(p_values[i].strip())))
                if p_values[i].split(",")[0].strip() == "0.5":
                continue 
                p = str(round(float(p_values[i].split(",")[0].strip())))
                l = str(float(labels[i].strip()))
                if p == l:
                correct += 1
                else:
                incorrect += 1
            print "Model "+m_string+" ~ correct: "+str(correct)+" | incorrect: "+str(incorrect)
            print "Model "+m_string+" Accuracy: "+str(float(correct)/(float(correct)+float(incorrect)))
