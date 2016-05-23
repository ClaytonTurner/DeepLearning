import numpy as np

accs = []
for i in range(20):
    f = open(str(i)+"_1_labels.txt","r")
    labels = f.readlines()
    f.close()
    f = open(str(i)+"_1_p_values.txt","r")
    p_values = f.readlines()
    f.close()
    correct = 0
    total = 0
    for i in range(len(labels)):
        label = str(float(labels[i].strip()))
        p_value = str(round(float(p_values[i].strip())))
        if p_value == label:
            correct += 1
            total += 1
        else:
            total += 1
    accuracy = float(correct)/float(total)
    accs.append(accuracy)

StdDev = np.std(np.asarray(accs))*1.96/np.sqrt(20.) # 1.96 = 95% confidence level
Avg = float(sum(accs))/(float(len(accs)))
print "Mean: ",str(Avg)
print "StdDev: ", str(StdDev)
print "CI: ["+str(Avg-StdDev)+", "+str(Avg+StdDev)+"]"
