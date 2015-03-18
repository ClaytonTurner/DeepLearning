import os

os.chdir(os.path.join(os.getcwd(),"results"))
results = os.listdir(os.getcwd())

test_errors = []
i = 0
for res in results:
	if not res[-4:] == ".swp":
		i += 1
		f = open(res,"r")
		line = f.readlines()[-1]
		f.close()
		test_errors.append(float(line.split(" ")[-2]))

import numpy as np
print "10-Fold Cross Validation accuracy: "+str(100.0 - np.mean(test_errors))+"%"
