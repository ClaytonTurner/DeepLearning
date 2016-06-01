# This file is not to be run directly, but within the sle_sda and rf folders
# It's easier this way since model can refer to hidden layers or a tree index
# Additionally, if anything is added we don't have to re-run things

#model = 0 # 0,1,then 2 for RF. 1,2,then 3 for NN

for ii in range(4):
	for j in range(3):
		for k in range(3):
			model = str(ii)+","+str(j)+","+str(k)
			p_values = []
			labels = []
			for i in range(20):
			    f = open(str(i)+"_"+str(model)+"_p_values.txt","r")
			    temp_p_values = f.readlines()
			    f.close()
			    f = open(str(i)+"_"+str(model)+"_labels.txt","r")
			    temp_labels = f.readlines()
			    f.close()
			    p_values.extend(temp_p_values)
			    labels.extend(temp_labels)

			f = open(str(model)+"_p_values.txt","w")
			f.write("".join(p_values))
			f.close()

			f = open(str(model)+"_labels.txt","w")
			f.write("".join(labels))
			f.close()
