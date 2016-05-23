models = [0,1,2]

for mo_no in models:
    f = open(str(mo_no)+"_p_values.txt","r")
    p_values = f.readlines()
    f.close()
    f = open(str(mo_no)+"_labels.txt","r")
    labels = f.readlines()
    f.close()

    new_p_values = []
    for p in p_values:
        first,second = p.split(",")
        if first == "0.5":
            continue
        else:
            new_p_values.append(first.strip())

    new_labels = []
    for l in labels:
        if l[0] == "0" or l[0] == "1":
            new_labels.append(l.strip())

    f = open("new_"+str(mo_no)+"_p_values.txt","w")
    f.write("\n".join(new_p_values))
    f.close()

    f = open("new_"+str(mo_no)+"_labels.txt","w")
    f.write("\n".join(new_labels))
    f.close()
