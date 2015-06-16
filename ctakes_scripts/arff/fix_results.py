f = open("sorted_golddata.txt","r")
lines = f.readlines()
f.close()

arfflist = []
subjectids = []

builder = ''
first = True
for l in lines:
        sid,attr,val = l.split("\t")
        if sid not in subjectids:
                if not first:
                        arfflist.append(builder+'},\n')
                first = False
                subjectids.append(sid)
                builder = '{1 '+sid+','+attr+' '+val.strip()+', '
        else: #then let's continue
                builder += attr+' '+val.strip()+', '

f = open("fix_data_out2gold.arff","w")
f.write(''.join(arfflist))
f.close()

