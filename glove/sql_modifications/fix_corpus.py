fname = "sle_corpus_small.txt"
f = open(fname,"r")
lines = f.readlines()
f.close()

s = str()
for line in lines:
	temp_s = "\n".join(filter(None,line.split("\\n")))
	s += temp_s



fname = "sle_corpus.out"
f = open(fname,"w")
f.write(s)
f.close()
