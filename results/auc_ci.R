library(cvAUC)

labels <- read.csv("labels.txt",sep="\n",header=F)
predprobs <- read.csv("p_values.txt",sep="\n",header=F)

#labels <- read.csv("icd9_labels",sep="\n",header=F)
#predprobs <- read.csv("icd9_values",sep="\n",header=F)

labels <- factor(round(labels[,1]))
predprobs <- predprobs[,1]
#folds <- c(rep(c(1,2,3,4),each=64),rep(c(5),each=65),rep(c(6,7,8,9),each=64),rep(c(10),each=65)) #	c(64,64,64,64,65,64,64,64,64,65)
#folds <- rep(1:10,c(64,64,64,64,65,64,64,64,64,65))

folds <- c(rep(rep(1:10,c(64,64,64,64,65,64,64,64,64,65)), times=30)) + rep(seq(0,290,10),each=642)

out <- ci.cvAUC(predictions=predprobs, labels=labels, folds=folds, confidence=0.95)
out
