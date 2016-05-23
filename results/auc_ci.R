library(cvAUC)

labels <- read.csv("../results_run_sle_sda_bow/cv_labels.txt",sep="\n",header=F)
predprobs <- read.csv("../results_run_sle_sda_bow/cv_p_values.txt",sep="\n",header=F)

#labels <- read.csv("icd9_labels",sep="\n",header=F)
#predprobs <- read.csv("icd9_values",sep="\n",header=F)

labels <- factor(round(labels[,1]))
predprobs <- predprobs[,1]
print(length(labels))
print(length(predprobs))
#folds <- c(rep(c(1,2,3,4),each=64),rep(c(5),each=65),rep(c(6,7,8,9),each=64),rep(c(10),each=65)) #	c(64,64,64,64,65,64,64,64,64,65)
#folds <- rep(1:10,c(64,64,64,64,65,64,64,64,64,65))

#folds <- c(rep(rep(1:10,c(63,63,63,63,63,63,63,63,63,63)), times=30)) + rep(seq(0,290,10),each=630)
#folds <- c(rep(rep(1:10,c(58,58,58,58,58,58,58,58,58,58)), times=30)) + rep(seq(0,290,10),each=560)
#folds <- c(rep(rep(1:10,c(56,56,56,56,57,56,56,56,56,57)), times=30)) + rep(seq(0,290,10),each=560)

#folds <- c(rep(rep(1:5,rep(c(100), times=20), times = 20)

#print(length(folds))

#out <- ci.cvAUC(predictions=predprobs, labels=labels, folds=folds, confidence=0.95)
out <- ci.cvAUC(predictions=predprobs, labels=labels, confidence=0.95)
out
