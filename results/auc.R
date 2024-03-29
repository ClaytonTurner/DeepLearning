library(AUC)

labels <- read.csv("labels.txt",sep="\n",header=F)
predprobs <- read.csv("p_values.txt",sep="\n",header=F)

#labels <- read.csv("icd9_labels",sep="\n",header=F)
#predprobs <- read.csv("icd9_values",sep="\n",header=F)

labels <- factor(round(labels[,1]))
predprobs <- predprobs[,1]

print("Accuracy")
sum(round(predprobs) == labels)/length(labels)

print("AUC (ignore)")
auc(roc(predprobs,labels))


