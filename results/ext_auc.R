library(AUC)

args <- commandArgs(T)

fold <- args[1]

labels <- read.csv(paste(fold,"_external_labels.txt",sep=""),sep="\n",header=F)
if(length(args)==2){
	predprobs <- read.csv(paste(fold,"_external_p_values_nn.txt",sep=""),sep="\n",header=F)
} else{
	predprobs <- read.csv(paste(fold,"_external_p_values.txt",sep=""),sep="\n",header=F)
}

labels <- factor(round(labels[,1]))
predprobs <- predprobs[,1]

print("AUC for external set")
auc(roc(predprobs,labels))
