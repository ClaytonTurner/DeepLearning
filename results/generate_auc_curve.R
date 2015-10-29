library(ROCR)

labels <- read.csv("labels.txt",sep="\n",header=F) 
predictions <- read.csv("p_values.txt",sep="\n",header=F)

pred <- prediction( predictions, labels )
perf <- performance( pred, "tpr", "fpr" )
pdf(plot( perf ))
