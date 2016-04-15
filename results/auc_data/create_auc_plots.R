library(pROC)

bow_nb_p = "nb_bow_p_values.txt"
bow_nb_l = "nb_bow_labels.txt"
cui_nb_p = "nb_cui_p_values.txt"
cui_nb_l = "nb_cui_labels.txt"
bow_nn_p = "nn_bow_p_values.txt" 
bow_nn_l = "nn_bow_labels.txt"
cui_nn_p = "nn_cui_p_values.txt"
cui_nn_l = "nn_cui_labels.txt"
bow_rf_p = "rf_bow_p_values.txt"
bow_rf_l = "rf_bow_labels.txt"
cui_rf_p = "rf_cui_p_values.txt"
cui_rf_l = "rf_cui_labels.txt"
bow_svm_p = "svm_bow_p_values.txt"
bow_svm_l = "svm_bow_labels.txt"
cui_svm_p = "svm_cui_p_values.txt"
cui_svm_l = "svm_cui_labels.txt"
w2v_p = "w2v_p_values.txt"
w2v_l = "w2v_labels.txt"

labels <- c(do.call("cbind",read.csv("nb_bow_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("nb_bow_p_values.txt",sep="\n",header=F)))
bow_nb_roc <- roc(labels,predictions) # Get the ROC data via labels/pvalues
plot(bow_nb_roc,col="cadetblue1") # Actually plot
par(new = T) # We're done here and we don't want to wipe the graphics frame for the next call

labels <- c(do.call("cbind",read.csv("nb_cui_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("nb_cui_p_values.txt",sep="\n",header=F)))
cui_nb_roc <- roc(labels,predictions)
plot(cui_nb_roc,col="cadetblue4",xaxt="n",yaxt="n")
par(new = T)

labels <- c(do.call("cbind",read.csv("nn_bow_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("nn_bow_p_values.txt",sep="\n",header=F)))
bow_nn_roc <- roc(labels,predictions)
plot(bow_nn_roc,col="chartreuse1",xaxt="n",yaxt="n")
par(new = T)

labels <- c(do.call("cbind",read.csv("nn_cui_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("nn_cui_p_values.txt",sep="\n",header=F)))
cui_nn_roc <- roc(labels,predictions)
plot(cui_nn_roc,col="chartreuse4",xaxt="n",yaxt="n")
par(new = T)

labels <- c(do.call("cbind",read.csv("rf_bow_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("rf_bow_p_values.txt",sep="\n",header=F)))
bow_rf_roc <- roc(labels,predictions)
plot(bow_rf_roc,col="chocolate1",xaxt="n",yaxt="n")
par(new = T)

labels <- c(do.call("cbind",read.csv("rf_cui_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("rf_cui_p_values.txt",sep="\n",header=F)))
cui_rf_roc <- roc(labels,predictions)
plot(cui_rf_roc,col="chocolate4",xaxt="n",yaxt="n")
par(new = T)

labels <- c(do.call("cbind",read.csv("svm_bow_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("svm_bow_p_values.txt",sep="\n",header=F)))
bow_svm_roc <- roc(labels,predictions)
plot(bow_svm_roc,col="deeppink1",xaxt="n",yaxt="n")
par(new = T)

labels <- c(do.call("cbind",read.csv("svm_cui_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("svm_cui_p_values.txt",sep="\n",header=F)))
cui_svm_roc <- roc(labels,predictions)
plot(cui_svm_roc,col="deeppink4",xaxt="n",yaxt="n")
par(new = T)

labels <- c(do.call("cbind",read.csv("w2v_labels.txt",sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv("w2v_p_values.txt",sep="\n",header=F)))
w2v_roc <- roc(labels,predictions)
plot(w2v_roc,col="gold",xaxt="n",yaxt="n")

legend("right", legend = c("NB BOW","NB CUI","NN BOW","NN CUI","RF BOW","RF CUI","SVM BOW","SVM CUI","W2V"), col=c("cadetblue1","cadetblue4","chartreuse1","chartreuse4","chocolate1","chocolate4","deeppink1","deeppink4","gold"), lty=1, lwd=5)
