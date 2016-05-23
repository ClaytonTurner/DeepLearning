library(pROC)

bow_nb_p = "results_run_nb_bow/best_p_values.txt"
bow_nb_l = "results_run_nb_bow/best_labels.txt"
cui_nb_p = "results_run_nb_cui/best_p_values.txt"
cui_nb_l = "results_run_nb_cui/best_labels.txt"
bow_nn_p = "results_run_sle_sda_bow/0_best_test_p_values.txt"
bow_nn_l = "results_run_sle_sda_bow/0_best_test_labels.txt"
cui_nn_p = "results_run_sle_sda_cui/0_best_test_p_values.txt"
cui_nn_l = "results_run_sle_sda_cui/0_best_test_labels.txt"
bow_rf_p = "results_run_rf_bow/1_best_test_p_values.txt"
bow_rf_l = "results_run_rf_bow/1_best_test_labels.txt"
cui_rf_p = "results_run_rf_cui/best_p_values.txt"
cui_rf_l = "results_run_rf_cui/best_labels.txt"
bow_svm_p = "results_run_svm_bow/best_p_values.txt"
bow_svm_l = "results_run_svm_bow/best_labels.txt"
cui_svm_p = "results_run_svm_cui/best_p_values.txt"
cui_svm_l = "results_run_svm_cui/best_labels.txt"
w2v_p = "results_run_inversion/fixed_best_p_values.txt"
w2v_l = "results_run_inversion/fixed_best_labels.txt"

print("BOW NB")
labels <- c(do.call("cbind",read.csv(bow_nb_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(bow_nb_p,sep="\n",header=F)))
bow_nb_roc <- roc(labels,predictions) # Get the ROC data via labels/pvalues
plot(bow_nb_roc,col="cadetblue1") # Actually plot
par(new = T) # We're done here and we don't want to wipe the graphics frame for the next call

print("CUI NB")
labels <- c(do.call("cbind",read.csv(cui_nb_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(cui_nb_p,sep="\n",header=F)))
cui_nb_roc <- roc(labels,predictions)
plot(cui_nb_roc,col="cadetblue4",xaxt="n",yaxt="n")
par(new = T)

print("BOW NN")
labels <- c(do.call("cbind",read.csv(bow_nn_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(bow_nn_p,sep="\n",header=F)))
bow_nn_roc <- roc(labels,predictions)
plot(bow_nn_roc,col="chartreuse1",xaxt="n",yaxt="n")
par(new = T)

print("CUI NN")
labels <- c(do.call("cbind",read.csv(cui_nn_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(cui_nn_p,sep="\n",header=F)))
cui_nn_roc <- roc(labels,predictions)
plot(cui_nn_roc,col="chartreuse4",xaxt="n",yaxt="n")
par(new = T)

print("BOW RF")
labels <- c(do.call("cbind",read.csv(bow_rf_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(bow_rf_p,sep="\n",header=F)))
bow_rf_roc <- roc(labels,predictions)
plot(bow_rf_roc,col="chocolate1",xaxt="n",yaxt="n")
par(new = T)

print("CUI RF")
labels <- c(do.call("cbind",read.csv(cui_rf_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(cui_rf_p,sep="\n",header=F)))
cui_rf_roc <- roc(labels,predictions)
plot(cui_rf_roc,col="chocolate4",xaxt="n",yaxt="n")
par(new = T)

print("BOW SVM")
labels <- c(do.call("cbind",read.csv(bow_svm_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(bow_svm_p,sep="\n",header=F)))
bow_svm_roc <- roc(labels,predictions)
plot(bow_svm_roc,col="deeppink1",xaxt="n",yaxt="n")
par(new = T)

print("CUI SVM")
labels <- c(do.call("cbind",read.csv(cui_svm_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(cui_svm_p,sep="\n",header=F)))
cui_svm_roc <- roc(labels,predictions)
plot(cui_svm_roc,col="deeppink4",xaxt="n",yaxt="n")
par(new = T)

print("Inversion")
labels <- c(do.call("cbind",read.csv(w2v_l,sep="\n",header=F)))
predictions <- c(do.call("cbind",read.csv(w2v_p,sep="\n",header=F)))
w2v_roc <- roc(labels,predictions)
plot(w2v_roc,col="gold",xaxt="n",yaxt="n")

legend("right", legend = c("NB BOW","NB CUI","NN BOW","NN CUI","RF BOW","RF CUI","SVM BOW","SVM CUI","W2V"), col=c("cadetblue1","cadetblue4","chartreuse1","chartreuse4","chocolate1","chocolate4","deeppink1","deeppink4","gold"), lty=1, lwd=5)
