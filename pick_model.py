import sys
# Slightly different for NNs as we use Theano mechanisms
is_nn = False
is_w2v = False
if len(sys.argv) > 2:
    if sys.argv[2] == "w2v":
        print "is_w2v == True"
        is_w2v = True
    else:
        print "is_nn == True"
        is_nn = True

iteration = sys.argv[1]

preamble = 'results_run_rf_bow/'
#preamble = 'results_run_rf_cui/'
#preamble = 'results_run_sle_sda_cui/'
#preamble = 'results_run_sle_sda_bow/'
#preamble = 'results_run_nb_cui/'# 99.99% sure we don't use this for nb, but screw it
#preamble = 'results_run_nb_bow/'# ditto
#preamble = 'results_run_svm_cui/'# ditto
#preamble = 'results_run_svm_bow/'# ditto
# This array will be used to see which model performed best
# We then look at that model's external test set accuracy
models = []

for i in range(0,3): # rf: 0,3. nn: 1,4. nb: 0
#    for j in range(20):
#for j in range(20):
#    for i in range(0,3): # rf: 0,3. nn: 1,4
    new_preamble = preamble + str(iteration)+"_"+str(i)
    #new_preamble = preamble + str(i)+"_"+str(j)
    f = open(new_preamble+"_labels.txt","r")
    labels = f.readlines()
    f.close()
    f = open(new_preamble+"_p_values.txt","r")
    p_values = f.readlines()
    f.close()
    correct = 0
    incorrect = 0
    fold_accuracies = []

    if is_nn: # this is because of how theano saves p value stuff
        #nn_actual_p_values = []
        for k in range(len(labels)):
            label = labels[k].strip()
            p_value = p_values[k].strip()
            p_value = str(round(float(p_value))) # This is because we have values like 0.000000e+00
            print p_value
            if p_value == "0.0": # We were right as per Theano
                #nn_actual_p_values.append(label+"\n")
                correct += 1
            else: # We were wrong
                incorrect += 1
                #if label == "0.0":
                #    nn_actual_p_values.append("1.0\n")
                #else:
                #    nn_actual_p_values.append("0.0\n")

    else:
        for k in range(len(labels)): # labels and p_values are the same size
            label = float(labels[k].strip())
            if is_w2v: # We could have stripped out the other p values, but this was faster
                guess = float(p_values[k].split(",")[0].strip())
            else:
                guess = float(p_values[k].strip())
            # If/else makes it easier since we can vary the amount of datapoints per fold
            if label == round(guess):
                correct += 1
            else:
                incorrect += 1
    #print "len(labels)",str(len(labels))
    #print new_preamble
    if correct + incorrect == 0: # We have a weird bug where sometimes we go one further for some reason
        continue
    else:
        print "correct",str(correct)
        print "incorrect",str(incorrect)
        fold_accuracies.append(float(correct)/(float(correct)+float(incorrect)))

    models.append(float(sum(fold_accuracies))/float(len(fold_accuracies)))
f = open(preamble+"model_accuracies.txt","w")
f.write("\t".join(map(str,models)))
f.close()
best_model = str(models.index(max(models)))
print "Using model "+best_model

#f = open(preamble+iteration+"_best_model.txt","w")
f = open(preamble+"best_model.txt","w")
f.write(best_model)
f.close()
