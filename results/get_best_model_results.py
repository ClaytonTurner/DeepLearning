import sys
# Slightly different for NNs as we use Theano mechanisms
is_nn = False
if len(sys.argv) > 1:
    print "is_nn == True"
    is_nn = True

# This array will be used to see which model performed best
# We then look at that model's external test set accuracy
fold_accuracies = []

for i in range(10):
    if i == 9: # We're always doing 10 fold so this is fine
        index = "10"
    else:
        index = "0"+str(i+1)
    f = open(index+"_labels.txt","r")
    labels = f.readlines()
    f.close()
    f = open(index+"_p_values.txt","r")
    p_values = f.readlines()
    f.close()

    correct = 0
    incorrect = 0
    for j in range(len(labels)): # labels and p_values are the same size
        label = float(labels[j].strip())
        guess = float(p_values[j].strip())
        # If/else makes it easier since we can vary the amount of datapoints per fold
        if label == round(guess):
            correct += 1
        else:
            incorrect += 1
    fold_accuracies.append(float(correct)/(float(correct)+float(incorrect)))

# Duplicates don't bother us - we'll just take the first, best model 
best_fold = fold_accuracies.index(max(fold_accuracies))
print "Using fold "+str(best_fold+1)+"'s model. Had "+str(float(correct)/(float(correct)+float(incorrect)))+" accuracy"
if best_fold == 9:
    index = "10"
else:
    index = "0"+str(best_fold+1)

if is_nn:
    f = open(str(index)+"_external_accuracy.txt","r")
    acc = float(f.read())
    print "Accuracy for nn is: "+str(1.-acc)
    f.close()
else:
    # This is an effective repeat with different files
    # We could abstract all this into a function to avoid repeats. Future TODO
    f = open(index+"_external_labels.txt","r")
    labels = f.readlines()
    f.close()
    f = open(index+"_external_p_values.txt","r")
    p_values = f.readlines()
    f.close()

    correct = 0
    incorrect = 0
    for i in range(len(labels)):
        label = float(labels[i].strip())
        guess = float(p_values[i].strip())
        #print label,guess
        if label == round(guess):
            correct += 1
        else:
            incorrect += 1
    print float(correct)/(float(correct)+float(incorrect))

# We need to correct the labels/guesses for neural networks
if is_nn:
    f = open(index+"_external_labels.txt","r")
    labels = f.readlines()
    f.close()
    f = open(index+"_external_p_values.txt","r")
    p_values = f.readlines()
    f.close()
    nn_external_p_values = []
    for i in range(len(labels)):
        label = labels[i].strip()
        p_value = p_values[i].strip()
        p_value = p_value[:3] # This is because we have values like 0.000000000e+00
        if p_value == "0.0": # We were right as per Theano
            nn_external_p_values.append(label+"\n")
        else: # We were wrong
            if label == "0.0":
                nn_external_p_values.append("1.0\n")
            else:
                nn_external_p_values.append("0.0\n")
    f = open(index+"_external_p_values_nn.txt","w")
    f.write("".join(nn_external_p_values))
    f.close()
import subprocess
print "Executing Rscript ext_auc.R"
if is_nn:
    subprocess.Popen("Rscript ext_auc.R "+str(index)+" nn",shell=True)
else:
    subprocess.Popen("Rscript ext_auc.R "+str(index),shell=True)
