import sys
# Slightly different for NNs as we use Theano mechanisms
is_nn = False
if len(sys.argv) > 1:
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
        guess = round(float(p_values[j].strip()))
        # If/else makes it easier since we can vary the amount of datapoints per fold
        if label == guess:
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
        guess = round(float(p_values[i].strip()))
        #print label,guess
        if label == guess:
            correct += 1
        else:
            incorrect += 1
    print float(correct)/(float(correct)+float(incorrect))
