cd ..
RESULTS=results_run_sle_sda_bow
mkdir $RESULTS

for i in `seq 0 19`; do
	for n_hidden in `seq 1 3`; do
		cat /dev/null > $RESULTS/${i}_${n_hidden}_p_values.txt
		cat /dev/null > $RESULTS/${i}_${n_hidden}_labels.txt
		for folds in `seq 0 4`; do
			python pickle_sledata_displace.py $i $folds BOWs
			python sle_SdA.py 1 $i $n_hidden $folds >> $RESULTS/${i}_${n_hidden}_${folds}.out
			cat $RESULTS/${i}_${n_hidden}_${folds}_p_values.txt >> $RESULTS/${i}_${n_hidden}_p_values.txt
			cat $RESULTS/${i}_${n_hidden}_${folds}_labels.txt >> $RESULTS/${i}_${n_hidden}_labels.txt
		done
	done
	# Pick the best n_hidden
	python pick_model.py $i nn
	n_hidden=`cat $RESULTS/best_model.txt`
	python pickle_sledata_displace_test.py $i BOWs 
	python sle_SdA.py 1 $i $n_hidden test >> $RESULTS/${i}_${n_hidden}_test.out

done

