cd ..
RESULTS=results_run_rf_cui
mkdir $RESULTS

for i in `seq 0 19`; do
	for tree_index in `seq 0 2`; do
		cat /dev/null > $RESULTS/${i}_${tree_index}_p_values.txt
		cat /dev/null > $RESULTS/${i}_${tree_index}_labels.txt
		for folds in `seq 0 4`; do
			python pickle_sledata_displace.py $i $folds
			python random_forest.py $i $tree_index $folds >> $RESULTS/${i}_${tree_index}_${folds}.out
			cat $RESULTS/${i}_${tree_index}_${folds}_p_values.txt >> $RESULTS/${i}_${tree_index}_p_values.txt
			cat $RESULTS/${i}_${tree_index}_${folds}_labels.txt >> $RESULTS/${i}_${tree_index}_labels.txt
		done
	done
	# Pick the best tree_index
	python pick_model.py $i
	tree_index=`cat $RESULTS/best_model.txt`
	python pickle_sledata_displace_test.py $i 
	python random_forest.py $i $tree_index test >> $RESULTS/${i}_${tree_index}_test.out

done

