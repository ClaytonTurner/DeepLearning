cd ..
RESULTS=results_run_nb_cui
mkdir $RESULTS

for i in `seq 0 19`; do
	cat /dev/null > $RESULTS/${i}_p_values.txt
	cat /dev/null > $RESULTS/${i}_labels.txt
	python pickle_sledata_displace_test.py $i
	python naive_bayes.py $i test >> $RESULTS/${i}_test.out	
done
