cd ..
RESULTS=results_run_nb_cui
mkdir $RESULTS
python pickle_sledata_displace_test.py 0 # 0 just so our script is generalized still 
python naive_bayes.py >> $RESULTS/0_test.out
