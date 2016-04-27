cd ..
RESULTS=results_run_nb_bow
mkdir $RESULTS
python pickle_sledata_displace_test.py 0 BOWs # 0 just so our script is generalized still 
python naive_bayes.py >> $RESULTS/0_test.out
