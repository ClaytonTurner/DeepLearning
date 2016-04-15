cd ..
RESULTS=results_run_svm_cui
mkdir $RESULTS
python pickle_sledata_displace_test.py 0 # 0 just so our script is generalized still 
python svm.py >> $RESULTS/0_test.out
