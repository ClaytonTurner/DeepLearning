cd ..
RESULTS=results_run_svm_bow
mkdir $RESULTS
python pickle_sledata_displace_test.py 0 BOWs # 0 just so our script is generalized still 
python svm.py >> $RESULTS/0_test.out
