#!/bin/bash

if [ $# -lt 1 ]
  then
    echo "Proper usage: ./run30Times.sh <script>"
    echo "Example: ./run30Times.sh rf_10fold_cv.sh"
    exit
fi

cd ../results

#remove possible old results
rm temp_labels.txt
rm temp_p_values.txt

for i in `seq 1 30`; do	
	./../pipeline_scripts/$1

	#append the results to a temp file
	cat 01_p_values.txt 02_p_values.txt 03_p_values.txt 04_p_values.txt 05_p_values.txt 06_p_values.txt 07_p_values.txt 08_p_values.txt 09_p_values.txt 10_p_values.txt >> temp_p_values.txt
	cat 01_labels.txt 02_labels.txt 03_labels.txt 04_labels.txt 05_labels.txt 06_labels.txt 07_labels.txt 08_labels.txt 09_labels.txt 10_labels.txt >> temp_labels.txt
done

#remove old results
rm labels.txt
rm p_values.txt

#change name of temp files to conform with R script inputs
mv temp_labels.txt labels.txt
mv temp_p_values.txt p_values.txt

Rscript auc_ci.R > 30times_$1.txt
Rscript auc.R > 30times_$1_accuracy.txt

Rscript generate_auc_curve.R

mv Rplots.pdf auc_curves/$1.pdf
