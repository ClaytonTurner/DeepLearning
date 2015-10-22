cd ..

cp sle.pkl.gz0 sle.pkl.gz
python naive_bayes.py 1 >> results/fold01.out

cp sle.pkl.gz1 sle.pkl.gz
python naive_bayes.py 2 >> results/fold02.out

cp sle.pkl.gz2 sle.pkl.gz
python naive_bayes.py 3 >> results/fold03.out

cp sle.pkl.gz3 sle.pkl.gz
python naive_bayes.py 4 >> results/fold04.out

cp sle.pkl.gz4 sle.pkl.gz
python naive_bayes.py 5 >> results/fold05.out

cp sle.pkl.gz5 sle.pkl.gz
python naive_bayes.py 6 >> results/fold06.out

cp sle.pkl.gz6 sle.pkl.gz
python naive_bayes.py 7 >> results/fold07.out

cp sle.pkl.gz7 sle.pkl.gz
python naive_bayes.py 8 >> results/fold08.out

cp sle.pkl.gz8 sle.pkl.gz
python naive_bayes.py 9 >> results/fold09.out

cp sle.pkl.gz9 sle.pkl.gz
python naive_bayes.py 10 >> results/fold10.out

cat results/01_p_values.txt results/02_p_values.txt results/03_p_values.txt results/04_p_values.txt results/05_p_values.txt results/06_p_values.txt results/07_p_values.txt results/08_p_values.txt results/09_p_values.txt results/10_p_values.txt > results/p_values.txt

cat results/01_labels.txt results/02_labels.txt results/03_labels.txt results/04_labels.txt results/05_labels.txt results/06_labels.txt results/07_labels.txt results/08_labels.txt results/09_labels.txt results/10_labels.txt > results/labels.txt
