cd ..

CLUSTERS=9121

python pickle_word2vec.py .85 1 $CLUSTERS
python naive_bayes.py 1 >> results/fold01.out

python pickle_word2vec.py .85 2 $CLUSTERS
python naive_bayes.py 2 >> results/fold02.out

python pickle_word2vec.py .85 3 $CLUSTERS
python naive_bayes.py 3 >> results/fold03.out

python pickle_word2vec.py .85 4 $CLUSTERS
python naive_bayes.py 4 >> results/fold04.out

python pickle_word2vec.py .85 5 $CLUSTERS
python naive_bayes.py 5 >> results/fold05.out

python pickle_word2vec.py .85 6 $CLUSTERS
python naive_bayes.py 6 >> results/fold06.out

python pickle_word2vec.py .85 7 $CLUSTERS
python naive_bayes.py 7 >> results/fold07.out

python pickle_word2vec.py .85 8 $CLUSTERS
python naive_bayes.py 8 >> results/fold08.out

python pickle_word2vec.py .85 9 $CLUSTERS
python naive_bayes.py 9 >> results/fold09.out

python pickle_word2vec.py .85 10 $CLUSTERS
python naive_bayes.py 10 >> results/fold10.out

cat results/01_p_values.txt results/02_p_values.txt results/03_p_values.txt results/04_p_values.txt results/05_p_values.txt results/06_p_values.txt results/07_p_values.txt results/08_p_values.txt results/09_p_values.txt results/10_p_values.txt > results/p_values.txt

cat results/01_labels.txt results/02_labels.txt results/03_labels.txt results/04_labels.txt results/05_labels.txt results/06_labels.txt results/07_labels.txt results/08_labels.txt results/09_labels.txt results/10_labels.txt > results/labels.txt
