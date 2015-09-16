cd ..

mv sle.pkl.gz0 sle.pkl.gz
python sle_SdA.py 1 1 >> results/fold01.out
mv sle.pkl.gz sle.pkl.gz0

mv sle.pkl.gz1 sle.pkl.gz
python sle_SdA.py 1 2 >> results/fold02.out
mv sle.pkl.gz sle.pkl.gz1

mv sle.pkl.gz2 sle.pkl.gz
python sle_SdA.py 1 3 >> results/fold03.out
mv sle.pkl.gz sle.pkl.gz2

mv sle.pkl.gz3 sle.pkl.gz
python sle_SdA.py 1 4 >> results/fold04.out
mv sle.pkl.gz sle.pkl.gz3

mv sle.pkl.gz4 sle.pkl.gz
python sle_SdA.py 1 5 >> results/fold05.out
mv sle.pkl.gz sle.pkl.gz4

mv sle.pkl.gz5 sle.pkl.gz
python sle_SdA.py 1 6 >> results/fold06.out
mv sle.pkl.gz sle.pkl.gz5

mv sle.pkl.gz6 sle.pkl.gz
python sle_SdA.py 1 7 >> results/fold07.out
mv sle.pkl.gz sle.pkl.gz6

mv sle.pkl.gz7 sle.pkl.gz
python sle_SdA.py 1 8 >> results/fold08.out
mv sle.pkl.gz sle.pkl.gz7

mv sle.pkl.gz8 sle.pkl.gz
python sle_SdA.py 1 9 >> results/fold09.out
mv sle.pkl.gz sle.pkl.gz8

mv sle.pkl.gz9 sle.pkl.gz
python sle_SdA.py 1 10 >> results/fold10.out
mv sle.pkl.gz sle.pkl.gz9

cat results/01_p_values.txt results/02_p_values.txt results/03_p_values.txt results/04_p_values.txt results/05_p_values.txt results/06_p_values.txt results/07_p_values.txt results/08_p_values.txt results/09_p_values.txt results/10_p_values.txt > results/p_values.txt

cat results/01_labels.txt results/02_labels.txt results/03_labels.txt results/04_labels.txt results/05_labels.txt results/06_labels.txt results/07_labels.txt results/08_labels.txt results/09_labels.txt results/10_labels.txt > results/labels.txt
