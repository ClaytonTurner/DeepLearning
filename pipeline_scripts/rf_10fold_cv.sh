cd ..

if [ $# -lt 1 ]
	then
		echo "Running CI version"
		for i in `seq 1 9`; do
			python pickle_sledata_displace.py .85 $i $1
			python random_forest.py $i >> results/fold0$1.out
		done
		python pickle_sledata_displace.py .85 10 $1
		python random_forest.py $i >> results/fold10.out
	else
		for i in `seq 1 9`; do
			python pickle_sledata_displace.py .85 $i
			python random_forest.py $i >> results/fold0$i.out
		done
fi

cat results/01_p_values.txt results/02_p_values.txt results/03_p_values.txt results/04_p_values.txt results/05_p_values.txt results/06_p_values.txt results/07_p_values.txt results/08_p_values.txt results/09_p_values.txt results/10_p_values.txt > results/p_values.txt

cat results/01_labels.txt results/02_labels.txt results/03_labels.txt results/04_labels.txt results/05_labels.txt results/06_labels.txt results/07_labels.txt results/08_labels.txt results/09_labels.txt results/10_labels.txt > results/labels.txt
