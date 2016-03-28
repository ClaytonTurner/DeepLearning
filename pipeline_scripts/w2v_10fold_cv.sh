cd ../word2vec

if [ $# -lt 1 ]
	then
		echo "Running CI version"
		for i in `seq 1 9`; do
			python Kaggle_Python_Word2Vec.py $i $1
		done
		python Kaggle_Python_Word2Vec.py 10 $1
	else
		for i in `seq 1 9`; do
			python Kaggle_Python_Word2Vec.py $i 
		done
		python Kaggle_Python_Word2Vec.py 10 
fi

