python pickle_sledata.py 3/10
python sle_SdA.py 1 >> results/fewcuis-33train-nopca-batch1.out
python sle_SdA.py 100 >> results/fewcuis-33train-nopca-batch100.out

python pickle_sledata.py 5/10
python sle_SdA.py 1 >> results/fewcuis-50train-nopca-batch1.out
python sle_SdA.py 100 >> results/fewcuis-50train-nopca-batch100.out

python pickle_sledata.py 8/10
python sle_SdA.py 1 >> results/fewcuis-80train-nopca-batch1.out
python sle_SdA.py 100 >> results/fewcuis-80train-nopca-batch100.out

python pickle_sledata.py 9/10
python sle_SdA.py 1 >> results/fewcuis-90train-nopca-batch1.out
python sle_SdA.py 100 >> results/fewcuis-90train-nopca-batch100.out


python pickle_sledata.py 3/10 10
python sle_SdA.py 1 >> results/fewcuis-33train-pca-batch1.out
python sle_SdA.py 100 >> results/fewcuis-33train-pca-batch100.out

python pickle_sledata.py 5/10 10
python sle_SdA.py 1 >> results/fewcuis-50train-pca-batch1.out
python sle_SdA.py 100 >> results/fewcuis-50train-pca-batch100.out

python pickle_sledata.py 8/10 10
python sle_SdA.py 1 >> results/fewcuis-80train-pca-batch1.out
python sle_SdA.py 100 >> results/fewcuis-80train-pca-batch100.out

python pickle_sledata.py 9/10 10
python sle_SdA.py 1 >> results/fewcuis-90train-pca-batch1.out
python sle_SdA.py 100 >> results/fewcuis-90train-pca-batch100.out





