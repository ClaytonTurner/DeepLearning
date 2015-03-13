python pickle_sledata.py 3
python sle_SdA.py 1 >> results/out.1.0
python sle_SdA.py 100 >> results/out.1.1

python pickle_sledata.py 10
python sle_SdA.py 1 >> results/out.2.0
python sle_SdA.py 100 >> results/out.2.1

python pickle_sledata.py 100
python sle_SdA.py 1 >> results/out.3.0
python sle_SdA.py 100 >> results/out.3.1

python pickle_sledata.py 1000
python sle_SdA.py 1 >> results/out.4.0
python sle_SdA.py 100 >> results/out.4.1
