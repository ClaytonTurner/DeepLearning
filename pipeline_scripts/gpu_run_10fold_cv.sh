cd ..

python pickle_sledata.py .85 1
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 1 >> results/fold01.out


python pickle_sledata.py .85 2
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 2 >> results/fold02.out


python pickle_sledata.py .85 3  
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 3 >> results/fold03.out


python pickle_sledata.py .85 4 
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 4 >> results/fold04.out


python pickle_sledata.py .85 5 
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 5 >> results/fold05.out


python pickle_sledata.py .85 6 
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 6 >> results/fold06.out


python pickle_sledata.py .85 7 
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 7 >> results/fold07.out


python pickle_sledata.py .85 8 
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 8 >> results/fold08.out


python pickle_sledata.py .85 9 
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 9 >> results/fold09.out


python pickle_sledata.py .85 10 
THEANO_FLAGS='cuda.root=/usr/local/cuda-7.0,floatX=float32,device=gpu0,nvcc.fastmath=True' python sle_SdA.py 1 10 >> results/fold10.out

cat results/01_p_values.txt results/02_p_values.txt results/03_p_values.txt results/04_p_values.txt results/05_p_values.txt results/06_p_values.txt results/07_p_values.txt results/08_p_values.txt results/09_p_values.txt results/10_p_values.txt > results/p_values.txt

cat results/01_labels.txt results/02_labels.txt results/03_labels.txt results/04_labels.txt results/05_labels.txt results/06_labels.txt results/07_labels.txt results/08_labels.txt results/09_labels.txt results/10_labels.txt > results/labels.txt
