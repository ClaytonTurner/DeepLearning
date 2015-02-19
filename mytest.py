import denoisingAutoencoder as dA
import SdA
import os
import sys

def run():
    #os.chdir("C:\\Users\\Clayton\\DeepLearning\\datasets")
    print os.getcwd()
    
    '''
    defaults for denoising autoencoder test
    learning_rate=0.1
    training_epochs=15
    dataset='mnist.pk1.gz'
    batch_size=20
    output_folder=dA_plots
    '''
    dA.test_dA()
    
    '''
    defaults for Stacked denoising auto-encoder class
    finetune_lr=0.1
    pretraining_epochs=15
    pretrain_lr=0.001
    training_epochs=1000
    dataset='mnist.pkl.gz'
    batch_size=1
    '''
    SdA.test_SdA()