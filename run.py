import tensorflow as tf
import numpy as np
from dataset import DataSet
from model import Model

def main():
    #get hyperparameter args
    #Conv layer - C + nfilters + _ + ksize + _ + stride + _ + maxpool
    model_arch = "C48_3_1_2/C96_3_1_2/C192_3_1_2/F512/F256"
    lr = 0.0005
    lreps = 0.001
    opt = 'adam'
    
    dropout = 0.33 
    norm = 'zca'
    flip = True
    batch = 128
    epochs = 100

    #get dataset
    dataset = DataSet('cifar10', norm=norm, flip=flip )
    input_shape, output_shape = dataset.get_dimensions()

    #create model
    model = Model(model_arch, lr, lreps, opt, input_shape, output_shape, dropout )
    
    #loop epochs using model on dataset
    for e in range(epochs):
        ###augment dataset for new training epoch
        x_train, y_train, x_test, y_test = dataset.augment_dataset()
        model.run(1, batch, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
