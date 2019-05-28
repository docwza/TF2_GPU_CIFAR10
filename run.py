import argparse

import tensorflow as tf
import numpy as np
from dataset import DataSet
from model import Model

def parse_cl_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", default=False, action='store_true', dest='gpu')

    #dataset params
    parser.add_argument("-data", type=str, default='cifar10', dest='data')
    parser.add_argument("-batch", type=int, default=128, dest='batch')
    parser.add_argument("-epochs", type=int, default=100, dest='epochs')

    #model params
    parser.add_argument("-lr", type=float, default=0.0005, dest='lr')
    parser.add_argument("-lreps", type=float, default=0.001, dest='lreps')
    parser.add_argument("-opt", type=str, default='adam', dest='opt')
    parser.add_argument("-arch", type=str, default="C48_3_1_2/C96_3_1_2/C192_3_1_2/F512/F256", dest='arch')
    parser.add_argument("-d", type=float, default=0.33, dest='d')

    parser.add_argument("-norm", type=str, default='z', dest='norm')

    args = parser.parse_args()
    return args

def main():
    #get hyperparameter args
    #Conv layer - C + nfilters + _ + ksize + _ + stride + _ + maxpool
    args = parse_cl_args()

    device = 'GPU' if args.gpu else 'CPU'
        
    with tf.device('/'+device+':0'):
        #model_arch = "C48_3_1_2/C96_3_1_2/C192_3_1_2/F512/F256"                        
        flip = True
                                                                                       
        #get dataset
        dataset = DataSet(args.data, norm=args.norm, flip=flip )
        input_shape, output_shape = dataset.get_dimensions()
                                                                                       
        #create model
        model = Model(args.arch, args.lr, args.lreps, args.opt, input_shape, output_shape, args.d)
        
        #loop epochs using model on dataset
        for e in range(args.epochs):
            ###augment dataset for new training epoch
            x_train, y_train, x_test, y_test = dataset.augment_dataset()
            
            model.run(1, args.batch, x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()
