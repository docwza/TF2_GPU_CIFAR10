import tensorflow as tf
import numpy as np

class DataSet(object):
    def __init__(self, name='mnist', norm='zca', flip=True):
        self.flip = flip

        if name == 'cifar10':
            dataset = tf.keras.datasets.cifar10
        elif name == 'cifar100':
            dataset = tf.keras.datasets.cifar100
        elif name == 'mnist':
            dataset = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        ##normalize dataset
        self.x_train, self.x_test = self.normalize(norm, x_train, x_test)
        self.y_train = y_train
        self.y_test = y_test

        ###get dimensions
        self.input_shape = x_train[0].shape 
        self.output_shape = np.max(y_train)+1 

    def get_dimensions(self):
        return self.input_shape, self.output_shape

    def normalize(self, norm, x_train, x_test):
        if norm.isnumeric():
            ###abs max normalization, supposed to bound [-1,1]
            N = float(norm)
            return x_train / N, x_test / N
        else:
            ###zero mean, unit std dev transformation
            mean = np.mean(x_train,axis=(0,1,2,3))
            std = np.std(x_train,axis=(0,1,2,3)) 
            x_train = (x_train-mean)/(std+1e-7)
            x_test = (x_test-mean)/(std+1e-7)
            return x_train, x_test

    def augment_dataset(self):
        data = self.x_train
        n = data.shape[0]                                                                     
        if self.flip:
            ###create random flips for image data augmentation 
            fliplr = np.random.choice([True, False], size=n)
            flipud = np.random.choice([True, False], size=n)
        else:
            fliplr = np.random.choice([False], size=n)
            flipud = np.random.choice([False], size=n)
                                                                                                  
        augment_x = np.stack([ self.image_augment( data[i], fliplr[i], flipud[i])  for i in range(n) ])
        return augment_x, self.y_train, self.x_test, self.y_test

    def image_augment(self, img, fliplr, flipud):
        augment_img = []                                         
        h = img.shape[0]
        w = img.shape[1]
        n = img.shape[-1]
        for i in range(n):
            layer = img[..., i]
            if fliplr:
                layer = np.fliplr(layer)
            if flipud:
                layer = np.flipud(layer)
                                                                 
            #layer += np.random.uniform(-0.15, 0.15, size=(h,w))
                                                                 
            augment_img.append(layer)
                                                                 
        img = np.stack(augment_img, axis=-1)
        return img
