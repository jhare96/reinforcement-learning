import tensorflow as tf
import numpy as np 

def stack_many(args):
    return tuple([np.stack(arg) for arg in args])

def normalise(x, mean, std):
    return (x-mean)/std

def fold_batch(x):
    rows, cols = x.shape[0], x.shape[1]
    y = x.reshape(rows*cols,*x.shape[2:])
    return y

def unfold_batch(x, length, batch_size):
    return x.reshape(length, batch_size, *x.shape[1:])

@tf.function
def keras_fold_batch(x):
    time, batch = x.shape[0], x.shape[1]
    return tf.keras.backend.reshape(x, shape=(time*batch, *x.shape[2:]) )


def one_hot(x, num_classes):
    return np.eye(num_classes)[x]


class rolling_stats(object):
    def __init__(self,mean=0,n=1,epsilon=1e-4):
        self.mean = mean
        self.n = n
        self.M2 = 0
        self.epsilon = epsilon
    
    def update(self,x):
        self.n +=1
        prev_mean = self.mean
        new_mean = prev_mean + ((x - prev_mean)/self.n)
        self.M2 += (x - new_mean) * (x - prev_mean)
        std = np.sqrt((self.M2 + self.epsilon) / self.n)
        self.mean = new_mean
        return self.mean, std