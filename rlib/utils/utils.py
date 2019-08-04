import tensorflow as tf
import numpy as np 

def fold_batch(x):
    rows, cols = x.shape[0], x.shape[1]
    y = x.reshape(rows*cols,*x.shape[2:])
    return y

@tf.function
def keras_fold_batch(x):
    time, batch = x.shape[0], x.shape[1]
    return tf.keras.backend.reshape(x, shape=(time*batch, *x.shape[2:]) )


def one_hot(x, num_classes):
    return np.eye(num_classes)[x]