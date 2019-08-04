import tensorflow as tf
import numpy as np
import time
from networks import*


def main():

    sess = tf.Session()

    cell = LSTMCell(256)
    hidden = cell.get_initial_state(32)
    #lstm = dynamic_masked_rnn(input, cell, hidden, mask)

    seqs = tf.placeholder(shape=(None,32,256), dtype=tf.float32)
    mask = tf.placeholder(shape=[None, 32], dtype=tf.float32)
    #features = mlp_layer(tf.reshape(seqs, shape=[-1,256]), 256)
    #features = tf.reshape(features, shape=[-1,32,256])

    output, hidden = dynamic_masked_rnn(cell, seqs, hidden, mask)
    print('output', output.get_shape().as_list())

    sess.run(tf.global_variables_initializer())
    seqs_np = np.random.uniform(size=(20,32,256)).astype(np.float32) 
    mask_np = np.zeros((20, 32))
    mask_np[1,0] = 1
    print('mask', mask_np)
    print('seq', seqs_np[:2,:3,0])
    out, hidden_out = sess.run([output, hidden], feed_dict = {seqs:seqs_np, mask:mask_np})
    print('out', out.shape)
    print('out', out[:2,:3,0])
    seqs_np = np.ones((200,32,256)) 
    mask_np = np.zeros((200))
    mask_np[[1,5,7]] = 1
    out2, hidden_out2 = sess.run([output, hidden], feed_dict = {seqs:seqs_np, mask:mask_np})
    print('out2', out2.shape)

    train_writer = tf.compat.v1.summary.FileWriter('logs/abc/rnn_test', sess.graph)


if __name__ == '__main__':
    main()