from rlib.networks.networks import* 

def one_to_many_rnn(cell, X, hidden_init, hidden_mask, num_timesteps, parallel_iterations=32, swap_memory=False, time_major=True, scope='rnn', trainable=True):
    ''' dynamic masked *hidden state* RNN for sequences that reset part way through an observation 
        e.g. A2C 
        args :
            cell - cell of type tf.nn.rnn_cell
            X - tensor of rank [batch, hidden]
            hidden_init - tensor or placeholder of intial cell hidden state
            mask - tensor or placeholder of length time, for hidden state masking e.g. [True, False, False] will mask first hidden state
            parallel_iterations - number of parallel iterations to run RNN over
            swap_memory - 
            time_major - bool flag to determine order of indices of input tensor 
            scope - tf variable_scope of dynamic RNN loop
            trainable - bool flag whether to perform backpropagation to RNN cell
    '''
    with tf.variable_scope(scope):
        
        def _body(t, input, hidden, output):
            out, hidden = cell(input, hidden, hidden_mask[t])
            return t+1, out, hidden, output.write(t, out)
        
        time_steps = num_timesteps
        stacked_output = tf.TensorArray(tf.float32, size=time_steps, dynamic_size=True)
        t = 0
        t, X, hidden, stacked_output = tf.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_body,
        loop_vars=(t, X, hidden_init, stacked_output),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        back_prop=trainable) # allow flag for reservoir computing 

        #maximum_iterations=time_steps,
        #)

        stacked_output = stacked_output.stack()
        if not time_major:
            stacked_output = tf.transpose(output, perm=[1,0,2])
        
    return stacked_output, hidden





if __name__ == "__main__":
    sess = tf.Session()

    cell = LSTMCell(256)
    hidden = cell.get_initial_state(32)
    #lstm = dynamic_masked_rnn(input, cell, hidden, mask)

    seqs = tf.placeholder(shape=(32,256), dtype=tf.float32)
    mask = tf.placeholder(shape=[20, 32], dtype=tf.float32)
    #features = mlp_layer(tf.reshape(seqs, shape=[-1,256]), 256)
    #features = tf.reshape(features, shape=[-1,32,256])

    output, hidden = one_to_many_rnn(cell, seqs, hidden, mask, 20)
    print('output', output.get_shape().as_list())

    sess.run(tf.global_variables_initializer())
    seqs_np = np.random.uniform(size=(32,256)).astype(np.float32) 
    mask_np = np.zeros((20, 32))
    mask_np[1,0] = 1
    #print('mask', mask_np)
    print('seq', seqs_np[:3,0])
    out, hidden_out = sess.run([output, hidden], feed_dict = {seqs:seqs_np, mask:mask_np})
    print('out', out.shape)
    print('out', out[:20,:3,0])
    

    train_writer = tf.compat.v1.summary.FileWriter('logs/abc/rnn_test', sess.graph)