import tensorflow as tf 
import numpy as np

def flatten(x, name='flatten'):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])], name=name)

def blank(x):
    return x

def conv_layer(input, output_channels, kernel_size, strides, padding, activation=tf.nn.relu, name='convolutional_layer', dtype=tf.float32, kernel_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
    # N-D convolutional layer
    with tf.variable_scope(name):
        input_shape = input.get_shape().as_list()[-1]
        #limit = tf.sqrt(6.0 / ((np.prod(kernel_size[:]) * tf.dtypes.cast(input_shape,dtype=tf.float32)) + (np.prod(kernel_size[:]) * output_channels)) )
        #w = tf.Variable(tf.random.uniform([*kernel_size, input_shape, output_channels], minval=-limit, maxval=limit), dtype=dtype, name=str(name+'filters'), trainable=True)
        w = tf.get_variable(name=name+'_kernel', shape=[*kernel_size, input_shape, output_channels], dtype=dtype, initializer=kernel_initialiser, trainable=trainable)
        #b = tf.Variable(tf.zeros([output_channels]), dtype=dtype, name=str(name+'bias'), trainable=True)
        b = tf.get_variable(name=name+'_bias', shape=[output_channels], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
        if activation is None:
            h = tf.add(tf.nn.convolution(input,w,padding,strides=strides), b)
        else:
            h = activation(tf.add(tf.nn.convolution(input,w,padding,strides=strides), b))
    return h

def conv2d(input, output_channels, kernel_size, strides, padding, activation=tf.nn.relu, name='convolutional2d_layer', dtype=tf.float32, kernel_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
    # 2D convolutional layer 
    with tf.variable_scope(name):
        input_shape = input.get_shape().as_list()[-1]
        w = tf.get_variable(name=name+'_kernel', shape=[*kernel_size, input_shape, output_channels], dtype=dtype, initializer=kernel_initialiser, trainable=trainable)
        b = tf.get_variable(name=name+'_bias', shape=[output_channels], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
        h = tf.add(tf.nn.conv2d(input,w,strides,padding), b)
        if activation is not None:
            h = activation(h)
    return h


def conv_transpose_layer(input, output_shape, kernel_size, strides, padding, activation=tf.nn.relu, name='conv_transpose_layer', dtype=tf.float32, kernel_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
    with tf.variable_scope(name):
        input_channels = input.get_shape().as_list()[-1]
        output_channels = output_shape[-1]
        stride_shape = [1, *strides, 1]
        w = tf.get_variable(name=name+'_kernel', shape=[*kernel_size, output_channels, input_channels], dtype=dtype, initializer=kernel_initialiser, trainable=trainable)
        b = tf.get_variable(name=name+'_bias', shape=[output_channels], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
        h = tf.nn.conv_transpose(input,w,output_shape,strides,padding)
        if activation is not None:
            h = activation(h)
    return h




def mlp_layer(input, output_size, activation=tf.nn.relu, dtype=tf.float32, name='dense_layer', weight_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
    with tf.variable_scope(name):
        input_shape = input.get_shape().as_list()[-1]
        w = tf.get_variable(name=name+'_weight', shape=[input_shape, output_size], dtype=dtype, initializer=weight_initialiser, trainable=trainable)
        b = tf.get_variable(name=name+'_bias', shape=[output_size], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
        if activation is None:
            h = tf.add(tf.matmul(input,w), b)
        else:
            h = activation(tf.add(tf.matmul(input,w), b))
    return h


class LSTMCell(object):
    def __init__(self, cell_size, input_size=None, dtype=tf.float32, name='lstm_cell', trainable=True):
        self._cell_size = cell_size
        input_size = input_size if input_size is not None else cell_size # input_size == cell_size by default 
        self._input_size = input_size
        with tf.variable_scope(name):
            # input gate
            self._Wxi = tf.get_variable(name=name+'_Wxi', shape=[input_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
            self._Whi = tf.get_variable(name=name+'_Whi', shape=[cell_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
            self._bi = tf.get_variable(name=name+'_bi', shape=[cell_size], dtype=dtype, trainable=trainable, initializer=tf.zeros_initializer)
            # forget gate
            self._Wxf = tf.get_variable(name=name+'_Wxf', shape=[input_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
            self._Whf = tf.get_variable(name=name+'_Whf', shape=[cell_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
            self._bf = tf.get_variable(name=name+'_bf', shape=[cell_size], dtype=dtype, trainable=trainable, initializer=tf.zeros_initializer)
            # output gate 
            self._Wxo = tf.get_variable(name=name+'_Wxo', shape=[input_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
            self._Who = tf.get_variable(name=name+'_Who', shape=[cell_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
            self._bo = tf.get_variable(name=name+'_bo', shape=[cell_size], dtype=dtype, trainable=trainable, initializer=tf.zeros_initializer)
            # cell gate 
            self._Wxc = tf.get_variable(name=name+'_Wxc', shape=[input_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
            self._Whc = tf.get_variable(name=name+'_Whc', shape=[cell_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
            self._bc = tf.get_variable(name=name+'_bc', shape=[cell_size], dtype=dtype, trainable=trainable, initializer=tf.zeros_initializer)
    
    def __call__(self, x, state, mask):
        prev_cell, prev_hidden = state 
        prev_cell *= tf.stack([1-mask for i in range(self._cell_size)], axis=-1)
        prev_hidden *= tf.stack([1-mask for i in range(self._cell_size)], axis=-1)
        f = tf.nn.sigmoid(tf.matmul(x, self._Wxf) + tf.matmul(prev_hidden, self._Whf) + self._bf)
        i = tf.nn.sigmoid(tf.matmul(x, self._Wxi) + tf.matmul(prev_hidden, self._Whi) + self._bi)
        o = tf.nn.sigmoid(tf.matmul(x, self._Wxo) + tf.matmul(prev_hidden, self._Who) + self._bo)
        c = tf.math.tanh(tf.matmul(x, self._Wxc) + tf.matmul(prev_hidden, self._Whc) + self._bc)
        cell = tf.multiply(prev_cell, f) + tf.multiply(i, c)
        hidden = tf.multiply(o, tf.math.tanh(cell))
        return hidden, tf.compat.v1.nn.rnn_cell.LSTMStateTuple(cell, hidden)
    
    def get_initial_state(self, batch_size):
        return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(tf.zeros((batch_size, self._cell_size), dtype=tf.float32), tf.zeros((batch_size, self._cell_size), dtype=tf.float32))
    
    @property
    def state_size(self):
        return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self._cell_size, self._cell_size)
            #if self._state_is_tuple else 2 * self._cell_size)

    @property
    def output_size(self):
        return self._cell_size
        


def dynamic_masked_rnn(cell, X, hidden_init, mask, parallel_iterations=32, swap_memory=False, time_major=True, scope='rnn', trainable=True):
    ''' dynamic masked *hidden state* RNN for sequences that reset part way through an observation 
        e.g. A2C 
        args :
            cell - cell of type tf.nn.rnn_cell
            X - tensor of rank [time, batch, hidden] if time major == True (Default); or [batch, time, hidden] if time major == False
            hidden_init - tensor or placeholder of intial cell hidden state
            mask - tensor or placeholder of length time, for hidden state masking e.g. [True, False, False] will mask first hidden state
            parallel_iterations - number of parallel iterations to run RNN over
            swap_memory - 
            time_major - bool flag to determine order of indices of input tensor 
            scope - tf variable_scope of dynamic RNN loop
            trainable - bool flag whether to perform backpropagation of RNN
    '''
    with tf.variable_scope(scope):

        if not time_major:
            X = tf.transpose(X, perm=[1,0,2])
        
        def _body(t, hidden, output):
            out, hidden = cell(X[t], hidden, mask[t])
            return t+1, hidden, output.write(t, out)
        
        time_steps = tf.shape(X)[0]
        output = tf.TensorArray(tf.float32, size=time_steps, dynamic_size=True)
        t = 0
        t, hidden, output = tf.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_body,
        loop_vars=(t, hidden_init, output),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        back_prop=trainable) # allow flag for reservoir computing 

        #maximum_iterations=time_steps,
        #)

        output = output.stack()
        if not time_major:
            output = tf.transpose(output, perm=[1,0,2])
        
    return output, hidden

def universe_cnn(input, conv_size=32):
    x = input / 255
    h1 = conv2d(x, conv_size, [3,3], [2,2], padding='SAME', name='conv_1', activation=tf.nn.elu)
    h2 = conv2d(h1, conv_size, [3,3], [2,2], padding='SAME', name='conv_2', activation=tf.nn.elu)
    h3 = conv2d(h2, conv_size, [3,3], [2,2], padding='SAME', name='conv_3', activation=tf.nn.elu)
    h4 = conv2d(h3, conv_size, [3,3], [2,2], padding='SAME', name='conv_4', activation=tf.nn.elu)
    fc = flatten(h4)
    return fc

def nips_cnn(input, conv1_size=16 ,conv2_size=21, dense_size=256, padding='VALID'):
    x = input/255
    h1 = conv2d(x,  output_channels=conv1_size, kernel_size=[8,8], strides=[4,4], padding=padding, activation=tf.nn.relu, dtype=tf.float32, name='conv_1')
    h2 = conv2d(h1, output_channels=conv2_size, kernel_size=[4,4], strides=[2,2], padding=padding, activation=tf.nn.relu, dtype=tf.float32, name='conv_2')
    fc = flatten(h2)
    dense = mlp_layer(fc, dense_size, activation=tf.nn.relu)
    return dense

def nature_reservoir(input, conv1_size=32 ,conv2_size=64, conv3_size=64, dense_size=512, padding='VALID'):
    x = input/255
    h1 = conv2d(x,  output_channels=conv1_size, kernel_size=[8,8],  strides=[4,4], padding=padding, activation=tf.nn.relu, dtype=tf.float32, name='conv_1', trainable=False)
    h2 = conv2d(h1, output_channels=conv2_size, kernel_size=[4,4],  strides=[2,2], padding=padding, activation=tf.nn.relu, dtype=tf.float32, name='conv_2', trainable=False)
    h3 = conv2d(h2, output_channels=conv3_size, kernel_size=[3,3],  strides=[1,1], padding=padding, activation=tf.nn.relu, dtype=tf.float32, name='conv_3', trainable=False)
    fc = flatten(h3)
    dense = mlp_layer(fc, dense_size, activation=tf.nn.relu, trainable=False)
    return dense

def nature_cnn(input, conv1_size=32 ,conv2_size=64, conv3_size=64, dense_size=512, padding='VALID', activation=tf.nn.relu):
    x = input/255
    h1 = conv2d(x,  output_channels=conv1_size, kernel_size=[8,8],  strides=[4,4], padding=padding, activation=activation, dtype=tf.float32, name='conv_1')
    h2 = conv2d(h1, output_channels=conv2_size, kernel_size=[4,4],  strides=[2,2], padding=padding, activation=activation, dtype=tf.float32, name='conv_2')
    h3 = conv2d(h2, output_channels=conv3_size, kernel_size=[3,3],  strides=[1,1], padding=padding, activation=activation, dtype=tf.float32, name='conv_3')
    fc = flatten(h3)
    dense = mlp_layer(fc, dense_size, activation=activation)
    return dense

def mlp(x, num_layers=2, dense_size=64, activation=tf.nn.relu, weight_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer):
    for i in range(num_layers):
        x = mlp_layer(x, dense_size, activation=activation, weight_initialiser=weight_initialiser, bias_initialiser=bias_initialiser, name='dense_' + str(i))
    return x

def lstm(input, cell_size=256, fold_output=True, time_major=True):
    hidden_in = tf.placeholder(tf.float32, [None, cell_size], name='hidden_in')
    cell_in = tf.placeholder(tf.float32, [None, cell_size], name='cell_in')
    hidden_tuple = (cell_in, hidden_in)
    
    lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
    #lstm_cell = LSTMCell(cell_size, trainable=False)
    state_in = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(cell_in, hidden_in)

    lstm_output, hidden_out = tf.compat.v1.nn.dynamic_rnn(lstm_cell, input, initial_state=state_in, time_major=time_major)
    if fold_output:
        lstm_output = tf.reshape(lstm_output, shape=[-1, cell_size], name='folded_lstm_output')
    return lstm_output, hidden_tuple, hidden_out


def lstm_masked(input, cell_size, batch_size, fold_output=True, time_major=True, parallel_iterations=32, swap_memory=False, trainable=True):
    hidden_in = tf.placeholder(tf.float32, [None, cell_size], name='hidden_in')
    cell_in = tf.placeholder(tf.float32, [None, cell_size], name='cell_in')
    mask = tf.placeholder(shape=[None, batch_size], dtype=tf.float32, name='mask') # hidden state mask
    hidden_tuple = (cell_in, hidden_in)
    
    input_size = input.get_shape()[-1].value
    lstm_cell = LSTMCell(cell_size, input_size, trainable=trainable)
    state_in = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(cell_in, hidden_in)

    lstm_output, hidden_out = dynamic_masked_rnn(lstm_cell, input, hidden_init=state_in, mask=mask, time_major=time_major,
                     parallel_iterations=parallel_iterations, swap_memory=swap_memory, trainable=trainable)
    if fold_output:
        lstm_output = tf.reshape(lstm_output, shape=[-1, cell_size], name='folded_lstm_output')
    return lstm_output, hidden_tuple, hidden_out, mask


#def echo_rnn(input, cell_size, fold_output=True, time_major=True):
    #rnn_cell = 

def fold_batch(x):
    rows, cols = x.shape[0], x.shape[1]
    y = np.reshape(x, (rows*cols,*x.shape[2:]))
    return y

def unfold_batch(x, batch_size):
    y = np.reshape(x, (-1, batch_size, *x.shape[1:]))
    return y


