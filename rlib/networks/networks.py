import torch
import numpy as np
from typing import List

# def flatten(x, name='flatten'):
#     return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])], name=name)

# def blank(x):
#     return x

# def conv_layer(input, output_channels, kernel_size, strides, padding, activation=tf.nn.relu, name='convolutional_layer', dtype=tf.float32, kernel_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
#     # N-D convolutional layer
#     with tf.variable_scope(name):
#         input_shape = input.get_shape().as_list()[-1]
#         #limit = tf.sqrt(6.0 / ((np.prod(kernel_size[:]) * tf.dtypes.cast(input_shape,dtype=tf.float32)) + (np.prod(kernel_size[:]) * output_channels)) )
#         #w = tf.Variable(tf.random.uniform([*kernel_size, input_shape, output_channels], minval=-limit, maxval=limit), dtype=dtype, name=str(name+'filters'), trainable=True)
#         w = tf.get_variable(name=name+'_kernel', shape=[*kernel_size, input_shape, output_channels], dtype=dtype, initializer=kernel_initialiser, trainable=trainable)
#         #b = tf.Variable(tf.zeros([output_channels]), dtype=dtype, name=str(name+'bias'), trainable=True)
#         b = tf.get_variable(name=name+'_bias', shape=[output_channels], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
#         if activation is None:
#             h = tf.add(tf.nn.convolution(input,w,padding,strides=strides), b)
#         else:
#             h = activation(tf.add(tf.nn.convolution(input,w,padding,strides=strides), b))
#     return h

# def conv2d(input, output_channels, kernel_size, strides, padding, activation=tf.nn.relu, name='convolutional2d_layer', dtype=tf.float32, kernel_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
#     # 2D convolutional layer 
#     with tf.variable_scope(name):
#         input_shape = input.get_shape().as_list()[-1]
#         w = tf.get_variable(name=name+'_kernel', shape=[*kernel_size, input_shape, output_channels], dtype=dtype, initializer=kernel_initialiser, trainable=trainable)
#         b = tf.get_variable(name=name+'_bias', shape=[output_channels], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
#         h = tf.add(tf.nn.conv2d(input,w,strides,padding), b)
#         if activation is not None:
#             h = activation(h)
#     return h

# def conv2d_transpose(input, output_shape, kernel_size, strides, padding, activation=tf.nn.relu, name='conv2d_transpose', dtype=tf.float32, kernel_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
#     with tf.variable_scope(name):
#         input_channels = input.get_shape().as_list()[-1]
#         output_channels = output_shape[-1]
#         stride_shape = [1, *strides, 1]
#         w = tf.get_variable(name=name+'_kernel', shape=[*kernel_size, output_channels, input_channels], dtype=dtype, initializer=kernel_initialiser, trainable=trainable)
#         b = tf.get_variable(name=name+'_bias', shape=[output_channels], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
#         h = tf.nn.conv2d_transpose(input, w , output_shape, stride_shape, padding) + b 
#         if activation is not None:
#             h = activation(h)
#     return h



# def conv_transpose_layer(input, output_shape, kernel_size, strides, padding, activation=tf.nn.relu, name='conv_transpose_layer', dtype=tf.float32, kernel_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
#     with tf.variable_scope(name):
#         input_channels = input.get_shape().as_list()[-1]
#         output_channels = output_shape[-1]
#         stride_shape = [1, *strides, 1]
#         w = tf.get_variable(name=name+'_kernel', shape=[*kernel_size, output_channels, input_channels], dtype=dtype, initializer=kernel_initialiser, trainable=trainable)
#         b = tf.get_variable(name=name+'_bias', shape=[output_channels], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
#         h = tf.nn.conv_transpose(input,w,output_shape,strides,padding) + b 
#         if activation is not None:
#             h = activation(h)
#     return h




# def mlp_layer(input, output_size, activation=tf.nn.relu, dtype=tf.float32, name='dense_layer', weight_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
#     with tf.variable_scope(name):
#         input_shape = input.get_shape().as_list()[-1]
#         w = tf.get_variable(name=name+'_weight', shape=[input_shape, output_size], dtype=dtype, initializer=weight_initialiser, trainable=trainable)
#         b = tf.get_variable(name=name+'_bias', shape=[output_size], dtype=dtype, initializer=bias_initialiser, trainable=trainable)
#         if activation is None:
#             h = tf.add(tf.matmul(input,w), b)
#         else:
#             h = activation(tf.add(tf.matmul(input,w), b))
#     return h


# class LSTMCell(object):
#     def __init__(self, cell_size, input_size=None, dtype=tf.float32, name='lstm_cell', trainable=True):
#         self._cell_size = cell_size
#         input_size = input_size if input_size is not None else cell_size # input_size == cell_size by default 
#         self._input_size = input_size
#         with tf.variable_scope(name):
#             # input gate
#             self._Wxi = tf.get_variable(name=name+'_Wxi', shape=[input_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
#             self._Whi = tf.get_variable(name=name+'_Whi', shape=[cell_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
#             self._bi = tf.get_variable(name=name+'_bi', shape=[cell_size], dtype=dtype, trainable=trainable, initializer=tf.zeros_initializer)
#             # forget gate
#             self._Wxf = tf.get_variable(name=name+'_Wxf', shape=[input_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
#             self._Whf = tf.get_variable(name=name+'_Whf', shape=[cell_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
#             self._bf = tf.get_variable(name=name+'_bf', shape=[cell_size], dtype=dtype, trainable=trainable, initializer=tf.zeros_initializer)
#             # output gate 
#             self._Wxo = tf.get_variable(name=name+'_Wxo', shape=[input_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
#             self._Who = tf.get_variable(name=name+'_Who', shape=[cell_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
#             self._bo = tf.get_variable(name=name+'_bo', shape=[cell_size], dtype=dtype, trainable=trainable, initializer=tf.zeros_initializer)
#             # cell gate 
#             self._Wxc = tf.get_variable(name=name+'_Wxc', shape=[input_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
#             self._Whc = tf.get_variable(name=name+'_Whc', shape=[cell_size, cell_size], dtype=dtype, trainable=trainable, initializer=tf.glorot_uniform_initializer)
#             self._bc = tf.get_variable(name=name+'_bc', shape=[cell_size], dtype=dtype, trainable=trainable, initializer=tf.zeros_initializer)
    
#     def __call__(self, x, state, mask):
#         prev_cell, prev_hidden = state 
#         prev_cell *= tf.stack([1-mask for i in range(self._cell_size)], axis=-1)
#         prev_hidden *= tf.stack([1-mask for i in range(self._cell_size)], axis=-1)
#         f = tf.nn.sigmoid(tf.matmul(x, self._Wxf) + tf.matmul(prev_hidden, self._Whf) + self._bf)
#         i = tf.nn.sigmoid(tf.matmul(x, self._Wxi) + tf.matmul(prev_hidden, self._Whi) + self._bi)
#         o = tf.nn.sigmoid(tf.matmul(x, self._Wxo) + tf.matmul(prev_hidden, self._Who) + self._bo)
#         c = tf.math.tanh(tf.matmul(x, self._Wxc) + tf.matmul(prev_hidden, self._Whc) + self._bc)
#         cell = tf.multiply(prev_cell, f) + tf.multiply(i, c)
#         hidden = tf.multiply(o, tf.math.tanh(cell))
#         return hidden, tf.nn.rnn_cell.LSTMStateTuple(cell, hidden)
    
#     def get_initial_state(self, batch_size):
#         return tf.nn.rnn_cell.LSTMStateTuple(tf.zeros((batch_size, self._cell_size), dtype=tf.float32), tf.zeros((batch_size, self._cell_size), dtype=tf.float32))
    
#     @property
#     def state_size(self):
#         return tf.nn.rnn_cell.LSTMStateTuple(self._cell_size, self._cell_size)
#             #if self._state_is_tuple else 2 * self._cell_size)

#     @property
#     def output_size(self):
#         return self._cell_size
        


# def dynamic_masked_rnn(cell, X, hidden_init, mask, parallel_iterations=32, swap_memory=False, time_major=True, scope='rnn', trainable=True):
#     ''' dynamic masked *hidden state* RNN for sequences that reset part way through an observation 
#         e.g. A2C 
#         args :
#             cell - cell of type tf.nn.rnn_cell
#             X - tensor of rank [time, batch, hidden] if time major == True (Default); or [batch, time, hidden] if time major == False
#             hidden_init - tensor or placeholder of intial cell hidden state
#             mask - tensor or placeholder of length time, for hidden state masking e.g. [True, False, False] will mask first hidden state
#             parallel_iterations - number of parallel iterations to run RNN over
#             swap_memory - bool flag to swap memory between GPU and CPU
#             time_major - bool flag to determine order of indices of input tensor 
#             scope - tf variable_scope of dynamic RNN loop
#             trainable - bool flag whether to perform backpropagation to RNN cell during while loop
#     '''
#     with tf.variable_scope(scope):

#         if not time_major:
#             X = tf.transpose(X, perm=[1,0,2])
        
#         def _body(t, hidden, output):
#             out, hidden = cell(X[t], hidden, mask[t])
#             return t+1, hidden, output.write(t, out)
        
#         time_steps = tf.shape(X)[0]
#         output = tf.TensorArray(tf.float32, size=time_steps, dynamic_size=True)
#         t = 0
#         t, hidden, output = tf.while_loop(
#         cond=lambda time, *_: time < time_steps,
#         body=_body,
#         loop_vars=(t, hidden_init, output),
#         parallel_iterations=parallel_iterations,
#         swap_memory=swap_memory,
#         back_prop=trainable) # allow flag for reservoir computing 

#         #maximum_iterations=time_steps,
#         #)

#         output = output.stack()
#         if not time_major:
#             output = tf.transpose(output, perm=[1,0,2])
        
#     return output, hidden


# def one_to_many_rnn(cell, X, hidden_init, num_timesteps, parallel_iterations=32, swap_memory=False, time_major=True, scope='rnn', trainable=True):
#     ''' one to many RNN for sequences that only have one input observation but many outputs
#         args :
#             cell - cell of type tf.nn.rnn_cell
#             X - tensor of rank [batch, hidden]
#             hidden_init - tensor or placeholder of intial cell hidden state
#             parallel_iterations - number of parallel iterations to run RNN over
#             swap_memory - bool flag to swap memory between GPU and CPU
#             time_major - bool flag to determine order of indices of input tensor 
#             scope - tf variable_scope of dynamic RNN loop
#             trainable - bool flag whether to perform backpropagation to RNN cell during while loop
#     '''
#     with tf.variable_scope(scope):
        
#         def _body(t, input, hidden, output):
#             out, hidden = cell(input, hidden)
#             return t+1, out, hidden, output.write(t, out)
        
#         time_steps = tf.shape(num_timesteps)[0]
#         stacked_output = tf.TensorArray(tf.float32, size=time_steps, dynamic_size=True)
#         t = 0
#         t, X, hidden, stacked_output = tf.while_loop(
#         cond=lambda time, *_: time < time_steps,
#         body=_body,
#         loop_vars=(t, X, hidden_init, stacked_output),
#         parallel_iterations=parallel_iterations,
#         swap_memory=swap_memory,
#         back_prop=trainable) # allow flag for reservoir computing 

#         #maximum_iterations=time_steps,
#         #)

#         stacked_output = stacked_output.stack()
#         if not time_major:
#             stacked_output = tf.transpose(stacked_output, perm=[1,0,2])
        
#     return stacked_output, hidden


# def universe_cnn(input, conv_size=32, trainable=True):
#     x = input / 255
#     h1 = conv2d(x , conv_size, [3,3], [2,2], padding='SAME', name='conv_1', activation=tf.nn.elu, trainable=trainable)
#     h2 = conv2d(h1, conv_size, [3,3], [2,2], padding='SAME', name='conv_2', activation=tf.nn.elu, trainable=trainable)
#     h3 = conv2d(h2, conv_size, [3,3], [2,2], padding='SAME', name='conv_3', activation=tf.nn.elu, trainable=trainable)
#     h4 = conv2d(h3, conv_size, [3,3], [2,2], padding='SAME', name='conv_4', activation=tf.nn.elu, trainable=trainable)
#     fc = flatten(h4)
#     print('cnn output', fc.get_shape().as_list())
#     return fc

# def universe_small_cnn(input, conv_size=32, dense_size=512, trainable=True, scale=True):
#     x = input / 255 if scale else input
#     h1 = conv2d(x , conv_size, [3,3], [2,2], padding='SAME', name='conv_1', activation=tf.nn.elu, trainable=trainable)
#     h2 = conv2d(h1, conv_size, [3,3], [2,2], padding='SAME', name='conv_2', activation=tf.nn.elu, trainable=trainable)
#     h3 = conv2d(h2, conv_size, [3,3], [2,2], padding='SAME', name='conv_3', activation=tf.nn.elu, trainable=trainable)
#     h4 = conv2d(h3, conv_size, [3,3], [2,2], padding='SAME', name='conv_4', activation=tf.nn.elu, trainable=trainable)
#     fc = flatten(h4)
#     dense = mlp_layer(fc, dense_size, activation=tf.nn.relu, trainable=trainable)
#     return dense

# def nips_cnn(input, conv1_size=16 ,conv2_size=32, dense_size=256, padding='VALID'):
#     x = input/255
#     h1 = conv2d(x,  output_channels=conv1_size, kernel_size=[8,8], strides=[4,4], padding=padding, activation=tf.nn.relu, dtype=tf.float32, name='conv_1')
#     h2 = conv2d(h1, output_channels=conv2_size, kernel_size=[4,4], strides=[2,2], padding=padding, activation=tf.nn.relu, dtype=tf.float32, name='conv_2')
#     fc = flatten(h2)
#     dense = mlp_layer(fc, dense_size, activation=tf.nn.relu)
#     return dense

# def nature_cnn(input, conv1_size=32 ,conv2_size=64, conv3_size=64, dense_size=512, padding='VALID', conv_activation=tf.nn.relu, dense_activation=tf.nn.relu, weight_initialiser=tf.initializers.glorot_uniform, scale=True, trainable=True):
#     x = input/255 if scale else input
#     h1 = conv2d(x,  output_channels=conv1_size, kernel_size=[8,8],  strides=[4,4], padding=padding, activation=conv_activation, kernel_initialiser=weight_initialiser, dtype=tf.float32, name='conv_1', trainable=trainable)
#     h2 = conv2d(h1, output_channels=conv2_size, kernel_size=[4,4],  strides=[2,2], padding=padding, activation=conv_activation, kernel_initialiser=weight_initialiser, dtype=tf.float32, name='conv_2', trainable=trainable)
#     h3 = conv2d(h2, output_channels=conv3_size, kernel_size=[3,3],  strides=[1,1], padding=padding, activation=conv_activation, kernel_initialiser=weight_initialiser, dtype=tf.float32, name='conv_3', trainable=trainable)
#     fc = flatten(h3)
#     dense = mlp_layer(fc, dense_size, activation=dense_activation, weight_initialiser=weight_initialiser, trainable=trainable)
#     return dense

# def nature_deconv(input):
#     feat_map = mlp_layer(input, 7*7*64, activation=tf.nn.relu, name='featmap1')
#     feat_map = tf.reshape(feat_map, shape=[-1,7,7,64], name='feature_map')
#     batch_size = tf.shape(feat_map)[0]
#     deconv1 = conv_transpose_layer(feat_map, output_shape=[batch_size,9,9,64], kernel_size=[3,3], strides=[1,1], padding='VALID', activation=tf.nn.relu, name='deconv1')
#     deconv2 = conv_transpose_layer(deconv1, output_shape=[batch_size,20,20,64], kernel_size=[4,4], strides=[2,2], padding='VALID', activation=tf.nn.relu, name='deconv2')
#     deconv3 = conv_transpose_layer(deconv2, output_shape=[batch_size,84,84,4], kernel_size=[8,8], strides=[4,4], padding='VALID', activation=tf.nn.relu, name='deconv3')
#     return deconv3

# def mlp(x, num_layers=2, dense_size=64, activation=tf.nn.relu, weight_initialiser=tf.glorot_uniform_initializer, bias_initialiser=tf.zeros_initializer, trainable=True):
#     for i in range(num_layers):
#         x = mlp_layer(x, dense_size, activation=activation, weight_initialiser=weight_initialiser, bias_initialiser=bias_initialiser, name='dense_' + str(i), trainable=trainable)
#     return x

# def lstm(input, cell_size=256, fold_output=True, time_major=True):
#     hidden_in = tf.placeholder(tf.float32, [None, cell_size], name='hidden_in')
#     cell_in = tf.placeholder(tf.float32, [None, cell_size], name='cell_in')
#     hidden_tuple = (cell_in, hidden_in)
    
#     lstm_cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
#     state_in = tf.nn.rnn_cell.LSTMStateTuple(cell_in, hidden_in)

#     lstm_output, hidden_out = tf.nn.dynamic_rnn(lstm_cell, input, initial_state=state_in, time_major=time_major)
#     if fold_output:
#         lstm_output = tf.reshape(lstm_output, shape=[-1, cell_size], name='folded_lstm_output')
#     return lstm_output, hidden_tuple, hidden_out


# def lstm_masked(input, cell_size, batch_size, fold_output=True, time_major=True, parallel_iterations=32, swap_memory=False, trainable=True):
#     hidden_in = tf.placeholder(tf.float32, [None, cell_size], name='hidden_in')
#     cell_in = tf.placeholder(tf.float32, [None, cell_size], name='cell_in')
#     mask = tf.placeholder(shape=[None, batch_size], dtype=tf.float32, name='mask') # hidden state mask
#     hidden_tuple = (cell_in, hidden_in)
    
#     input_size = input.get_shape()[-1].value
#     lstm_cell = LSTMCell(cell_size, input_size, trainable=trainable)
#     state_in = tf.nn.rnn_cell.LSTMStateTuple(cell_in, hidden_in)

#     lstm_output, hidden_out = dynamic_masked_rnn(lstm_cell, input, hidden_init=state_in, mask=mask, time_major=time_major,
#                      parallel_iterations=parallel_iterations, swap_memory=swap_memory, trainable=trainable)
#     if fold_output:
#         lstm_output = tf.reshape(lstm_output, shape=[-1, cell_size], name='folded_lstm_output')
#     return lstm_output, hidden_tuple, hidden_out, mask


def conv2d_outsize(height, width, kernel_size, stride, padding):
    h_out = ((height + 2*padding[0] - (kernel_size[0] -1) -1) // stride[0]) + 1
    w_out = ((width + 2*padding[1] - (kernel_size[1] -1) -1) // stride[1]) + 1
    return h_out, w_out

class Universe2CNN(torch.nn.Module):
    def __init__(self, input_shape, conv1_size=64, conv2_size=64, conv3_size=64, dense_size=512, padding=[0,0], conv_activation=torch.nn.ELU, dense_activation=torch.nn.ReLU, weight_initialiser=torch.nn.init.xavier_uniform_, scale=True, trainable=True):
        # input_shape [channels, height, width]
        super(Universe2CNN, self).__init__()
        self.scale = scale
        self.input_shape = input_shape
        self.dense_size = dense_size
        
        self.h1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[0], conv1_size, kernel_size=[3,3], stride=[2,2], padding=padding), conv_activation())
        self.h2 = torch.nn.Sequential(torch.nn.Conv2d(conv1_size, conv2_size, kernel_size=[3,3], stride=[2,2], padding=padding), conv_activation())
        self.h3 = torch.nn.Sequential(torch.nn.Conv2d(conv2_size, conv3_size, kernel_size=[3,3], stride=[2,2], padding=padding), conv_activation())
        self.avgpool = torch.nn.AvgPool2d(kernel_size=[3,3], stride=[2,2], padding=padding)
        self.flatten = torch.nn.Flatten()
        h, w, c = self._conv_outsize()
        self.dense = torch.nn.Sequential(torch.nn.Linear(h*w*c, dense_size), dense_activation())
        
        print('final outsize', (c, h, w))
        self.initialiser = weight_initialiser
        self.init_weights()
    
    def init_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            self.initialiser(module.weight)

    def _conv_outsize(self):
        _, h, w = self.input_shape
        h, w = conv2d_outsize(h, w, self.h1[0].kernel_size, self.h1[0].stride, self.h1[0].padding)
        h, w = conv2d_outsize(h, w, self.h2[0].kernel_size, self.h2[0].stride, self.h2[0].padding)
        h, w = conv2d_outsize(h, w, self.h3[0].kernel_size, self.h3[0].stride, self.h3[0].padding)
        h, w = conv2d_outsize(h, w, self.h3[0].kernel_size, self.h3[0].stride, self.h3[0].padding)
        return h, w, self.h3[0].out_channels

    def forward(self, x):
        x = x/255 if self.scale else x
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class UniverseCNN(torch.nn.Module):
    def __init__(self, input_shape, conv1_size=64, conv2_size=64, conv3_size=64, conv4_size=64, padding=[0,0], conv_activation=torch.nn.ELU, weight_initialiser=torch.nn.init.xavier_uniform_, scale=True, trainable=True):
        # input_shape [channels, height, width]
        super(UniverseCNN, self).__init__()
        self.scale = scale
        self.input_shape = input_shape
        
        self.h1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[0], conv1_size, kernel_size=[3,3], stride=[2,2], padding=padding), conv_activation())
        self.h2 = torch.nn.Sequential(torch.nn.Conv2d(conv1_size, conv2_size, kernel_size=[3,3], stride=[2,2], padding=padding), conv_activation())
        self.h3 = torch.nn.Sequential(torch.nn.Conv2d(conv2_size, conv3_size, kernel_size=[3,3], stride=[2,2], padding=padding), conv_activation())
        self.h4 = torch.nn.Sequential(torch.nn.Conv2d(conv2_size, conv3_size, kernel_size=[3,3], stride=[2,2], padding=padding), conv_activation())
        self.flatten = torch.nn.Flatten()
        h, w, c = self._conv_outsize()
        self.dense_size = h*w*c
        print('final outsize', (c, h, w))
        self.initialiser = weight_initialiser
        self.init_weights()
    
    def init_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            self.initialiser(module.weight)

    def _conv_outsize(self):
        _, h, w = self.input_shape
        h, w = conv2d_outsize(h, w, self.h1[0].kernel_size, self.h1[0].stride, self.h1[0].padding)
        h, w = conv2d_outsize(h, w, self.h2[0].kernel_size, self.h2[0].stride, self.h2[0].padding)
        h, w = conv2d_outsize(h, w, self.h3[0].kernel_size, self.h3[0].stride, self.h3[0].padding)
        h, w = conv2d_outsize(h, w, self.h4[0].kernel_size, self.h4[0].stride, self.h4[0].padding)
        return h, w, self.h4[0].out_channels

    def forward(self, x):
        x = x/255 if self.scale else x
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = self.flatten(x)
        return x

class NatureCNN(torch.nn.Module):
    def __init__(self, input_shape, conv1_size=32, conv2_size=64, conv3_size=64, dense_size=512, padding=[0,0], conv_activation=torch.nn.ReLU, dense_activation=torch.nn.ReLU, weight_initialiser=torch.nn.init.xavier_uniform_, scale=True, trainable=True):
        # input_shape [channels, height, width]
        super(NatureCNN, self).__init__()
        self.scale = scale
        self.dense_size = dense_size
        self.input_shape = input_shape
        self.h1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[0], conv1_size, kernel_size=[8,8], stride=[4,4], padding=padding), conv_activation())
        self.h2 = torch.nn.Sequential(torch.nn.Conv2d(conv1_size, conv2_size, kernel_size=[4,4], stride=[2,2], padding=padding), conv_activation())
        self.h3 = torch.nn.Sequential(torch.nn.Conv2d(conv2_size, conv3_size, kernel_size=[3,3], stride=[1,1], padding=padding), conv_activation())
        self.flatten = torch.nn.Flatten()
        h, w, c = self._conv_outsize()
        self.dense = torch.nn.Sequential(torch.nn.Linear(h*w*c, dense_size), dense_activation())
        self.initialiser = weight_initialiser
        self.init_weights()
    
    def init_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            self.initialiser(module.weight)

    def _conv_outsize(self):
        _, h, w = self.input_shape
        h, w = conv2d_outsize(h, w, self.h1[0].kernel_size, self.h1[0].stride, self.h1[0].padding)
        h, w = conv2d_outsize(h, w, self.h2[0].kernel_size, self.h2[0].stride, self.h2[0].padding)
        h, w = conv2d_outsize(h, w, self.h3[0].kernel_size, self.h3[0].stride, self.h3[0].padding)
        return h, w, self.h3[0].out_channels

    def forward(self, x):
        x = x/255 if self.scale else x
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class MaskedRNN(torch.nn.Module):
    ''' dynamic masked *hidden state* RNN for sequences that reset part way through an observation 
        e.g. A2C 
        args :
            cell - cell of type tf.nn.rnn_cell
            X - tensor of rank [time, batch, hidden] if time major == True (Default); or [batch, time, hidden] if time major == False
            hidden_init - tensor or placeholder of intial cell hidden state
            mask - tensor or placeholder of length time, for hidden state masking e.g. [True, False, False] will mask first hidden state
            parallel_iterations - number of parallel iterations to run RNN over
            swap_memory - bool flag to swap memory between GPU and CPU
            time_major - bool flag to determine order of indices of input tensor 
            scope - tf variable_scope of dynamic RNN loop
            trainable - bool flag whether to perform backpropagation to RNN cell during while loop
    '''
    def __init__(self, cell, time_major=True):
        super(MaskedRNN, self).__init__()
        self.cell = cell
        self.time_major = time_major
    
    def forward(self, x, hidden=None, mask=None):
        '''args:
            x - tensor of rank [time, batch, hidden] if time major == True (Default); or [batch, time, hidden] if time major == False
            mask - tensor of rank [time], for hidden state masking e.g. [True, False, False] will mask first hidden state
        returns:
        '''

        if not self.time_major:
            x = x.transpose(1, 0, 2)
        
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[1]).cuda()
        
        outputs = []
        for t in range(x.shape[0]):
            output, hidden = self.cell(x[t], hidden, mask[t])
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs if self.time_major else outputs.transpose(1, 0, 2)
        return outputs, hidden

def lstmgate(cell_size, input_size, trainable=True):
    input_weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(size=[input_size, cell_size], requires_grad=trainable)))
    hidden_weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(size=[cell_size, cell_size], requires_grad=trainable)))
    bias = torch.nn.Parameter(torch.zeros(size=[cell_size], requires_grad=trainable))
    return input_weight, hidden_weight, bias

def gemmlstmgate(cell_size, input_size, trainable=True):
    input_weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(size=[cell_size*4, input_size], requires_grad=trainable)))
    hidden_weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(size=[cell_size*4, cell_size], requires_grad=trainable)))
    bias_input = torch.nn.Parameter(torch.zeros(size=[cell_size*4], requires_grad=trainable))
    bias_hidden = torch.nn.Parameter(torch.zeros(size=[cell_size*4], requires_grad=trainable))
    return input_weight, hidden_weight, bias_input, bias_hidden

class MaskedLSTMCell(torch.nn.Module):
    def __init__(self, cell_size, input_size=None, trainable=True):
        super(MaskedLSTMCell, self).__init__()
        self._cell_size = cell_size
        input_size = input_size if input_size is not None else cell_size # input_size == cell_size by default 
        self._input_size = input_size
        self.Wi, self.Wh, self.bi, self.bh = gemmlstmgate(cell_size, input_size, trainable) # batch gemm
 
    def init_hidden(self, batch_size, dtype, device):
        cell = torch.zeros(batch_size, self._cell_size, dtype=dtype, device=device)
        hidden = torch.zeros(batch_size, self._cell_size, dtype=dtype, device=device)
        return (cell, hidden)

    def forward(self, x, state=None, mask=None):
        if state is None:
            prev_cell, prev_hidden = self.init_hidden(x.shape[0], input.dtype, input.device)
        else:
            prev_cell, prev_hidden = state
        if mask is not None:
            prev_cell *= (1-mask).view(-1, 1)
            prev_hidden *= (1-mask).view(-1, 1)
            
        gates = (torch.matmul(x, self.Wi.t()) + self.bi + torch.matmul(prev_hidden, self.Wh.t())) + self.bh
        i, f, c, o = gates.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)

        cell = prev_cell * f + i * c
        hidden = o * torch.tanh(cell)
        return hidden, (cell, hidden)