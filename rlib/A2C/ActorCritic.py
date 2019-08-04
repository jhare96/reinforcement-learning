import tensorflow as tf
import numpy as np
from networks.networks import mlp_layer, lstm, lstm_masked


class ActorCritic_LSTM(object):
    def __init__(self, model_head, input_shape, action_size, num_envs, cell_size,
                 lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip = 0.5, opt=False, **model_head_args):
        self.lr, self.lr_final = lr, lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.cell_size = cell_size
        self.num_envs = num_envs
        self.sess = None

        with tf.variable_scope('input'):
            self.state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='time_batch_state') # [time*batch, *input_shape]

        with tf.variable_scope('encoder_network'):
            self.dense = model_head(self.state, **model_head_args)
            dense_size = self.dense.get_shape()[1].value
            unfolded_state = tf.reshape(self.dense, shape=[-1, num_envs, dense_size], name='unfolded_state')
        
        with tf.variable_scope('lstm'):
            #self.lstm_output, self.hidden_in, self.hidden_out = lstm(unfolded_state, cell_size=cell_size, fold_output=True, time_major=True)
            self.lstm_output, self.hidden_in, self.hidden_out, self.mask = lstm_masked(unfolded_state, cell_size=cell_size, batch_size=num_envs, fold_output=True, time_major=True)

        with tf.variable_scope('critic'):
            self.V = tf.reshape( mlp_layer(self.lstm_output, 1, name='state_value', activation=None), shape=[-1])

        
        with tf.variable_scope("actor"):
            self.policy_distrib = mlp_layer(self.lstm_output, action_size, activation=tf.nn.softmax, name='policy_distribution')
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.R = tf.placeholder(dtype=tf.float32, shape=[None])
            Advantage = self.R - self.V
            value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage))

            log_policy = tf.math.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
            log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
            policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(Advantage))

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
    
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        self.loss =  policy_loss + 0.5 * value_loss - 0.01 * entropy

        if opt:
            global_step = tf.Variable(0, trainable=False)
            tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)

            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)

            
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)
    
    def get_initial_hidden(self, batch_size):
        return np.zeros((batch_size, self.cell_size)), np.zeros((batch_size, self.cell_size))
    
    def reset_batch_hidden(self, hidden, idxs):
        return hidden * np.stack([idxs for i in range(self.cell_size)], axis=1)

    def forward(self, state, hidden):
        mask = np.zeros((1, self.num_envs)) # state = [time, batch, ...]
        feed_dict = {self.state:state, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1], self.mask:mask}
        policy, value, hidden = self.sess.run([self.policy_distrib, self.V, self.hidden_out], feed_dict = feed_dict)
        return policy, value, hidden

    def backprop(self, state, y, a, hidden, dones):
        feed_dict = {self.state : state, self.R : R, self.actions: a, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1], self.mask:dones}
        *_,l = self.sess.run([self.train_op, self.loss], )
        return l
    
    def set_session(self, sess):
        self.sess = sess