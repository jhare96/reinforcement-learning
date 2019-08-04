import tensorflow as tf
import numpy as np
import gym
import os, time
import threading
from A2C import ActorCritic
from networks import*
from SyncMultiEnvTrainer import SyncMultiEnvTrainer
from VecEnv import*


class Curiosity_onenet(object):
    def __init__(self, model, input_size, dense_size, action_size, inv_model_scale, policy_importance, reward_scale, lr=0.001, lr_final=0.001, decay_steps=600000.0, grad_clip=0.5, model_args={} ):
        self.reward_scale, self.inv_model_scale, self.policy_importance = reward_scale, inv_model_scale, policy_importance
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip

        try:
            iterator = iter(input_size)
        except TypeError:
            input_size = (input_size,)
        
        self.state = tf.placeholder("float", shape=[None, *input_size], name="input")
        self.next_state = tf.placeholder("float", shape=[None, *input_size], name="input")

        with tf.variable_scope('Shared_Net'):
            self.phi1 = model(input=self.state,  **model_args)
        
        with tf.variable_scope('Shared_Net', reuse=True):
            self.phi2 = model(input=self.next_state,  **model_args)

        with tf.variable_scope('ActorCritic'):
            
            with tf.variable_scope('critic'):
                self.V = tf.reshape(mlp_layer(self.phi1, 1, name='state_value', activation=None), shape=[-1])
            
            with tf.variable_scope("actor"):
                self.policy_distrib = mlp_layer(self.phi1, action_size, activation=tf.nn.softmax, name='policy_distribution')
                self.actions = tf.placeholder(tf.int32, [None])
                actions_onehot = tf.one_hot(self.actions,action_size)
                
            with tf.variable_scope('losses'):
                self.y = tf.placeholder(dtype=tf.float32, shape=[None])
                Advantage = self.y - self.V
                value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage))

                log_policy = tf.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
                log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
                policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(Advantage))

                entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))

                ac_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        with tf.variable_scope('ICM'):
            with tf.variable_scope('Forward_Model'):
                concat = tf.concat([self.phi1, tf.stop_gradient(self.policy_distrib)], 1, name='state-action-concat')
                f1 = mlp_layer(concat, 256, activation=tf.nn.relu, name='foward_model')
                pred_state = mlp_layer(f1, dense_size, activation=None, name='pred_state')
                self.intr_reward = 0.5 * tf.reduce_mean(tf.square(pred_state - self.phi2), axis=1) * dense_size # mean squared error 
                forward_loss = tf.reduce_mean(self.intr_reward)  # mean across batch
            
            with tf.variable_scope('Inverse_Model'):
                concat = tf.concat([self.phi1, self.phi2], 1, name='state-nextstate-concat')
                max_action_onehot = tf.one_hot(tf.math.argmax(self.policy_distrib, axis=1), action_size) # most likely actions
                pred_action = mlp_layer(concat, action_size, activation=None, name='pred_state')
                inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_action, labels=max_action_onehot)) # batch inverse loss
            
            ICM_loss = (1-inv_model_scale) * inverse_loss + inv_model_scale * forward_loss
        
        with tf.variable_scope('Gradients'):
            self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
            global_step = tf.Variable(0, trainable=False)
            tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)

            self.loss = ICM_loss + policy_importance * ac_loss
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads = tf.gradients(self.loss, weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, weights))

            self.train_op = self.optimiser.apply_gradients(grads_vars, global_step=global_step)

    def forward(self,sess,state):
        return sess.run([self.policy_distrib, self.V], feed_dict = {self.state:state})
    
    def intrinsic_reward(self, sess, state, policy, next_state):
        forward_loss = sess.run(self.intr_reward, feed_dict={self.state:state, self.next_state:next_state})
        intr_reward = forward_loss * self.reward_scale
        return intr_reward
    
    def backprop(self, sess, state, next_state, y, actions, policies):
        feed_dict = {self.state:state, self.actions:actions, self.y:y, self.next_state:next_state}
        _, l = sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l

        