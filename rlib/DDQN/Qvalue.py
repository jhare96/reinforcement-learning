import tensorflow as tf
import numpy as np
import scipy 
import random
import time
import copy
from rlib.networks.networks import conv_layer, mlp_layer


class MLP(object):
    def __init__(self, input_size, h1_size, h2_size, action_size, name):
        with tf.variable_scope(name):
            with tf.variable_scope("weights"):
                self.x = tf.placeholder("float", shape=[None,input_size], name="input")
                w1_limit = tf.sqrt(6.0 / (input_size + h1_size))
                self.w1 = tf.Variable(tf.random_uniform([input_size, h1_size], minval = -w1_limit, maxval = w1_limit), dtype=tf.float32, name="w1", trainable=True)
                self.b1 = tf.Variable(tf.zeros([h1_size]), dtype=tf.float32, name="b1", trainable=True)

                w2_limit = tf.sqrt(6.0 / (h1_size + h2_size))
                self.w2 = tf.Variable(tf.random_uniform([h1_size,h2_size], minval = -w2_limit, maxval = w2_limit), dtype=tf.float32, name="w2", trainable=True)
                self.b2 = tf.Variable(tf.zeros([h2_size]), dtype=tf.float32, name="b2", trainable=True)

                w3_limit = tf.sqrt(6.0 / (h2_size + action_size))
                self.w3 = tf.Variable(tf.random_uniform([h2_size,action_size], minval = -w3_limit, maxval = w3_limit), dtype=tf.float32, name="w3", trainable=True)
                self.b3 = tf.Variable(tf.zeros([action_size]), dtype=tf.float32, name="b3", trainable=True)

            with tf.variable_scope("MLP"):
                h1 = tf.nn.tanh(tf.add(tf.matmul(self.x,self.w1), self.b1))
                h2 = tf.nn.tanh(tf.add(tf.matmul(h1,self.w2), self.b2))
                self.Qsa = tf.add(tf.matmul(h2,self.w3), self.b3)
                self.y = tf.placeholder("float", shape=[None])
                self.actions = tf.placeholder("float", shape=[None,action_size])
                self.Qvalue = tf.reduce_sum(tf.multiply(self.Qsa, self.actions), axis = 1)
            self.loss = tf.reduce_mean(tf.square(self.y - self.Qvalue))
            self.optimizer = tf.train.RMSPropOptimizer(0.0001, momentum=0.95).minimize(self.loss)
                #self.optimizer = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(self.loss)    
                #self.QTarget = tf.placeholder("float", shape=[None,action_size])
            
        
    
    def backprop(self,sess,x,y,a):
        _,l = sess.run([self.optimizer,self.loss], feed_dict = {self.x : x, self.y : y, self.actions : a})
        return l
    
    def forward(self,sess,state):
        if len(state.shape) == 1:
            state = state[np.newaxis,:]
        return sess.run(self.Qsa, feed_dict = {self.x: state})
    


class Qvalue(object):
    def __init__(self, image_size, num_channels, h1_size ,h2_size, h3_size, h4_size, action_size, name, learning_rate=0.00025):
        self.name = name 
        self.image_size = image_size
        self.num_channels = num_channels
        self.action_size = action_size
        self.h1_size, self.h2_size, self.h3_size, self.h4_size = h1_size, h2_size, h3_size, h4_size

        with tf.variable_scope(self.name):
            self.x = tf.placeholder("float", shape=[None,self.image_size, self.image_size, self.num_channels], name="input")
            x = self.x/255
            h1 = conv_layer(x, output_channels=h1_size, kernel_size=[8,8],  strides=[4,4], padding="VALID", dtype=tf.float32, name='conv_1')
            h2 = conv_layer(h1,output_channels=h2_size, kernel_size=[4,4],  strides=[2,2], padding="VALID", dtype=tf.float32, name='conv_2')
            h3 = conv_layer(h2,output_channels=h3_size, kernel_size=[3,3],  strides=[1,1], padding="VALID", dtype=tf.float32, name='conv_3')
            fc = tf.contrib.layers.flatten(h3)
            h4 = mlp_layer(fc,h4_size)
            with tf.variable_scope("State_Action"):
                self.Qsa = mlp_layer(h4,action_size,activation=None)
                self.y = tf.placeholder("float", shape=[None])
                self.actions = tf.placeholder("float", shape=[None,action_size])
                self.Qvalue = tf.reduce_sum(tf.multiply(self.Qsa, self.actions), axis = 1)
                self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.y, predictions=self.Qvalue))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95).minimize(self.loss)

    
    def forward(self,sess,state):
        if len(state[0].shape) == 2:
            state = state[np.newaxis,:]
        return sess.run(self.Qsa, feed_dict = {self.x: state})

    def backprop(self,sess,x,y,a):
        _,l = sess.run([self.optimizer,self.loss], feed_dict = {self.x : x, self.y : y, self.actions:a})
        return l
    
