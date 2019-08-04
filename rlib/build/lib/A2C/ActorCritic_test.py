import numpy as np
import tensorflow as tf
import time, sys, datetime
import gym


class ActorCritic(object):
    def __init__(self, input_size, h1_size, h2_size, action_size, lr=5e-4):
        self.state = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None])
        self.I = tf.placeholder(tf.float32, [1])

        with tf.variable_scope("critic"):
            h1 = self.mlp_layer(self.state,h1_size,activation=tf.nn.relu, name='critic_dense')
            self.V = tf.reshape(self.mlp_layer(h1,1,activation=None,name='state_action_value'),shape=[-1])
            Advantage = self.y - self.V
            value_loss = tf.square(Advantage)
            
        with tf.variable_scope("actor"):
            h1 = self.mlp_layer(self.state,h1_size,activation=tf.nn.relu, name='actor_dense')
            self.policy_distrib = self.mlp_layer(h1,action_size,activation=tf.nn.softmax,name='policy_distribution')
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        
            log_policy = tf.log(tf.reduce_sum(tf.multiply(self.policy_distrib, actions_onehot), axis=1))
            
            policy_loss = -log_policy * tf.stop_gradient(Advantage)  #* self.I
            entropy = tf.reduce_sum(self.policy_distrib * -tf.log(self.policy_distrib),axis=1)
            policy_loss -=  0.01 * entropy 
            
        
       
        print("entropy shape", entropy.get_shape().as_list())
        self.loss =  tf.reduce_mean( policy_loss + 0.5 * value_loss )
        print("loss shape", self.loss.get_shape().as_list())
        self.optimiser = tf.train.AdamOptimizer(lr).minimize(self.loss)
        
    
    def forward(self,sess,state):
        return sess.run([self.policy_distrib, self.V], feed_dict = {self.state:state})

    def get_policy(self,sess,state):
        return sess.run(self.policy_distrib, feed_dict = {self.state: state})
    
    def get_value(self,sess,state):
        return sess.run(self.V, feed_dict = {self.state: state})

    def backprop(self,sess,state,y,a,I):
        *_,l = sess.run([self.optimiser, self.loss], feed_dict = {self.state : state, self.y : y, self.actions: a, self.I: I})
        return l
    
    
    def mlp_layer(self, input, output_size, activation=tf.nn.relu, dtype=tf.float32, name='dense_layer'):
        with tf.variable_scope(name):
            input_shape = input.get_shape().as_list()[-1]
            limit = tf.sqrt(6.0 / (input_shape + output_size) )
            w = tf.Variable(tf.random_uniform([input_shape, output_size], minval = -limit, maxval = limit), dtype=dtype, name=str(name+'_w'), trainable=True)
            b = tf.Variable(tf.zeros([output_size]), dtype=dtype, name=str(name+'_b'), trainable=True)
            if activation is None:
                h = tf.add(tf.matmul(input,w), b)
            else:
                h = activation(tf.add(tf.matmul(input,w), b))
        return h

def episodic_ac(actorcritic,env,sess,tensorboard_vars):
    train_writer,tf_epLoss,tf_epScore,tf_sum_epLoss,tf_sum_epScore = tensorboard_vars
    s = env.reset()
    epscore =[]
    eploss = []
    episode = 0
    I = np.array([1.0], dtype=np.float32)
    y = np.zeros((1))
    for episode in range(4000):
        values = []
        actions = []
        rewards = []
        states = []

        for t in range(2):
            pi = actorcritic.get_policy(sess,s.reshape(1,-1))
            #v = actorcritic.get_value(sess,s)[0]
            
            a = int(np.random.choice(2,p=pi[0]))
            #env.render()

            s_, r, done, info = env.step(a)
            
            I *= 0.99
            #values.append(v)
            actions.append(a)
            rewards.append(r)
            states.append(s)
        
            epscore.append(r)

            
            if done:
                I[:] =1
                s = env.reset()

                break

            s = s_    
  
        R = np.zeros((len(rewards)))
        if done:
            value = 0
        else:
            value = actorcritic.get_value(sess,s.reshape(1,-1))
        for i in reversed(range(len(rewards))):
            value = rewards[i] + 0.99 * value
            R[i] = value
            
        
        actions = np.array(actions)
        states = np.stack(states, axis=0)
        values = np.array(values)
        #advantage = R - values
        l = actorcritic.backprop(sess,states,R,actions,I)
        eploss.append(l)
        print("episode %i, epsiode score %f, episode loss %f" %(episode,np.sum(epscore),np.mean(eploss)))
        sumscore, sumloss = sess.run([tf_sum_epScore, tf_sum_epLoss], feed_dict = {tf_epScore:np.sum(epscore), tf_epLoss:np.mean(eploss)})
        train_writer.add_summary(sumloss, episode)
        train_writer.add_summary(sumscore, episode)
        eploss = []
        epscore = [] 


def online_ac(actorcritic,env,sess,tensorboard_vars):
    train_writer,tf_epLoss,tf_epScore,tf_sum_epLoss,tf_sum_epScore = tensorboard_vars
    s = env.reset()
    epscore =[]
    eploss = []
    episode = 0
    I = np.array([1.0], dtype=np.float32)
    y = np.zeros((1))
    T = 0
    for episode in range(3000):
        for t in range(int(1e4)):
            pi = actorcritic.get_policy(sess,s.reshape(1,-1))
            # v = actorcritic.get_value(sess,s.reshape(1,-1))[0]
            
            a = int(np.random.choice(2,p=pi[0]))
            #if episode % 100 == 0 :
            #env.render()

            s_, r, done, info = env.step(a)
            v_ = actorcritic.get_value(sess,s_.reshape(1,-1))
            
            if done:
                y[:] = r
            else:
                y[0] = r + 0.99 * v_

            l = actorcritic.backprop(sess,s.reshape(1,-1),y,np.array([a],dtype=np.int32),I)
            s = s_
            
            eploss.append(l)
            epscore.append(r)

            # I *= 0.99

            if done:
                s = env.reset()
                I[:] = 1
                print("step %i, epsiode score %f, episode loss %f" %(T,np.sum(epscore),np.mean(eploss)))
                break
                sumscore, sumloss = sess.run([tf_sum_epScore, tf_sum_epLoss], feed_dict = {tf_epScore:np.sum(epscore), tf_epLoss:np.mean(eploss)})
                train_writer.add_summary(sumloss, episode)
                train_writer.add_summary(sumscore, episode)
                eploss = []
                epscore = [] 
                
            T += 1
                
        
def main():
    sess = tf.Session()
    actorcritic = ActorCritic(4,32,32,2)

    current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    train_log_dir = 'logs/'+ "ActorCritic" + '/' + current_time + '/train'
    train_writer = tf.summary.FileWriter(train_log_dir + '/train', sess.graph)
    tf_epLoss = tf.placeholder('float',name='epsiode_loss')
    tf_epScore =  tf.placeholder('float',name='episode_score')
    tf_sum_epLoss = tf.summary.scalar('epsiode_loss', tf_epLoss)
    tf_sum_epScore = tf.summary.scalar('episode_score', tf_epScore)
    tb_vars = [train_writer,tf_epLoss,tf_epScore,tf_sum_epLoss,tf_sum_epScore]
    
    init = tf.global_variables_initializer()
    sess.run(init)
    env = gym.make("CartPole-v0")
    episodic_ac(actorcritic,env,sess,tb_vars)
    #online_ac(actorcritic,env,sess,tb_vars)

     

if __name__ == "__main__":
    main()