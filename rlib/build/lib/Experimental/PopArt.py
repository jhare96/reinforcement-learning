import numpy as np 
import tensorflow as tf
import datetime, time
import gym
import tensorflow_probability as tfp
from Qvalue import mlp_layer

class ActorCritic_PopArt(object):
    def __init__(self, input_size, h1_size, h2_size, action_size, lr=5e-4):
        self.state = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None])
        self.scale = tf.Variable(tf.ones([1]),trainable=False)
        self.shift = tf.Variable(tf.zeros([1]),trainable=False)
        self.W = tf.Variable(tf.eye(num_rows=h2_size,num_columns=1,), trainable=False)
        self.b = tf.Variable(tf.zeros([1]), trainable=False)


        with tf.variable_scope("critic"):
            h1 = mlp_layer(self.state,h1_size,activation=tf.nn.tanh, name='critic_dense')
            self.V = tf.matmul(h1,self.W) + self.b 
            Advantage = self.y - self.V
            #value_loss = tf.square(Advantage)
            norm_Advantage = self.V - (Advantage - self.shift)/ self.scale
            
        with tf.variable_scope("actor"):
            h1 = mlp_layer(self.state,h1_size,activation=tf.nn.tanh, name='actor_dense')
            self.policy_distrib = mlp_layer(h1,action_size,activation=tf.nn.softmax,name='policy_distribution')
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        
            log_policy = tf.math.log(tf.reduce_sum(tf.multiply(self.policy_distrib, actions_onehot), axis=1))
            
            policy_loss = -log_policy * tf.stop_gradient(norm_Advantage) 
            entropy = tf.reduce_sum(self.policy_distrib * -tf.math.log(self.policy_distrib),axis=1)
            policy_loss -=  0.01 * entropy 

        
        self.loss =  tf.reduce_mean( policy_loss + 0.5 * norm_Advantage )
        self.optimiser = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

        with tf.variable_scope("popart"):
            scale_new = tf.math.reduce_std(self.V, axis=0)
            print('scale new',scale_new.get_shape().as_list())
            shift_new = tf.reduce_mean(self.V, axis=0)
            print('shift new',shift_new.get_shape().as_list())
            W_new = tf.multiply(self.W, self.scale/scale_new)
            b_new = (self.scale * self.b + self.shift -shift_new)/ scale_new
            print('b new',b_new.get_shape().as_list())
            old_vars = [self.W, self.b, self.scale, self.shift]
            new_vars = [W_new, b_new, scale_new, shift_new]
            self.assign = []
            for old, new in zip(old_vars, new_vars):
                self.assign.append(tf.assign(old, new))
            print('assign', self.assign)
            #self.assign = [tf.assign(old,new) for old, new in zip(old_vars, new_vars)]
    
    def rescale(self,sess,states):
        w,b,sc,sh,_ = sess.run([self.W, self.b, self.scale, self.shift, self.assign], feed_dict = {self.state: states})
        print('weight',w,'bias',b,'scale',sc,'shift', sh)
        #tf.print(self.W),tf.print(self.b), tf.print(self.scale), tf.print(self.shift) 
    
    def forward(self,sess,state):
        return sess.run([self.policy_distrib, self.V], feed_dict = {self.state:state})

    def get_policy(self,sess,state):
        return sess.run(self.policy_distrib, feed_dict = {self.state: state})
    
    def get_value(self,sess,state):
        return sess.run(self.V, feed_dict = {self.state: state})

    def backprop(self,sess,state,y,a):
        *_,l = sess.run([self.optimiser, self.loss], feed_dict = {self.state : state, self.y : y, self.actions: a})
        return l





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
            print("pi", pi)
            # v = actorcritic.get_value(sess,s.reshape(1,-1))[0]
            
            a = int(np.random.choice(2))
            #if episode % 100 == 0 :
            #env.render()

            s_, r, done, info = env.step(a)
            v_ = actorcritic.get_value(sess,s_.reshape(1,-1))
            
            if done:
                y[:] = r
            else:
                y[0] = r + 0.99 * v_
            
            actorcritic.rescale(sess,s.reshape(1,-1))
            l = actorcritic.backprop(sess,s.reshape(1,-1),y,np.array([a],dtype=np.int32))
            time.sleep(0.2)
            s = s_
            
            eploss.append(l)
            epscore.append(r)

            # I *= 0.99

            if done:
                s = env.reset()
                I[:] = 1
                print("step %i, epsiode score %f, episode loss %f" %(T,np.sum(epscore),np.mean(eploss)))
                sumscore, sumloss = sess.run([tf_sum_epScore, tf_sum_epLoss], feed_dict = {tf_epScore:np.sum(epscore), tf_epLoss:np.mean(eploss)})
                train_writer.add_summary(sumloss, T)
                train_writer.add_summary(sumscore, T)
                eploss = []
                epscore = [] 
                
            T += 1
                
        
def main():
    sess = tf.Session()
    actorcritic = ActorCritic_PopArt(4,32,32,2)

    current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    train_log_dir = 'logs/'+ "PopArt" + '/' + current_time + '/train'
    train_writer = tf.summary.FileWriter(train_log_dir + '/train', sess.graph)
    tf_epLoss = tf.placeholder('float',name='epsiode_loss')
    tf_epScore =  tf.placeholder('float',name='episode_score')
    tf_sum_epLoss = tf.summary.scalar('epsiode_loss', tf_epLoss)
    tf_sum_epScore = tf.summary.scalar('episode_score', tf_epScore)
    tb_vars = [train_writer,tf_epLoss,tf_epScore,tf_sum_epLoss,tf_sum_epScore]
    
    init = tf.global_variables_initializer()
    sess.run(init)
    env = gym.make("CartPole-v0")
    #episodic_ac(actorcritic,env,sess,tb_vars)
    online_ac(actorcritic,env,sess,tb_vars)

     

if __name__ == "__main__":
    main()