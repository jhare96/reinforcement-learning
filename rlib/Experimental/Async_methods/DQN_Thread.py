import gym
import numpy as np
import scipy 
import tensorflow as tf
import threading
from collections import deque
import time

UPDATE_TARGET = 40000
NUM_WORKERS = 4
REPLAY_START_SIZE = 5000

main_lock = threading.Lock()

class Qvalue(object):
    def __init__(self, image_size, num_channels, h1_size ,h2_size, h3_size, h4_size, action_size):
        self.image_size = image_size
        self.num_channels = num_channels
        self.action_size = action_size
        self.h1_size, self.h2_size, self.h3_size, self.h4_size = h1_size, h2_size, h3_size, h4_size
        
    def init_var(self):
        self.x = tf.placeholder("float", shape=[None,self.image_size, self.image_size, self.num_channels])
        ##He initialisation
        self.w1 = tf.Variable(tf.random_normal([8, 8, self.num_channels, self.h1_size], stddev = tf.sqrt(2/(self.image_size+self.h1_size))), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([self.h1_size]), dtype=tf.float32)
        
        self.w2 = tf.Variable(tf.random_normal([4, 4, self.h1_size, self.h2_size], stddev = tf.sqrt(2/(self.h1_size+self.h2_size))), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([self.h2_size]), dtype=tf.float32)

        self.w3 = tf.Variable(tf.random_normal([3, 3, self.h2_size, self.h3_size], stddev = tf.sqrt(2/(self.h2_size+self.h3_size))), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([self.h3_size]), dtype=tf.float32)
     
        self.w4 = tf.Variable(tf.random_normal([7**2 * self.h3_size, self.h4_size], stddev = tf.sqrt(2/self.h3_size)), dtype=tf.float32)
        self.b4 = tf.Variable(tf.zeros([self.h4_size]), dtype=tf.float32)

        self.w5 = tf.Variable(tf.random_normal([self.h4_size, self.action_size], stddev = tf.sqrt(2/self.h4_size)), dtype=tf.float32)
        self.b5 = tf.Variable(tf.zeros([self.action_size]), dtype=tf.float32)
        
        self.Qsa = self.forward(self.x)
        self.y = tf.placeholder("float", shape=[None])
        self.actions = tf.placeholder("float", shape=[None,self.action_size])
        self.Qvalue = tf.reduce_sum(tf.multiply(self.Qsa,self.actions),axis=1)
        print("Qvalue shape",self.Qvalue.get_shape().as_list() )
        #self.loss = tf.losses.mean_squared_error(self.y,self.Qvalue)
        self.loss = tf.reduce_mean(tf.math.square(self.y-self.Qvalue))
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)    
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate =0.00025,decay=0.95,momentum=0.0,epsilon=0.01).minimize(self.loss)

    def forward(self,x):
        h1 = tf.nn.relu(tf.add(tf.nn.convolution(x,self.w1,"VALID",strides=[4,4]), self.b1))
        h2 = tf.nn.relu(tf.add(tf.nn.convolution(h1,self.w2,"VALID",strides=[2,2]), self.b2))
        h3 = tf.nn.relu(tf.add(tf.nn.convolution(h2,self.w3,"VALID",strides=[1,1]), self.b3))
        print("h3 shape", h3.get_shape().as_list())
        fc = tf.contrib.layers.flatten(h3)
        print("fc shape", fc.get_shape().as_list())
        print("w3 shape", self.w4.get_shape().as_list())
        h4 = tf.nn.relu(tf.add(tf.matmul(fc,self.w4), self.b4))
        q = tf.add(tf.matmul(h4,self.w5), self.b5)
        print("Qsa shape", q.get_shape().as_list())
        print("q reduce max " ,tf.reduce_max(q,axis=1).get_shape().as_list())
        return q

    def backprop(self,sess,x,y,a):
        feed_dict = {self.x : x, self.y : y, self.actions:a}
        _,l = sess.run([self.optimizer,self.loss], feed_dict = feed_dict)
        return l
    
    def copy(self,Q):
        tf.assign(self.w1,Q.w1)
        tf.assign(self.b1,Q.b1)
        tf.assign(self.w2,Q.w2)
        tf.assign(self.b2,Q.b2)
        tf.assign(self.w3,Q.w3)
        tf.assign(self.b3,Q.b3)
        tf.assign(self.w4,Q.w4)
        tf.assign(self.b4,Q.b4)
        tf.assign(self.w5,Q.w5)
        tf.assign(self.b5,Q.b5)


class ReplayMemory(object):
    def __init__(self,replay_length):
        self.replay_length = replay_length
        self.D = deque(maxlen = self.replay_length)
    def addMemory(self,state,action,reward,next_state,final_state):
        self.D.append([state,action,reward,next_state,final_state])
    def clearMemory(self):
        self.D = []
    def sample(self,mini_batch_size):
        idxs = np.random.choice(len(self.D), mini_batch_size, replace=False)
        states = np.concatenate([self.D[i][0] for i in idxs])
        actions = np.array([self.D[i][1] for i in idxs])
        rewards = np.array([self.D[i][2] for i in idxs])
        next_states = np.concatenate([self.D[i][3] for i in idxs])
        final_state = np.array([self.D[i][4] for i in idxs])
        return (states,actions,rewards,next_states,final_state)

class A3C(object):
    def __init__(self,num_workers,num_actions,env_name,*qvalue_args):
        self.Q = Qvalue(*qvalue_args,num_actions)
        self.QTarget = Qvalue(*qvalue_args,num_actions)
        self.Q.init_var()
        self.QTarget.init_var()
        config = tf.ConfigProto() #GPU 
        config.gpu_options.allow_growth=True #GPU
        #config = tf.ConfigProto(device_count = {'GPU': 0}) #CPU ONLY
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.replayMemory = ReplayMemory(30000)
        self.epsilon = [np.array([1.0],dtype=np.float64)]
        self.n = [np.array([0],dtype=np.int32)]

        self.workers = []
        for i in range(num_workers):
            worker = Worker(i+1,self.Q,self.QTarget,self.sess,self.replayMemory,self.epsilon,self.n,num_actions,env_name)
            self.workers.append(worker)
        
    
    def run(self):
        try:
            for worker in self.workers:
                worker.start()
                time.sleep(1)
        except KeyboardInterrupt:
            for w in workers:
                w.env.close()
                w.daemon=True
                exit()


class Worker(threading.Thread):
    def __init__(self,ID,Q,QTarget,sess,replayMemory,epsilon,n,num_actions,env_name):
        threading.Thread.__init__(self)
        ## Shared attributes -----------------------
        self.Q = Q
        self.QTarget = QTarget
        self.sess = sess
        self.replayMemory = replayMemory
        self.epsilon = epsilon
        self.n = n 
        ## Local attributes ---------------------------------------------------------
        self.env = gym.make(env_name)
        self.images = np.zeros((100,84,84,1))
        self.imgidx = 0
        self.mini_batch_size = 32 * 4
        self.action_size = num_actions
        self.gamma = 0.99

        self.workerID = ID
        #self.daemon = True
    
    def run(self):
        #self.run_env(self.env)
        self.train(self.env,self.Q,self.replayMemory)
    
    def stack_frames(self,image,reset=False):
        if self.imgidx == self.images.shape[0]-1:
            self.images[:3] = self.images[-3:]
            self.images[3] = image
            self.imgidx = 4
            return np.concatenate([self.images[i] for i in range(self.imgidx-4, self.imgidx)]
                        ,axis=2)[np.newaxis,:,:,:]
        
        elif reset == True:
            self.images[:4] = image
            self.imgidx = 4
            return np.concatenate([self.images[i] for i in range(self.imgidx-4, self.imgidx)]
                        ,axis=2)[np.newaxis,:,:,:]
        else:
            self.images[self.imgidx] = image
            self.imgidx += 1
            return np.concatenate([self.images[i] for i in range(self.imgidx-4, self.imgidx)]
                        ,axis=2)[np.newaxis,:,:,:]
    
    def preprocess(self,img):
        img = scipy.misc.imresize(img, [110,84,3])[110-84:,0:84,:]
        img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        img= img/256.00
        img = img.reshape([1,84,84,1])
        return img
    
    def update_Q(self):
        ##sample N points from replay memory D
        batch = self.replayMemory.sample(self.mini_batch_size)
        batch_states = batch[0]
        batch_actions = batch[1]
        batch_rewards = batch[2]
        batch_next_states = batch[3]
        batch_final_state = np.array(batch[4])
        
        y = np.zeros((self.mini_batch_size))
        batch_Qsa = self.sess.run(self.Q.Qsa, feed_dict = {self.Q.x: batch_next_states})
        target_Q = self.sess.run(self.QTarget.Qsa, feed_dict = {self.QTarget.x: batch_next_states})
        actions = np.zeros((self.mini_batch_size,self.action_size))
        for i in range(0,self.mini_batch_size):
            actions[i,batch_actions[i]] = 1
            if batch_final_state[i] == 1:
                y[i] = batch_rewards[i]
            else:
                a = np.argmax(batch_Qsa[i])
                y[i] = batch_rewards[i] + self.gamma * target_Q[i,a] # Double Q Learning 
                #y[i] = batch_rewards[i] + self.gamma*np.max(target_Q[i]) # DQN with fixed weight update

        
        loss = self.Q.backprop(self.sess,batch_states,y,actions)
        return loss
    
    def populate_memory(self,env,num_samples,useQ=False):
        observation = env.reset()
        state = self.stack_frames(self.preprocess(observation),reset=True)
        done = False
        for _ in range(num_samples):

            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            
            if done or _%2000 == 0:
                env.reset()
                next_state = self.stack_frames(self.preprocess(observation))
                self.replayMemory.addMemory(state,action,reward,next_state,1)
            else: 
                with main_lock:
                    env.render()
                next_state = self.stack_frames(self.preprocess(observation))
                self.replayMemory.addMemory(state,action,reward,next_state,0)

            state = next_state
    
    def train(self,env,Q,replayMemory):
        self.populate_memory(env,num_samples=1000)
        for episode in range(1000):
            observation = env.reset()
            state = self.stack_frames(self.preprocess(observation),reset=True)
            episode_reward = []
            episode_loss = 0
            for t in range(2000):
                if self.epsilon[0] > 0.1:
                    self.epsilon[0] -= 9e-6
                if np.random.uniform() <= self.epsilon[0]:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.sess.run(Q.Qsa, feed_dict = {Q.x: state})) 
                
                observation, reward, done, info = env.step(action)
                episode_reward.append(reward)

                reward = np.clip(reward,-1,1)
                with main_lock:
                    env.render()

                if done or t == 2000 -1:
                    env.reset()
                    num_lives = 5
                    next_state = self.stack_frames(np.zeros_like(self.preprocess(observation)),reset=False)
                    self.replayMemory.addMemory(state,action,reward,next_state,1)
                    tot_reward = np.sum(episode_reward)
                    episode_loss += self.update_Q()
                    print('Worker %i Episode %s Total_reward %i epsilon %f average_loss %f Total frames processed %i' %(self.workerID,
                     episode, tot_reward, self.epsilon[0], episode_loss/(t*self.mini_batch_size), self.n[0]))
                    break
                else:
                    next_state = self.stack_frames(self.preprocess(observation))
                    self.replayMemory.addMemory(state,action,reward,next_state,0)
                    state = next_state
                    if t % 4 == 0 :
                        episode_loss += self.update_Q()
                
                if self.n[0] % UPDATE_TARGET == 0:
                    self.n[0]+= 1
                    self.QTarget.copy(self.Q)
                    print("updated Target network")
                else:
                    self.n[0]+=1
            
            if episode % 10 == 0:
                self.saver.save(self.sess, "models/model.ckpt")
                print("saved model")
            



def main():
    print("HELLO")
    #pool = mp.Pool(4)
    print("HELLO2")
    a3c = A3C(4,6,"SpaceInvaders-v0",84,4,32,64,64,512)
    a3c.run()

    # try:
    #     workers = []
    #     q = Q()
    #     for i in range(2):
    #         worker = threading.Thread(target=q.run_env)
    #         #worker = Worker()
    #         worker.start()
    #         workers.append(worker)
    # except KeyboardInterrupt:
    #     print("Caught KeyboardInterrupt, terminating workers")
    #     for w in workers:
    #         w.daemon=True
    #     exit()

if __name__ == "__main__":
    main()
