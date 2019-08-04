import gym
import tensorflow as tf
import numpy as np
import scipy 
import time
##Kautenja super mario bros
#from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
#import gym_super_mario_bros
#from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
#import multiprocessing as mp
#import resource
from VecEnv import *
import sys, os, datetime
from collections import deque
from ReplayMemory import replayMemory, FrameBuffer, NumpyReplayMemory
from Qvalue import Qvalue, MLP
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"


    
class DoubleDQN(object):
    def __init__(self, Qvalue_type, *Qvalue_args, env, file_loc, action_size, train_freq, replay_length, mini_batch_size=32, gamma=0.99,
                 total_steps=50e6, save=True, epsilon_start=1.0, epsilon_final=0.1, epsilon_steps=1e6, update_target_freq=10000, validate_freq=1e5,
                 model_dir='logs/DDQN/', modelname='DDQN'):
        self.image_size, self.num_stacked = Qvalue_args[:2]
        self.save = save
        self.train_freq = train_freq
        self.action_size = action_size
        self.env = env
        if Qvalue_type.lower() == "cnn":
            self.pixels = True
            self.Q = Qvalue(*Qvalue_args, action_size=action_size, name="Q")
            self.QTarget = Qvalue(*Qvalue_args, action_size=action_size, name="QTarget")
        else:
            self.pixels = False
            self.Q = MLP(*Qvalue_args,action_size=action_size,name="Q")
            self.QTarget = MLP(*Qvalue_args,action_size=action_size,name="QTarget")
        
        self.update_weights = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='QTarget'), tf.trainable_variables('Q'))]


        config = tf.ConfigProto() #GPU 
        config.gpu_options.allow_growth=True #GPU
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #config = tf.ConfigProto(device_count = {'GPU': 0}) #CPU ONLY
        self.sess = tf.Session(config=config)

        self.gamma = gamma
        self.total_steps = int(total_steps)
        self.epsilon = epsilon_start
        self.epsilon_step = (epsilon_start - epsilon_final) / epsilon_steps
        self.epsilon_steps = epsilon_steps
        self.mini_batch_size = mini_batch_size
        self.replay_length = replay_length
        self.update_freq = update_target_freq
        self.validate_freq = validate_freq


        self.model_dir = model_dir
        self.modelname = modelname

        current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        train_log_dir = train_log_dir + current_time + '/train'
        
        tf_epLoss = tf.compat.v1.placeholder('float',name='epsiode_loss')
        tf_epReward =  tf.compat.v1.placeholder('float',name='episode_reward')
        self.tf_placeholders = (tf_epLoss,tf_epReward)

        tf_sum_epLoss = tf.compat.v1.summary.scalar('epsiode_loss', tf_epLoss)
        tf_sum_epReward = tf.compat.v1.summary.scalar('episode_reward', tf_epReward)
        self.tf_summary_scalars= (tf_sum_epLoss,tf_sum_epReward)
        
        self.train_writer = tf.compat.v1.summary.FileWriter(train_log_dir, self.sess.graph)

        

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()

        
        #store transition action state in memory
        #self.replayMemory = NumpyReplayMemory(self.replay_length,self.image_size,self.image_size,self.num_stacked)
        self.replayMemory = replayMemory(self.replay_length, pixels=self.pixels)
        #self.framebuffer = FrameBuffer(self.replay_length,self.image_size,self.image_size,self.num_stacked)

    
    def load_weights(self,modelname, model_dir="models/"):
        if os.path.exists(model_dir + modelname+ ".ckpt"+ ".meta"):
            self.saver.restore(self.sess,model_dir+modelname+".ckpt")
            print("loaded:", model_dir+modelname)
        else:
            print(model_dir + modelname, " does not exist")



    def populate_memory(self,env,num_samples,useQ=False,render=True):
        observation = env.reset()
        env.render()
        state = self.preprocess(observation,reset=True)
        for _ in range(num_samples):

            if np.random.uniform() <= self.epsilon and useQ == True:
                print("sampled from Q function ")
                action = np.argmax(self.sess.run(self.Q.Qsa, feed_dict = {self.Q.x: state}))
            else:
                action = env.action_space.sample()
            
            observation, reward, done, info = env.step(action)
            reward = np.clip(reward,-1,1)
            next_state = self.preprocess(observation)
            self.replayMemory.addMemory(state,action,reward,next_state,done)
            state = next_state

            if render: 
                env.render()
            
            if done or _%2000 == 0:
                observation = env.reset()
                state = self.preprocess(observation,reset=True)
        
        print("Finished populating memory")
    
    def update_Q(self):
        ##sample N points from replay memory D
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_final_states = self.replayMemory.sample(self.mini_batch_size)
        #print("nxt img")
        #for i in range(self.mini_batch_size):
            #scipy.misc.imshow(batch_states[i,:,:,0])
        #print("batch_states shape", batch_states.shape)
        y = np.zeros((self.mini_batch_size))
        batch_Qsa = self.Q.forward(self.sess, batch_next_states)   #self.sess.run(self.Q.Qsa, feed_dict = {self.Q.x: batch_next_states})
        target_Q = self.QTarget.forward(self.sess, batch_next_states)    #self.sess.run(self.QTarget.Qsa, feed_dict = {self.QTarget.x: batch_next_states})
        actions = np.zeros((self.mini_batch_size,self.action_size))
        for i in range(0,self.mini_batch_size):
            actions[i,batch_actions[i]] = 1
            if batch_final_states[i] == 1:
                y[i] = batch_rewards[i]
            else:
                a = np.argmax(batch_Qsa[i])
                y[i] = batch_rewards[i] + self.gamma * target_Q[i,a] # Double Q Learning
                
                #print("Q[i,a]", batch_Qsa[i,:])
                #y[i] = batch_rewards[i] + self.gamma*np.max(target_Q[i]) # DQN with fixed weight update
                #batch_state_Qsa[i,a] = y[i]
        #print("batch rewards", batch_rewards)
        #print("y", y)
        loss = self.Q.backprop(self.sess,batch_states,y,actions)
        return loss
    
    def preprocess(self,state,reset=False):
        if self.pixels:
            return self.replayMemory.stack_frames(state,reset=reset)
        else:
            return state

    def validate(self,env,num_episodes,max_steps=10000,test_exploration=0.001,print_all=False):
        env = self.env
        episode_scores = []
        for episode in range(num_episodes):
            episode_reward = []
            observation = env.reset()
            state = self.preprocess(observation,reset=True)
            for t in range(max_steps):
                if np.random.uniform() <= test_exploration:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.Q.forward(self.sess,state))
                
                observation, reward, done, info = env.step(action)
                next_state = self.preprocess(observation)
                state = next_state
                env.render()
                
                episode_reward.append(reward)

                if done or t == max_steps -1:
                    tot_reward = np.sum(episode_reward)
                    if print_all:
                        print('Episode %s Total_reward %i epsilon %f'
                        %(episode, tot_reward, self.epsilon))
                    episode_scores.append(tot_reward)
                    break
        
        print("Validation Average score over %i episodes: %f" %(num_episodes,np.mean(episode_scores)))
        return np.mean(episode_scores)
    
    def epsilon_schedule(self, t):
        if t < self.epsilon_steps:
                self.epsilon -= self.epsilon_step
        else:
            self.epsilon = self.epsilon_final
    
    def train(self, render=False):

        env = self.env

        tf_epLoss, tf_epScore, = self.tf_placeholders
        tf_sum_epLoss, tf_sum_epScore, = self.tf_summary_scalars
    
        #self.QTarget.copy(self.Q)
        self.populate_memory(env,self.replay_length//10,render=False)
        
        start = time.time()
        episode = 0
        episode_reward = []
        episode_score = []
        episode_loss = []
        state = env.reset()
        terminal = False
        lives = 0
        for t in range(self.total_steps):
            self.epsilon_schedule(t)
            if np.random.uniform() < self.epsilon:
                action = env.action_space.sample()
            else:
                state_action = self.Q.forward(self.sess,state)
                action = np.argmax(state_action)
            
            next_state, reward, done, info = env.step(action)

            if info['ale.lives'] < lives:
                terminal = True
            else:
                lives = info['ale.lives']
                terminal = done
            
            episode_score.append(reward)
            reward = np.clip(reward,-1,1)
            episode_reward.append(reward)
            self.replayMemory.addMemory(state,action,reward,next_state,terminal)
            state = next_state

            if render:
                env.render()
            
            if done or t == self.total_steps -1:
                episode += 1
                #Reset
                state = env.reset()
                
            
            if t % self.train_freq == 0:
                episode_loss.append(self.update_Q())
            
            if t % self.validate_freq == 0 and t > 0:
                avg_loss = np.mean(episode_loss)
                score = self.validate(env,20)
                time_taken = time.time()-start
                fps = update_freq / time_taken
                print(update_freq, " frames time taken: ", time_taken, ', fps:', fps)
                print('Episode %s, total_step %i, validation score %f, epsilon %f, average_loss %f, fps %f'
                    %(episode, t, score, self.epsilon, avg_loss , fps))
                sumscore, sumloss = self.sess.run([tf_sum_epScore, tf_sum_epLoss], feed_dict = {tf_epScore:score, tf_epLoss:avg_loss})
                self.train_writer.add_summary(sumloss, t)
                self.train_writer.add_summary(sumscore, t)
                start = time.time()

            if t % self.update_freq == 0 and t > 0:
                self.sess.run([self.update_weights])
                print("updated Target network")
                if self.save:
                    self.saver.save(self.sess, str(model_dir + modelname + ".ckpt") )
            
            
                

##main ----------------------------------------------------------------------------------------------------               
def main():
    #soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    #resource.setrlimit(resource.RLIMIT_AS, (16e6* 0.85, hard))
    
    #env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    #env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    env = AtariEnv(gym.make('PongDeterministic-v4'), k=4, episodic=False, reset=True, clip_reward=True, Noop=True)
    env = gym.make('MountainCar-v0')
    env.reset()
    env.render()
    print(env.action_space)
    CNN_para = [84,4,32,64,64,512]
    MLP_para = [4,64,64]
    DQN = DoubleDQN("MLP", *MLP_para, env=env, action_size=6, train_freq=4, replay_length=500000, mini_batch_size=32)
    
    model_dir = "models/DDQN/"
    modelname = "DDQN_ABC"
    DQN.load_weights(modelname, model_dir)
    DQN.train(env,int(5e6+1),)
    DQN.validate(env,20)

if __name__== "__main__":
    main()
