import gym
import numpy as np
import scipy 
import tensorflow as tf
import threading
from collections import deque
from Qvalue import Qvalue
from ReplayMemory import NumpyReplayMemory
import time, datetime

main_lock = threading.Lock()

class AsyncQvalue(Qvalue):
    def __init__(self, image_size, num_channels, h1_size ,h2_size, h3_size, h4_size, action_size, name, learning_rate=0.00025, max_grad_norm=1.0):
        super(Qvalue, self).__init__()
        self.name = name 
        self.image_size = image_size
        self.num_channels = num_channels
        self.action_size = action_size
        self.h1_size, self.h2_size, self.h3_size, self.h4_size = h1_size, h2_size, h3_size, h4_size

        with tf.variable_scope(self.name):
            self.x = tf.placeholder("float", shape=[None,self.image_size, self.image_size, self.num_channels], name="input")
            x = self.x/255
            h1 = self.conv_layer(x,  *[8,8], output_channels=h1_size, strides=[4,4], padding="VALID", dtype=tf.float32, name='conv_1')
            h2 = self.conv_layer(h1, *[4,4], output_channels=h2_size, strides=[2,2], padding="VALID", dtype=tf.float32, name='conv_2')
            h3 = self.conv_layer(h2, *[3,3], output_channels=h3_size, strides=[1,1], padding="VALID", dtype=tf.float32, name='conv_3')
            fc = tf.contrib.layers.flatten(h3)
            h4 = self.mlp_layer(fc,h4_size)
            with tf.variable_scope("State_Action"):
                self.Qsa = self.mlp_layer(h4,action_size,activation=None)
                self.y = tf.placeholder("float", shape=[None])
                self.actions = tf.placeholder("float", shape=[None,action_size])
                self.Qvalue = tf.reduce_sum(tf.multiply(self.Qsa, self.actions), axis = 1)
                self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.y, predictions=self.Qvalue))
            
            
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95)
            tvs = tf.trainable_variables(scope=self.name)
            #grads = self.optimizer.compute_gradients(self.loss, tvs)
            #self.grads = [(tf.clip_by_norm(grad, max_grad_norm), var) for grad, var in grads]


            #self.accum_grads = tf.pla
            #self.train_step = self.optimizer.apply_gradients([(accum_vars[i]/6, gv[1]) for i, gv in enumerate(grads)])
            # with tf.variable_scope("gradients"):
            #     ## Retrieve all trainable variables you defined in your graph
            #     tvs = tf.trainable_variables()
            #     #grads = tf.gradients(self.loss, tvs)
            #     ## Creation of a list of variables with the same shape as the trainable ones
            #     # initialized with 0s
            #     accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
            #     self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
            #     print("accumalted variables", accum_vars)
            #     ## Calls the compute_gradients function of the optimizer to obtain... the list of gradients
            #     grads = self.optimizer.compute_gradients(self.loss, tvs)
            #     #grads = [(tf.clip_by_norm(grad, max_grad_norm), var) for grad, var in grads]
            #     ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
            #     self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads)]

            #     ## Define the training step (part with variable value update)
            #     self.train_step = self.optimizer.apply_gradients([(accum_vars[i]/6, gv[1]) for i, gv in enumerate(grads)])

    def update_gradients(self,sess,grads):


        
    def calculate_gradients(self,sess,state,y,a):
        grads, l = sess.run([self.grads, self.loss], feed_dict = {self.x : state, self.y : y, self.actions:a})
        return l, self.grads
        
    def update_step(self,sess):
        sess.run(self.train_step)
        sess.run(self.zero_ops)



class AsyncDQN(object):
    def __init__(self, num_workers, num_actions, env_name, file_loc, workers_to_render=0, *qvalue_args):
        self.Q = AsyncQvalue(*qvalue_args,action_size=num_actions,name='Q_Master', learning_rate=1e-5)
        self.QTarget = Qvalue(*qvalue_args,action_size=num_actions,name='QTarget', learning_rate=1e-5)
        config = tf.ConfigProto() #GPU 
        config.gpu_options.allow_growth=True #GPU
        #config = tf.ConfigProto(device_count = {'GPU': 0}) #CPU ONLY
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        
        self.update_weights = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='QTarget'), tf.trainable_variables('Q'))]

        self.batch = []
        self.epsilon = np.array([1.0],dtype=np.float64)
        self.T = np.array([0],dtype=np.int32)
        validating = np.array([False], dtype=np.bool)
        updating_target = np.array([False], dtype=np.bool)
        updating_async = np.array([False], dtype=np.bool)
        sync_variables = [validating, updating_target, updating_async]
        start_time = [time.time()]

        #Tensorboard Variables
        model_dir, modelname = file_loc
        current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        train_log_dir = 'logs/Async_DQN/'+ modelname + '/' + current_time + '/train'
        train_writers, tf_placeholders, tf_summary_scalars = [],[],[]
        tf_epLoss = tf.placeholder('float',name='epsiode_loss')
        tf_epReward =  tf.placeholder('float',name='episode_reward')
        tf_epScore =  tf.placeholder('float',name='episode_score')
        tf_valReward =  tf.placeholder('float',name='validation_reward')
        tf_placeholders.append([tf_epLoss,tf_epReward,tf_epScore,tf_valReward])

        tf_sum_epLoss = tf.summary.scalar('epsiode_loss', tf_epLoss)
        tf_sum_epReward = tf.summary.scalar('episode_reward', tf_epReward)
        tf_sum_epScore = tf.summary.scalar('episode_score', tf_epScore)
        tf_sum_valReward = tf.summary.scalar('validation_reward', tf_valReward)
        tf_summary_scalars.append([tf_sum_epLoss,tf_sum_epReward,tf_sum_epScore,tf_sum_valReward])
        
        #for i in range(num_workers):
        train_writers.append(tf.summary.FileWriter(train_log_dir, self.sess.graph))
            
        self.workers = []
        for i in range(num_workers):
            if i < workers_to_render:
                render = True
            else:
                render = False
            
            worker = Worker(i+1,qvalue_args,self.sess,self.batch,4,self.epsilon,self.T,
                            num_actions,env_name,file_loc,train_writers[0],tf_placeholders,tf_summary_scalars,
                            self.saver,self.update_weights,start_time,render,sync_variables)
            self.workers.append(worker)
        
    
    def run(self):
        try:
            for worker in self.workers:
                worker.start()
                #time.sleep(2)
        except KeyboardInterrupt:
            for w in workers:
                w.env.close()
                w.daemon=True
                exit()


class Worker(threading.Thread):
    def __init__(self,ID,qvalue_args,QTarget,sess,batch,k,epsilon,T,num_actions,env_name,file_loc,train_writer,
                    tf_placeholders,tf_summary_scalars,saver,update_weights,start,render,sync_variables):
        threading.Thread.__init__(self)
        ## Shared attributes -----------------------
        self.sess = sess
        self.saver = saver
        self.update_target = update_weights
        self.batch = batch
        self.epsilon = epsilon
        self.T = T
        self.start_time = start
        #Flags to ensure multiple threads don't update or validate simultaneously 
        self.validating, self.updating_target, self.updating_async = sync_variables 
        ## Local attributes ---------------------------------------------------------
        with tf.device('/cpu:0'):
            self.Q = Qvalue(*qvalue_args, action_size=action_size, name='Q_'+ str(self.workerID))
        
        self.QTarget = QTarget
        
        self.update_local = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='Q_' + str(self.workerID) ), tf.trainable_variables('Q_Master'))]


        self.env = gym.make(env_name)
        self.stack_size = k 
        self._stacked_frames = deque([], maxlen=k)
        #self.mini_batch_size = 32 * 4
        self.action_size = num_actions
        self.gamma = 0.99
        self.workerID = ID
        self.render = render
        #self.epsilon = np.random.uniform()
        self.epsilon_final = 0.1
        self.epsilon_steps = 1e6
        self.epsilon_step = (self.epsilon - self.epsilon_final) / self.epsilon_steps
        #self.daemon = True
        #misc---------------------------------------
        self.model_dir, self.modelname = file_loc
        self.train_writer = train_writer
        self.tf_placeholders = tf_placeholders
        self.tf_summary_scalars = tf_summary_scalars

    def run(self):
        #self.run_env(self.env)
        self.train(self.env,render=self.render)
    
    def preprocess(self,frame):
        frame = scipy.misc.imresize(frame, [110,84,3])[110-84:,0:84,:]
        frame = np.dot(frame[...,:3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        return frame

    def stack_frames(self,frame,reset=False):
        frame = self.preprocess(frame)
        if reset:
            for _ in range(self.stack_size):
                self._stacked_frames.append(frame)
            stacked_state = np.stack(self._stacked_frames, axis=2)
        else:
            self._stacked_frames.append(frame)
            stacked_state = np.stack(self._stacked_frames, axis=2)
        return stacked_state
    
    def update_Q(self, state, action, reward, next_state, terminal):
        ##sample N points from replay memory D
        #batch = self.replayMemory.sample(self.mini_batch_size)
        self.batch = []
        
        y = np.zeros((1))
        Qsa = self.Q.forward(self.sess, state[np.newaxis,:,:,:])
        target_Q = self.QTarget.forward(self.sess, state[np.newaxis,:,:,:])
        actions = np.zeros((1,self.action_size))
        actions[0,action] = 1
        if terminal:
            y[0] = reward
        else:
            a = np.argmax(Qsa)
            y[0] = reward + self.gamma * target_Q[0,a] # Double Q Learning 
            #y[i] = batch_rewards[i] + self.gamma*np.max(target_Q[i]) # DQN with fixed weight update

        loss, grads = self.Q.calculate_gradients(self.sess,state[np.newaxis,:,:,:],y,actions)
        return loss, grads
    
    def validate(self,env,num_episodes,max_steps=10000,test_exploration=0.001,print_all=False):
        episode_scores = []
        for episode in range(num_episodes):
            episode_reward = []
            observation = env.reset()
            state = self.stack_frames(observation, reset=True)
            for t in range(max_steps):
                if np.random.uniform() <= test_exploration:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.Q.forward(self.sess,state))
                
                observation, reward, done, info = env.step(action)
                next_state = self.stack_frames(observation)
                state = next_state
                
                with main_lock:
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
    
    
    def update_epsilon_greedy(self):
        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_step

    def train(self,env,num_steps=int(1e6),render=False):
        #unpack tensorboard placeholders and summary scalars
        tf_epLoss, tf_epReward, tf_epScore, tf_valReward = self.tf_placeholders[0]
        tf_sum_epLoss, tf_sum_epReward, tf_sum_epScore, tf_sum_valReward = self.tf_summary_scalars[0]

        #self.populate_memory(env,num_samples=1000)
        observation = env.reset()
        state = self.stack_frames(observation,reset=True)
        episode = 0
        episode_reward = []
        episode_score = []
        episode_loss = []
        terminal = False
        lives = 0
        
        self.sess.run([self.update_local])

        for t in range(num_steps):

            self.update_epsilon_greedy()

            if np.random.uniform() < self.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(self.Q.forward(self.sess,state))
            
            observation, reward, done, info = env.step(action)
            
            if info['ale.lives'] < lives:
                terminal = True
            else:
                lives = info['ale.lives']
                terminal = done
        
            episode_score.append(reward)
            reward = np.clip(reward,-1,1)
            episode_reward.append(reward)
            next_state = self.stack_frames(observation)
            self.batch.append((state,action,reward,next_state,terminal))
        

            if render:
                with main_lock:
                    env.render()
            
            if done or t == num_steps -1:
                episode += 1
                tot_reward = np.sum(episode_reward)
                tot_score = np.sum(episode_score)
                avg_loss = np.mean(episode_loss)

                print('Episode %s, Total score/reward %i/%i,  epsilon %f, average_loss %f, Total steps %i'
                    %(episode, tot_score, tot_reward, self.epsilon, avg_loss, self.T[0]))
                #Reset
                observation = env.reset()
                state = self.stack_frames(observation,reset=True)
                #Statistics 
                episode_reward = []
                episode_score = []
                episode_loss = []
                l, r, s = self.sess.run([tf_sum_epLoss, tf_sum_epReward, tf_sum_epScore], feed_dict={tf_epLoss:avg_loss,tf_epReward:tot_reward,tf_epScore:tot_score})
                self.train_writer.add_summary(l, self.T[0])
                self.train_writer.add_summary(r, self.T[0])
                self.train_writer.add_summary(s, self.T[0])
            
            episode_loss.append(self.update_Q(state, action, reward, next_state, terminal))
            state = next_state
            self.T += 1

            if self.validating == True and self.workerID == 1:
                avg_reward = self.validate(env,20)
                _, val_r = self.sess.run([tf_valReward, tf_sum_valReward], feed_dict={tf_valReward:avg_reward})
                self.train_writer.add_summary(val_r, t)
                self.validating[:] = False
            else:
                while self.validating == True:
                    time.sleep(2)

            if self.T % ASYNC_UPDATE == 0:
                self.updating_async[:] = True
                self.Q.update_step(self.sess)
                self.updating_async[:] = False

            if self.T % UPDATE_TARGET == 0:
                self.updating_target[:] = True
                self.sess.run([self.update_target])
                print("updated Target network")
                print(UPDATE_TARGET, " frames time taken: ", time.time()-self.start_time[0])
                self.start_time[0] = time.time()
                self.saver.save(self.sess, str(self.model_dir + self.modelname + ".ckpt") )
                self.updating_target[:] = False
            
            #if self.T % VALIDATE_FREQ == 0 and self.validating == False:
                #self.validating[:] = True


    
# Update frequencies must not be factors to avoid shared (T)ime incrementing before updates completed
ASYNC_UPDATE = 6
UPDATE_TARGET = 10000
VALIDATE_FREQ = 5e5 + 1
NUM_WORKERS = 16
#REPLAY_START_SIZE = 5000
model_dir = "models/Async_DQN/"
modelname = "DDQN_Pong"
action_size = 6


def main():
    print("HELLO")
    #pool = mp.Pool(4)
    print("HELLO2")
    a3c = AsyncDQN(NUM_WORKERS,action_size,"PongDeterministic-v4",[model_dir, modelname],1, *[84,4,32,64,64,512])
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
