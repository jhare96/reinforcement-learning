import tensorflow as tf 
import numpy as np 
import gym
from Qvalue import mlp_layer, conv_layer
from SyncMultiEnvTrainer import SyncMultiEnvTrainer
from VecEnv import AtariEnv, BatchEnv, Env, StackEnv, FireResetEnv
from Qvalue import Qvalue
from collections import deque
import time

class DQN(object):
    def __init__(self, input_size, h1_size ,h2_size, h3_size, h4_size, action_size, name, learning_rate=0.00025, grad_clip = 0.5, decay_steps=6e5, learning_rate_final=0.0001):
        self.name = name 
        self.action_size = action_size
        self.h1_size, self.h2_size, self.h3_size, self.h4_size = h1_size, h2_size, h3_size, h4_size

        with tf.variable_scope(self.name):
            self.x = tf.placeholder("float", shape=[None, *input_size], name="input")
            x = self.x/255
            h1 = conv_layer(x,  [8,8], output_channels=h1_size, strides=[4,4], padding="VALID", dtype=tf.float32, name='conv_1')
            h2 = conv_layer(h1, [4,4], output_channels=h2_size, strides=[2,2], padding="VALID", dtype=tf.float32, name='conv_2')
            h3 = conv_layer(h2, [3,3], output_channels=h3_size, strides=[1,1], padding="VALID", dtype=tf.float32, name='conv_3')
            fc = tf.contrib.layers.flatten(h3)
            h4 = mlp_layer(fc,h4_size)
            with tf.variable_scope("State_Action"):
                self.Qsa = mlp_layer(h4,action_size,activation=None)
                self.weighted_TD = tf.placeholder("float", shape=[None])
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                actions_onehot = tf.one_hot(self.actions, action_size)
                self.Qvalue = tf.reduce_sum(tf.multiply(self.Qsa, actions_onehot), axis = 1)
                
                self.loss = tf.reduce_sum(tf.multiply(self.weighted_TD, self.Qvalue))
            
            
            global_step = tf.Variable(0, trainable=False)
            tf.train.polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=learning_rate_final, power=1.0, cycle=False, name=None)
            optimiser = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, epsilon=1e-5)

            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)

    
    def forward(self,sess,state):
        return sess.run(self.Qsa, feed_dict = {self.x: state})

    def backprop(self,sess,x,y,a):
        _,l = sess.run([self.train_op,self.loss], feed_dict = {self.x : x, self.y : y, self.actions:a})
        return l

class SyncDDQN_ER(SyncMultiEnvTrainer):
    def __init__(self, env_constructor, env_id, num_envs, model, target_model, file_loc, val_env, action_size,
                     train_mode='nstep', total_steps=1000000, nsteps=5, replay_length = 1e6,
                     validate_freq=None, save_freq=0, render_freq=0, update_target_freq=10000, blocking=False,
                     epsilon_start=1, epsilon_final=0.01, epsilon_steps = 1e6, test_epsilon=0.01, **env_args):
        
        super().__init__(env_constructor, env_id, num_envs, model, file_loc, val_env, train_mode=train_mode, total_steps=total_steps,
                         nsteps=nsteps, validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq, update_target_freq=update_target_freq,
                         blocking=blocking, **env_args)
        
        self.target_model = target_model
        self.epsilon = np.array([epsilon_start], dtype=np.float64)
        self.schedule = self.linear_schedule(self.epsilon , epsilon_final, epsilon_steps)
        self.test_epsilon = np.array([test_epsilon] , dtype=np.float64)
        self.action_size = action_size
        #self.runner = SyncDDQN.Runner(self.model, self.target_model, self.epsilon, schedule, self.env, self.num_envs, self.nsteps, self.action_size, self.sess)
        
        self.update_weights = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='QTarget'), tf.trainable_variables('Q'))]
        
        self.states = self.env.reset()
        self.replay_length = replay_length
        self.replay = deque([], maxlen=int(replay_length))
    
    def get_action(self, state):
        if np.random.uniform() < self.test_epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.model.forward(self.sess, state))
        return action

    def update_target(self):
        self.sess.run(self.update_weights)
    
    def one_hot(self,x,n_labels):
        return np.eye(n_labels)[x]

    def _train_nstep(self):
        '''
            Episodic training loop for synchronous training over multiple environments

            num_steps - number of Total training steps across all environemnts
            nsteps - number of steps performed in environment before updating ActorCritic
        '''
        tf_epLoss,tf_epScore,tf_epReward,tf_valReward = self.tf_placeholders
        tf_sum_epLoss,tf_sum_epScore,tf_sum_epReward,tf_sum_valReward = self.tf_summary_scalars
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        
        # main loop
        for update in range(1,num_updates+1):
            batch = []
            for t in range(self.nsteps):
                actions = np.argmax(self.model.forward(self.sess,self.states), axis=1)
                random = np.random.uniform(size=(self.num_envs))
                random_actions = np.random.randint(self.action_size, size=(self.num_envs))
                actions = np.where(random < self.epsilon, random_actions, actions)
                next_states, rewards, dones, infos = self.env.step(actions)
                batch.append((self.states, actions, rewards, dones, infos))
                self.states = next_states
            
            
            self.schedule.step()
            action_values = self.target_model.forward(self.sess, self.states)
            actions_one_hot = self.one_hot(actions,self.action_size)
            values = np.sum(action_values * actions_one_hot, axis=1)

            states, actions, rewards, dones, infos, = zip(*batch)
            states, actions, rewards, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(dones)
            
            T = len(rewards)
            
            # Calculate R for advantage A = R - V 
            R = np.zeros((T,self.num_envs))
            R[-1] = values * (1-dones[-1])
            
            for i in reversed(range(T-1)):
                # restart score if done as wrapped env continues after end of episode
                R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[i])  
                
            
            
            # add each experience in multi-env rollout to replay
            states, actions, R, = self.fold_batch(states), self.fold_batch(actions), self.fold_batch(R)
            for j in range(states.shape[0]):
                self.replay.append([states[j],actions[j],R[j]])
                self.priority.append([max(self.priority)])
            
            # sample from replay buffer 
            #idxs = np.random.choice(np.arange(len(self.replay)), size=32, replace=False)
            sample_states = []
            w_TD = []
            sample_actions = []
            alpha = 0.6
            beta = 0.4
            for j in range(32):
                prob_dense = np.array(self.priority)
                prob = np.power(prob_dense, alpha) / np.sum(np.power(prob_dense, alpha))
                idx = np.random.choice(np.arange(len(self.replay)), size=1, replace=False, p=prob)
                self.weight[idx] = np.power(self.replay_length * prob[idx], -beta) / np.max(self.weight)
                state, action, reward = self.replay[idx]
                sample_states.append(state)
                sample_actions.append(action)
                TD_error = reward - self.model.forward(self.sess, state)
                self.priority[idx] = np.abs(TD_error)
                w_TD.append(self.weight[idx] * TD_error)
            
            l = self.backprop(self.sess, sample_states, w_TD, sample_actions)

            #sample = [self.replay[i] for i in idxs ]
            #states = np.array([sample[i][0]for i in range(len(sample))])
            #actions = np.array([sample[i][1]for i in range(len(sample))])
            #rewards_n = np.array([sample[i][2]for i in range(len(sample))])
            


            l = self.model.backprop(self.sess,states,rewards_n,actions)

            if self.render_freq > 0 and update % (self.validate_freq * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if update % self.validate_freq == 0:
                tot_steps = update * self.num_envs * self.nsteps
                time_taken = time.time() - start
                frames_per_update = (self.validate_freq * self.num_envs * self.nsteps)
                fps = frames_per_update /time_taken 
                score = self.validate(5,5000,render=render)
                print("update %i, validation score %f, total steps %i, loss %f, time taken for %i frames:%f, fps %f" %(update,score,tot_steps,l,frames_per_update,time_taken,fps))
                sumscore, sumloss = self.sess.run([tf_sum_epScore, tf_sum_epLoss], feed_dict = {tf_epScore:score, tf_epLoss:l})
                self.train_writer.add_summary(sumloss, tot_steps)
                self.train_writer.add_summary(sumscore, tot_steps)
                start = time.time()
            
            if self.save_freq > 0 and update % self.save_freq == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + self.model_name + '_' + str(s) + ".ckpt") )
            
            if self.target_freq > 0 and update % self.target_freq == 0:
                self.update_target()
        
        self.env.close()
        score = self.validate(50,5000,True)
        print(self.env_id + ', Final Score ', score)
        tot_steps = (num_updates+1) * self.num_envs * self.nsteps
        sumscore = self.sess.run(tf_sum_epScore, feed_dict = {tf_epScore:score})
        self.train_writer.add_summary(sumscore, tot_steps)

    
    class Runner(object):
        def __init__(self, Q, TargetQ, epsilon, epsilon_schedule, env, num_envs, num_steps, action_size, sess):
            self.Q = Q
            self.TargetQ = TargetQ
            self.epsilon = epsilon
            self.schedule = epsilon_schedule
            self.env = env
            self.num_envs = num_envs
            self.num_steps = num_steps
            self.action_size = action_size
            self.sess = sess
            
            self.states = self.env.reset()
            
        def one_hot(self,x,n_labels):
            return np.eye(n_labels)[x]
        
    
    class linear_schedule(object):
        def __init__(self, epsilon, epsilon_final, num_steps=1000000):
            self._counter = 0
            self._epsilon = epsilon
            self._step = (epsilon_final - epsilon) / num_steps
            self._num_steps = num_steps
        
        def step(self,):
            if self._counter < self._num_steps :
                self._epsilon += self._step
            self._counter += 1
        
        def get_epsilon(self,):
            return self._epsilon




def main(env_id):
    num_workers = 32
    model_dir = "models/"
    action_size = 4
    modelname = "Rainbow_" + env_id
    nsteps = 5
    
    dqn_cnn_args = {'input_size':[84,84,4], 'action_size':action_size,  'learning_rate':1e-3, 'grad_clip':0.5, 'decay_steps':50e6/(num_workers*nsteps),
                    'h1_size':32, 'h2_size':64, 'h3_size':64, 'h4_size':512}
    
    dqn_mlp_args = {'input_size':4, 'h1_size':32, 'h2_size':32, 'action_size':2}

    
    Q = DQN(**dqn_cnn_args, name='Q')
    TargetQ = DQN(**dqn_cnn_args, name='QTarget')  
    
    val_env = StackEnv(FireResetEnv(gym.make(env_id))) #validation env (no reset at life loss)
    
    

    DDQN = SyncDDQN_ER(env_constructor = AtariEnv,
                    env_id = env_id,
                    num_envs = num_workers,
                    model = Q,
                    target_model = TargetQ,
                    file_loc = [model_dir, modelname],
                    val_env = val_env,
                    action_size = action_size,
                    train_mode ='nstep',
                    total_steps = int(50e6),
                    nsteps = nsteps,
                    validate_freq = 16000//num_workers,
                    save_freq = 10000,
                    render_freq = 0,
                    update_target_freq = 10000//(num_workers*nsteps),
                    blocking = False,
                    epsilon_start = 1,
                    epsilon_final = 0.01,
                    epsilon_steps = 1e6,
                    test_epsilon = 0.01,
                    k = 4)
    

    DDQN.train()

    del DDQN

    tf.reset_default_graph()

if __name__ == "__main__":
    env_id_list = ['SpaceInvadersDeterministic-v4', 'PongDeterministic-v4',  'SeaquestDeterministic-v4', 'BreakoutDeterministic-v4']
    for env_id in env_id_list:
        main(env_id)