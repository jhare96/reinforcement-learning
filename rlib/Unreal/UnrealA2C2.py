import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time, datetime
import threading
import scipy
from rlib.utils.utils import fold_batch, one_hot, RunningMeanStd
from collections import deque
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*

from rlib.A2C.ActorCritic import ActorCritic

# A2C-CNN version of Unsupervised Reinforcement Learning with Auxiliary Tasks (UNREAL) https://arxiv.org/abs/1611.05397
# Modifications:
#   no action-reward fed into policy
#   Use greyscaled images
#   deconvolute to pixel grid that overlaps FULL image
#   Generalised Advantage Estimation


class UnrealA2C2(object):
    def __init__(self,  policy_model, input_shape, action_size, pixel_control=True, RP=1.0, PC=1.0, VR=1.0, entropy_coeff=0.001, value_coeff=0.5, lr=1e-3, grad_clip = 0.5, policy_args ={}):
        self.RP, self.PC, self.VR = RP, PC, VR
        self.lr = lr
        self.entropy_coeff, self.value_coeff = entropy_coeff, value_coeff
        self.pixel_control = pixel_control
        self.grad_clip = grad_clip
        self.action_size = action_size

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        
        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE):
            self.policy = ActorCritic(policy_model, input_shape, action_size, entropy_coeff=entropy_coeff, value_coeff=value_coeff, 
                                        build_optimiser=False, **policy_args)
            self.replay_policy = ActorCritic(policy_model, input_shape, action_size, entropy_coeff=entropy_coeff, value_coeff=value_coeff,
                                                build_optimiser=False, **policy_args)

        if pixel_control:
            with tf.variable_scope('pixel_control', reuse=tf.AUTO_REUSE):
                self.Qaux = self._build_pixel(self.replay_policy.dense)
                
                self.Qaux_target = tf.placeholder("float", [None, 21, 21]) # temporal difference target for Q_aux
                self.Qaux_actions = tf.placeholder(tf.int32, [None])
                one_hot_actions = tf.one_hot(self.Qaux_actions, action_size)
                pixel_action = tf.reshape(one_hot_actions, shape=[-1,1,1, action_size], name='pixel_action')
                Q_aux_action = tf.reduce_sum(self.Qaux * pixel_action, axis=3)
                pixel_loss = 0.5 * tf.reduce_mean(tf.square(self.Qaux_target - Q_aux_action)) # l2 loss for Q_aux over all pixels and batch

        
        with tf.variable_scope('value_replay'):
            replay_loss = 0.5 * tf.reduce_mean(tf.square(self.replay_policy.R - self.replay_policy.V))
        

        self.reward_state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='reward_state')
        with tf.variable_scope('ActorCritic/encoder_network', reuse=True):
            reward_enc = policy_model(self.reward_state)

        with tf.variable_scope('reward_model'):
            self.reward_target = tf.placeholder(tf.float32, shape=[None, 3], name='reward_target')
            r1 = mlp_layer(reward_enc, 128, activation=tf.nn.relu, name='reward_hidden')
            print('rl shape', r1.get_shape().as_list())
            pred_reward = mlp_layer(r1, 3, activation=None, name='pred_reward')
            print('pred reward shape', pred_reward.get_shape().as_list())
            #reward_loss = 0.5 * tf.reduce_mean(tf.square(self.reward_target - pred_reward)) #mse
            reward_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=pred_reward, onehot_labels=self.reward_target))  # cross entropy over caterogical reward 
            print('reward loss ', reward_loss)
            
        
        

        self.on_policy_loss = self.policy.loss
        self.auxiliary_loss = RP * reward_loss +  VR * replay_loss 
        if pixel_control:
            self.auxiliary_loss += PC * pixel_loss 
        self.loss = self.on_policy_loss + self.auxiliary_loss 
        
        #self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
        self.optimiser = tf.train.AdamOptimizer(lr)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))
        self.train_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)
        

    def _build_pixel(self, input):
        # ignoring cropping from paper hence deconvoluting to size 21x21 feature map (as 84x84 / 4 == 21x21)
        feat_map = mlp_layer(input, 32*8*8, activation=tf.nn.relu, name='feat_map_flat')
        feat_map = tf.reshape(feat_map, shape=[-1,8,8,32], name='feature_map')
        batch_size = tf.shape(feat_map)[0]
        deconv1 = conv_transpose_layer(feat_map, output_shape=[batch_size,10,10,32], kernel_size=[3,3], strides=[1,1], padding='VALID', activation=tf.nn.relu)
        deconv_advantage = conv2d_transpose(deconv1, output_shape=[batch_size,21,21,self.action_size],
                kernel_size=[3,3], strides=[2,2], padding='VALID', activation=None, name='deconv_adv')
        deconv_value = conv2d_transpose(deconv1, output_shape=[batch_size,21,21,1],
                kernel_size=[3,3], strides=[2,2], padding='VALID', activation=None, name='deconv_value')

        # Auxillary Q value calculated via dueling network 
        # Z. Wang, N. de Freitas, and M. Lanctot. Dueling Network Architectures for Deep ReinforcementLearning. https://arxiv.org/pdf/1511.06581.pdf
        Qaux = tf.nn.relu(deconv_value + deconv_advantage - tf.reduce_mean(deconv_advantage, axis=3, keep_dims=True))
        print('Qaux', Qaux.get_shape().as_list())
        return Qaux

    def forward(self, state):
        return self.policy.forward(state)

    def forward_all(self, state):
        feed_dict = {self.policy.state:state, self.replay_policy.state:state}
        return self.sess.run([self.policy.policy_distrib, self.policy.V, self.Qaux], feed_dict=feed_dict)
    
    def get_pixel_control(self, state):
        feed_dict = {self.replay_policy.state:state}
        return self.sess.run(self.Qaux, feed_dict=feed_dict)

    def backprop(self, states, R, actions, dones,
                    reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, replay_dones):
        feed_dict = {self.policy.state:states, self.policy.actions:actions, self.policy.R:R,
            self.reward_target:rewards, self.reward_state:reward_states,
            self.replay_policy.state:replay_states, self.replay_policy.R:replay_R}
        
        if self.pixel_control:
            feed_dict[self.Qaux_target] = Qaux_target
            feed_dict[self.Qaux_actions] =  Qaux_actions
        _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l

    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)
        self.replay_policy.set_session(sess)


class Unreal_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model,  val_envs, train_mode='nstep', log_dir='logs/UnrealA2C2', model_dir='models/UnrealA2C2', total_steps=1000000, nsteps=5,
                normalise_obs=True, validate_freq=1000000, save_freq=0, render_freq=0, num_val_episodes=50, replay_length=2000, log_scalars=True, gpu_growth=True):
        
        super().__init__(envs, model,  val_envs, train_mode=train_mode,  log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars, gpu_growth=gpu_growth)
        
        self.replay = deque([], maxlen=replay_length) #replay length per actor
        self.runner = self.Runner(self.model, self.env, self.nsteps, self.replay)

        hyper_paras = {'learning_rate':model.lr, 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':self.num_envs,
                  'total_steps':self.total_steps, 'entropy_coefficient':model.entropy_coeff, 'value_coefficient':model.value_coeff}
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
        
        self.normalise_obs = normalise_obs
        
        if self.normalise_obs:
            self.obs_running = RunningMeanStd()
            self.state_mean = np.zeros_like(self.runner.states)
            self.state_std = np.ones_like(self.runner.states)
            self.aux_reward_rolling = RunningMeanStd()
    
    def populate_memory(self):
        for t in range(2000//self.nsteps):
            states, *_ = self.runner.run()
            #self.state_mean, self.state_std = self.obs_running.update(fold_batch(states)[...,-1:])
            self.update_minmax(states)


    def update_minmax(self, obs):
        minima = obs.min()
        maxima = obs.max()
        if minima < self.state_min:
            self.state_min = minima
        if maxima > self.state_max:
            self.state_max = maxima
    
    def norm_obs(self, obs):
        return (obs - self.state_min) * (1/(self.state_max - self.state_min))
    
    def auxiliary_target(self, pixel_rewards, last_values, dones):
        T = len(pixel_rewards)
        R = np.zeros((T,*last_values.shape))
        dones = dones[:,:,np.newaxis,np.newaxis]
        R[-1] = last_values * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = pixel_rewards[i] + 0.99 * R[i+1] * (1-dones[-1])
        
        return R
    
    def pixel_rewards(self, prev_state, states):
        T = len(states) # time length 
        B = states.shape[1] #batch size
        pixel_rewards = np.zeros((T,B,21,21))
        states = states[...,-1:]
        prev_state = prev_state[...,-1:]
        if self.normalise_obs:
            states = self.norm_obs(states)
            #print('states, max', states.max(), 'min', states.min(), 'mean', states.mean())
            prev_state = self.norm_obs(prev_state)
            
        pixel_rewards[0] = np.abs(states[0] - prev_state).reshape(-1,21,4,21,4).mean(axis=(2,4))
        for i in range(1,T):
            pixel_rewards[i] = np.abs(states[i] - states[i-1]).reshape(-1,21,4,21,4).mean(axis=(2,4))
        #print('pixel reward',pixel_rewards.shape, 'max', pixel_rewards.max(), 'mean', pixel_rewards.mean())
        return pixel_rewards

    def sample_replay(self):
        sample_start = np.random.randint(1, len(self.replay) -self.nsteps -2)
        replay_sample = []
        for i in range(sample_start, sample_start+self.nsteps):
            replay_sample.append(self.replay[i])
                
        replay_states = np.stack([replay_sample[i][0] for i in range(len(replay_sample))])
        replay_actions = np.stack([replay_sample[i][1] for i in range(len(replay_sample))])
        replay_rewards = np.stack([replay_sample[i][2] for i in range(len(replay_sample))])
        replay_values = np.stack([replay_sample[i][3] for i in range(len(replay_sample))])
        replay_dones = np.stack([replay_sample[i][4] for i in range(len(replay_sample))])
        #print('replay_hiddens dones shape', replay_dones.shape)
        
        next_state = self.replay[sample_start+self.nsteps][0] # get state 
        _, replay_last_values = self.model.forward(next_state)
        replay_R = self.GAE(replay_rewards, replay_values, replay_last_values, replay_dones, gamma=0.99, lambda_=0.95) + replay_values

        if self.model.pixel_control:
            prev_states = self.replay[sample_start-1][0]
            Qaux_value = self.model.get_pixel_control(next_state)
            pixel_rewards = self.pixel_rewards(prev_states, replay_states)
            Qaux_target = self.auxiliary_target(pixel_rewards, np.max(Qaux_value, axis=-1), replay_dones)
        else:
            Qaux_target = np.zeros((len(replay_states),1,1,1)) # produce fake Qaux to save writing unecessary code
        
        return fold_batch(replay_states), fold_batch(replay_actions), fold_batch(replay_R), fold_batch(Qaux_target), fold_batch(replay_dones)
    
    def sample_reward(self):
        # worker = np.random.randint(0,self.num_envs) # randomly sample from one of n workers
        replay_rewards = np.array([self.replay[i][2] for i in range(len(self.replay))])
        worker = np.argmax(np.sum(replay_rewards, axis=0)) # sample experience from best worker
        nonzero_idxs = np.where(np.abs(replay_rewards) > 0)[0] # idxs where |reward| > 0 
        zero_idxs = np.where(replay_rewards == 0)[0] # idxs where reward == 0 
        
        
        if len(nonzero_idxs) ==0 or len(zero_idxs) == 0: # if nonzero or zero idxs do not exist i.e. all rewards same sign 
            idx = np.random.randint(len(replay_rewards))
        elif np.random.uniform() > 0.5: # sample from zero and nonzero rewards equally
            #print('nonzero')
            idx = np.random.choice(nonzero_idxs)
        else:
            idx = np.random.choice(zero_idxs)
        
        
        reward_states = self.replay[idx][0][worker]
        #reward_states = np.stack([replay_states[i] for i in range(idx-3,idx)])
        #sign = int(np.sign(self.replay[idx][2][worker]))
        sign = int(np.sign(replay_rewards[idx,worker]))
        reward = np.zeros((1,3))
        reward[0,sign] = 1 # catergorical [zero, positive, negative]
    
        return reward_states[np.newaxis], reward
    
    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        self.state_min = 0
        self.state_max = 0
        self.populate_memory()
        # main loop
        start = time.time()
        for t in range(1,num_updates+1):
            states, actions, rewards, values, dones, infos, last_values = self.runner.run()
            
            # R = self.nstep_return(rewards, last_values, dones, clip=False)
            R = self.GAE(rewards, values, last_values, dones, gamma=0.99, lambda_=0.95) + values
            
            # stack all states, actions and Rs across all workers into a single batch
            states, actions, rewards, R = fold_batch(states), fold_batch(actions), fold_batch(rewards), fold_batch(R)
            
            #self.state_mean, self.state_std = self.obs_running.update(states[...,-1:]) # update state normalisation statistics
            self.update_minmax(states)

            reward_states, sample_rewards = self.sample_reward()
            replay_states, replay_actions, replay_R, Qaux_target, replay_dones = self.sample_replay()
            
            l = self.model.backprop(states, R, actions,  dones,
                reward_states, sample_rewards, Qaux_target, replay_actions, replay_states, replay_R, replay_dones)
            
            if self.render_freq > 0 and t % ((self.validate_freq // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + '/' + str(s) + ".ckpt") )
                print('saved model')


    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps, replay):
            super().__init__(model, env, num_steps)
            self.replay = replay
            self.action_size = self.model.action_size

        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values = self.model.forward(self.states)
                #Qaux = self.model.get_pixel_control(self.states, self.prev_hidden, self.prev_actions_rewards[np.newaxis])
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, rewards, dones, infos = self.env.step(actions)

                rollout.append((self.states, actions, rewards, values, dones, infos))
                self.replay.append((self.states, actions, rewards, values, dones, infos)) # add to replay memory
                self.states = next_states
            
            states, actions, rewards, values, dones, infos = zip(*rollout)
            states, actions, rewards, values, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(values), np.stack(dones)
            _, last_values = self.model.forward(next_states)
            return states, actions, rewards, values, dones, infos, last_values

    def get_action(self, state):
        policy, value = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        #action = np.argmax(policy)
        return action

    def validate(self,env,num_ep,max_steps,render=False):
        episode_scores = []
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            for t in range(max_steps):
                policy, value = self.model.forward(state[np.newaxis])
                #print('policy', policy, 'value', value)
                action = np.random.choice(policy.shape[1], p=policy[0])
                next_state, reward, done, info = env.step(action)
                state = next_state
                episode_score.append(reward)
                
                if render:
                    with self.lock:
                        env.render()

                if done or t == max_steps -1:
                    tot_reward = np.sum(episode_score)
                    with self.lock:
                        self.validate_rewards.append(tot_reward)
                    
                    break
        if render:
            with self.lock:
                env.close()  

def main(env_id, Atari=True):
    num_envs = 32
    nsteps = 20

    env = gym.make(env_id)
    
    
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(16)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/UnrealA2C2/' + env_id + '/' + current_time
    model_dir = "models/UnrealA2C2/" + env_id + '/' + current_time



    model = UnrealA2C2(nature_cnn,
                      input_shape = input_size,
                      action_size = action_size,
                      PC=1,
                      entropy_coeff=0.001,
                      lr=1e-3,
                      pixel_control=True,
                      grad_clip=0.5,
                      policy_args={})

    

    auxiliary = Unreal_Trainer(envs = envs,
                                model = model,
                                model_dir = model_dir,
                                log_dir = train_log_dir,
                                val_envs = val_envs,
                                train_mode = 'nstep',
                                total_steps = 50e6,
                                nsteps = nsteps,
                                normalise_obs=True,
                                validate_freq = 1e6,
                                save_freq = 0,
                                render_freq = 0,
                                num_val_episodes = 50,
                                log_scalars=True,
                                gpu_growth=True)

    
    
    

    auxiliary.train()

    del auxiliary

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = ['SpaceInvadersDeterministic-v4', 'MontezumaRevengeDeterministic-v4' 'FreewayDeterministic-v4', 'PongDeterministic-v4' ]
    #env_id_list = ['MountainCar-v0','CartPole-v1', 'Acrobot-v1']
    for env_id in env_id_list:
        main(env_id)
    