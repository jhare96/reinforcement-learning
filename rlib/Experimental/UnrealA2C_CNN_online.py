import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
import scipy
from rlib.utils.utils import fold_batch, one_hot, stack_many, rolling_stats, normalise
from collections import deque
from rlib.A2C.A2C import ActorCritic
from rlib.A2C.ActorCritic import ActorCritic_LSTM
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*


def concat_action_reward(actions, rewards, num_classes):
    concat = one_hot(actions, num_classes)
    concat[:,-1] = rewards   
    return concat  

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

def AtariEnv_(env, k=4, episodic=True, reset=True, clip_reward=True, Noop=True, time_limit=None):
    # Wrapper function for Determinsitic Atari env 
    # assert 'Deterministic' in env.spec.id
    if reset:
        env = FireResetEnv(env)
    if Noop:
        max_op = 7
        env = NoopResetEnv(env,max_op)
    
    if clip_reward:
        env = ClipRewardEnv(env)

    if episodic:
        env = EpisodicLifeEnv(env)

    env = AtariRescaleColour(env)
    
    if k > 1:
        env = StackEnv(env,k)
    if time_limit is not None:
        env = TimeLimitEnv(env, time_limit)
    return env



class ActorCritic(object):
    def __init__(self, model, input_shape, action_size, entropy_coeff=0.001, value_coeff=0.5, lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip = 0.5, optimisation=False, **model_args):
        self.lr, self.lr_final = lr, lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.sess = None

        with tf.variable_scope('encoder_network'):
            self.state = tf.placeholder(tf.float32, shape=[None, *input_shape])
            print('state shape', self.state.get_shape().as_list())
            self.dense = model(self.state, **model_args)
        
        with tf.variable_scope('critic'):
            self.V = tf.reshape(mlp_layer(self.dense, 1, name='state_value', activation=None), shape=[-1])
        
        with tf.variable_scope("actor"):
            self.policy_distrib = mlp_layer(self.dense, action_size, activation=tf.nn.softmax, name='policy_distribution') 
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.R = tf.placeholder(dtype=tf.float32, shape=[None])
            value_loss = 0.5 * tf.reduce_mean(tf.square(self.R - self.V))

            self.Advantage = tf.placeholder(dtype=tf.float32, shape=[None], name='Adv')
            log_policy = tf.math.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
            log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
            policy_loss =  tf.reduce_mean(-log_policy_actions * self.Advantage)

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
    
        
        
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        self.loss =  policy_loss + value_coeff * value_loss - entropy_coeff * entropy
        
        if optimisation:
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)

    def forward(self, state):
        return self.sess.run([self.policy_distrib, self.V], feed_dict = {self.state:state})

    def get_policy(self, state):
        return self.sess.run(self.policy_distrib, feed_dict = {self.state: state})
    
    def get_value(self, state):
        return self.sess.run(self.V, feed_dict = {self.state: state})

    def backprop(self, state, R, Adv, a):
        feed_dict = {self.state:state, self.R:R, self.Advantage:Adv, self.actions:a}
        *_,l = self.sess.run([self.train_op, self.loss], feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess

class UnrealA2C(object):
    def __init__(self,  policy_model, input_shape, action_size, cell_size, num_envs, RP=1.0, PC=1.0, VR=1.0, entropy_coeff=0.001, value_coeff=0.5, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}):
        self.RP, self.PC, self.VR = RP, PC, VR
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.entropy_coeff, self.value_coeff = entropy_coeff, value_coeff
        self.grad_clip = grad_clip
        self.action_size = action_size
        print('action_size', action_size)

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        
        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE):
            self.policy = ActorCritic(policy_model, input_shape, action_size, entropy_coeff=entropy_coeff, value_coeff=value_coeff, lr=lr, lr_final=lr, decay_steps=decay_steps, grad_clip=grad_clip, **policy_args)

        with tf.variable_scope('pixel_control', reuse=tf.AUTO_REUSE):
            self.Qaux = self._build_pixel(self.policy.dense)
            
            self.Qaux_target = tf.placeholder("float", [None, 21, 21]) # temporal difference target for Q_aux
            self.Qaux_actions = tf.placeholder(tf.int32, [None])
            one_hot_actions = tf.one_hot(self.Qaux_actions, action_size)
            pixel_action = tf.reshape(one_hot_actions, shape=[-1,1,1, action_size], name='pixel_action')
            Q_aux_action = tf.reduce_sum(self.Qaux * pixel_action, axis=3)
            pixel_loss = 0.5 * tf.reduce_mean(tf.square(self.Qaux_target - Q_aux_action)) # l2 loss for Q_aux over all pixels and batch
        

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
        self.auxiliary_loss = PC * pixel_loss + RP * reward_loss 
        self.loss = self.on_policy_loss + self.auxiliary_loss 
        
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)

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
        mask = np.zeros((1, state.shape[0]))
        feed_dict = {self.policy.state:state, self.policy.state:state}
        return self.sess.run([self.policy.policy_distrib, self.policy.V, self.Qaux], feed_dict=feed_dict)
    
    def get_pixel_control(self, state):
        mask = np.zeros((1, state.shape[0]))
        feed_dict = {self.policy.state:state}
        return self.sess.run(self.Qaux, feed_dict=feed_dict)

    def backprop(self, states, R, Adv, actions, dones, reward_states, rewards, Qaux_target):
        feed_dict = {self.policy.state:states, self.policy.actions:actions,
            self.policy.R:R, self.policy.Advantage:Adv,
            self.Qaux_target:Qaux_target, self.Qaux_actions:actions,
            self.reward_target:rewards, self.reward_state:reward_states}
        _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l

    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)


class Unreal_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars)
        
        self.replay = deque([], maxlen=2000)
        self.runner = self.Runner(self.model, self.env, self.nsteps, self.replay)

        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':self.num_envs,
                  'total_steps':self.total_steps, 'entropy_coefficient':model.entropy_coeff, 'value_coefficient':0.5}
        
        if log_scalars:
            filename = file_loc[1] + self.current_time + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
        
        self.obs_running = rolling_stats()
        self.state_mean = np.zeros_like(self.runner.states)
        self.state_std = np.ones_like(self.runner.states)
        self.aux_reward_rolling = rolling_stats()
    
    def populate_memory(self):
        for t in range(40//self.nsteps):
            states, *_ = self.runner.run()
            self.state_mean, self.state_std = self.obs_running.update(states.mean(axis=(0,1))[:,:,-1:])
    
    def norm_img(self, obs, clip=True):
        norm = (obs-self.state_mean)/self.state_std
        if clip:
            norm = np.clip(norm, -5, 5)
        #print('norm, max', norm.max(), 'min', norm.min(), 'mean', norm.mean())
        return norm
    
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
        norm_states = self.norm_img(states[...,-1:])
        pixel_rewards[0] = np.abs(norm_states[0] - self.norm_img(prev_state)[...,-1:]).reshape(-1,21,4,21,4).mean(axis=(2,4))
        for i in range(1,T):
            pixel_rewards[i] = np.abs(norm_states[i] - norm_states[i-1]).reshape(-1,21,4,21,4).mean(axis=(2,4))
        #print('pixel reward',pixel_rewards.shape, 'max', pixel_rewards.max(), 'mean', pixel_rewards.mean())
        return pixel_rewards

    
    def sample_reward(self):
       # worker = np.random.randint(0,self.num_envs) # randomly sample from one of n workers
        replay_rewards = np.array([self.replay[i][2] for i in range(len(self.replay))])[3:]
        worker = np.argmax(np.sum(replay_rewards, axis=0)) # sample experience from best worker
        #replay_states = np.array([self.replay[i][0][worker] for i in range(len(self.replay))])[3:]
        #print('replay_rewards', replay_rewards.shape)
        #print('replay_states', replay_states.shape)
        nonzero_idxs = np.where(np.abs(replay_rewards) > 0)[0] # idxs where |reward| > 0 
        zero_idxs = np.where(replay_rewards == 0)[0] # idxs where reward == 0 
        
        
        if len(nonzero_idxs) ==0 or len(zero_idxs) == 0: # if nonzero or zero idxs do not exist i.e. all rewards same sign 
            idx = np.random.randint(len(replay_rewards))
        elif np.random.uniform() > 0.5: # sample from zero and nonzero rewards equally
            #print('nonzero')
            idx = np.random.choice(nonzero_idxs)
        else:
            idx = np.random.choice(zero_idxs)
        
        
        reward_states =self.replay[idx][0][worker]
        #reward_states = np.stack([replay_states[i] for i in range(idx-3,idx)])
        #sign = int(np.sign(self.replay[idx][2][worker]))
        sign = int(np.sign(replay_rewards[idx,worker]))
        reward = np.zeros((1,3))
        reward[0,sign] = 1 # catergorical [zero, positive, negative]
    
        return reward_states[np.newaxis], reward
    
    def print_stats(self, string, x):
        print(string, 'mean', x.mean(), 'min', x.min(), 'max', x.max())

    def _train_nstep(self):
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        #self.validate(self.val_envs[0], 1, 1000)
        self.populate_memory()
        # main loop
        for t in range(1,num_updates+1):
            states, actions, rewards, values, dones, last_values, prev_state, Qaux = self.runner.run()
            self.state_mean, self.state_std = self.obs_running.update(fold_batch(states).mean(axis=0)[:,:,-1:])
            pixel_rewards = self.pixel_rewards(prev_state, states)
            pix_rew_mean, pix_rew_std = self.aux_reward_rolling.update(self.auxiliary_target(pixel_rewards, np.max(Qaux, axis=-1), dones).mean())
            Qaux_target = self.auxiliary_target(pixel_rewards/pix_rew_std, np.max(Qaux, axis=-1), dones)

            # R = self.nstep_return(rewards, last_values, dones, clip=False)
            Adv = self.GAE(rewards, values, last_values, dones, gamma=0.99, lambda_=0.95)
            R = Adv + values
            #self.print_stats('R', R)
            #self.print_stats('Adv', Adv)
            # stack all states, actions and Rs across all workers into a single batch
            states, actions, rewards, R, Adv, Qaux_target = fold_batch(states), fold_batch(actions), fold_batch(rewards), fold_batch(R), fold_batch(Adv), fold_batch(Qaux_target)
            
            reward_states, sample_rewards = self.sample_reward()
            #replay_states, replay_actions, replay_R, Qaux_target, replay_dones = self.sample_replay()
            
            l = self.model.backprop(states, R, Adv, actions,  dones,
                reward_states, sample_rewards, Qaux_target)
            
            if self.render_freq > 0 and t % (self.validate_freq * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % self.validate_freq == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % self.save_freq == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + self.current_time + '/' + str(s) + ".ckpt") )
                print('saved model')


    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps, replay):
            super().__init__(model, env, num_steps)
            self.replay = replay
            self.action_size = self.model.action_size
            self.first_state = self.states.copy()

        def run(self,):
            rollout = []
            first_state = self.first_state
            for t in range(self.num_steps):
                policies, values = self.model.forward(self.states)
                #Qaux = self.model.get_pixel_control(self.states, self.prev_hidden, self.prev_actions_rewards[np.newaxis])
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, rewards, dones, infos = self.env.step(actions)

                rollout.append((self.states, actions, rewards, values, dones))
                self.replay.append((self.states, actions, rewards, dones)) # add to replay memory
                self.first_state = self.states.copy()
                self.states = next_states
            
            states, actions, rewards, values, dones = stack_many(zip(*rollout))
            _, last_values = self.model.forward(next_states)
            Qaux = self.model.get_pixel_control(next_states)
            return states, actions, rewards, values, dones, last_values, first_state, Qaux

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


    config = tf.ConfigProto() #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.Session(config=config)

    print('gpu aviabliable', tf.test.is_gpu_available())

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
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(1)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=4, reset=reset, episodic=True, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/Unreal/' + env_id + '/'
    model_dir = "models/Unreal/" + env_id + '/'



    model = UnrealA2C(nature_cnn,
                      input_shape = input_size,
                      action_size = action_size,
                      cell_size = 256,
                      num_envs = num_envs,
                      PC=1,
                      entropy_coeff=0.001,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={})

    

    auxiliary = Unreal_Trainer(envs = envs,
                                model = model,
                                file_loc = [model_dir, train_log_dir],
                                val_envs = val_envs,
                                train_mode = 'nstep',
                                total_steps = 50e6,
                                nsteps = nsteps,
                                validate_freq = 1e5,
                                save_freq = 0,
                                render_freq = 0,
                                num_val_episodes = 25,
                                log_scalars = False)

    
    
    

    auxiliary.train()

    del auxiliary

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = ['FreewayDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'MontezumaRevengeDeterministic-v4',  'PongDeterministic-v4' ]
    #env_id_list = ['MountainCar-v0','CartPole-v1']
    for env_id in env_id_list:
        main(env_id)
    