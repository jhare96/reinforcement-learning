import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time, datetime
import threading
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*
from rlib.utils.utils import fold_batch, one_hot, Welfords_algorithm, stack_many, RunningMeanStd

from rlib.RND.RND import PPO, predictor_cnn, predictor_mlp, rolling_obs, RewardForwardFilter




class RANDAL(object):
    def __init__(self, policy_model, target_model, input_shape, action_size, pixel_control=True, value_coeff=1.0, intr_coeff=0.5, extr_coeff=1.0, lr=1e-4, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, RND_args={}):
        self.intr_coeff, self.extr_coeff =  intr_coeff, extr_coeff
        self.lr, self.decay_steps = lr, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.sess = None
        self.pixel_control = pixel_control
        #self.pred_prob = 1

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        

        with tf.variable_scope('Policy', reuse=tf.AUTO_REUSE):
            self.policy = PPO(policy_model, input_shape, action_size, value_coeff=value_coeff, intr_coeff=intr_coeff, extr_coeff=extr_coeff, lr=lr, **policy_args)
            self.replay_policy = PPO(policy_model, input_shape, action_size, value_coeff=value_coeff, intr_coeff=intr_coeff, extr_coeff=extr_coeff, lr=lr, **policy_args)
        
        if len(input_shape) == 3: # if obs is img, only use final frame
            next_state_shape = input_shape[:-1] + (1,)
        else: 
            next_state_shape = input_shape
        self.next_state = tf.placeholder(tf.float32, shape=[None, *next_state_shape], name='next_state')
        self.state_mean = tf.placeholder(tf.float32, shape=[*next_state_shape], name="mean")
        self.state_std = tf.placeholder(tf.float32, shape=[*next_state_shape], name="std")
        norm_next_state = tf.clip_by_value((self.next_state - self.state_mean) / self.state_std, -5, 5)

        with tf.variable_scope('target_model'):
            target_state = target_model(norm_next_state, trainable=False)
        
        with tf.variable_scope('predictor_model'):
            pred_next_state = target_model(norm_next_state, trainable=True)
            self.intr_reward = tf.reduce_mean(tf.square(pred_next_state - tf.stop_gradient(target_state)), axis=-1)
            feat_loss = tf.reduce_mean(self.intr_reward)

        with tf.variable_scope('pixel_control', reuse=tf.AUTO_REUSE):
            self.Qaux = self._build_pixel(self.replay_policy.dense)
            
            self.Qaux_target = tf.placeholder("float", [None, 21, 21]) # temporal difference target for Q_aux
            self.Qaux_actions = tf.placeholder(tf.int32, [None])
            one_hot_actions = tf.one_hot(self.Qaux_actions, action_size)
            pixel_action = tf.reshape(one_hot_actions, shape=[-1,1,1, action_size], name='pixel_action')
            Q_aux_action = tf.reduce_sum(self.Qaux * pixel_action, axis=3)
            pixel_loss = 0.5 * tf.reduce_mean(tf.square(self.Qaux_target - Q_aux_action)) # l2 loss for Q_aux over all pixels and batch
        
        with tf.variable_scope('value_replay'):
            replay_loss = 0.5 * tf.reduce_mean(tf.square(self.replay_policy.R_extr - self.replay_policy.Ve))


        self.reward_state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='reward_state')
        with tf.variable_scope('Policy/encoder_network', reuse=True):
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

        self.loss = self.policy.loss + feat_loss + reward_loss 
        if pixel_control:
            self.loss += pixel_loss

        #self.optimiser = tf.train.AdamOptimizer(lr)
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)
        
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))

        self.train_op = self.optimiser.apply_gradients(grads_vars)
    
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
    
    def get_pixel_control(self, state):
        feed_dict = {self.replay_policy.state:state}
        return self.sess.run(self.Qaux, feed_dict=feed_dict)
    
    def intrinsic_reward(self, next_state, state_mean, state_std):
        feed_dict={self.next_state:next_state, self.state_mean:state_mean, self.state_std:state_std}
        intr_reward = self.sess.run(self.intr_reward, feed_dict=feed_dict)
        return intr_reward
   
    def backprop(self, state, next_state, R_extr, R_intr, Adv, actions, old_policy, state_mean, state_std,
        replay_states, replay_actions, replay_Rextr, Qaux_target, replay_dones, reward_states, sample_rewards):
        
        feed_dict = {self.policy.state:state, self.policy.actions:actions,
                     self.next_state:next_state, self.state_mean:state_mean, self.state_std:state_std,
                     self.policy.R_extr:R_extr, self.policy.R_intr:R_intr, self.policy.Advantage:Adv,
                     self.policy.old_policy:old_policy,
                     self.reward_target:sample_rewards, self.reward_state:reward_states,
                     self.replay_policy.state:replay_states, self.replay_policy.R_extr:replay_Rextr}

        if self.pixel_control:
            feed_dict[self.Qaux_target] = Qaux_target
            feed_dict[self.Qaux_actions] =  replay_actions

        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)


class RANDAL_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, init_obs_steps=128*50, num_epochs=4, num_minibatches=4, validate_freq=1000000.0,
                 save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True, gpu_growth=True):
        
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars,
                            gpu_growth=gpu_growth)

        self.replay = deque([], maxlen=2000)

        self.runner = self.Runner(self.model, self.env, self.nsteps, self.replay)
        self.alpha = 1
        self.pred_prob = 1 / (self.num_envs / 32.0)
        self.lambda_ = 0.95
        self.init_obs_steps = init_obs_steps
        self.state_min, self.state_max = 0, 0 
        self.num_epochs, self.num_minibatches = num_epochs, num_minibatches
        self.normalise_obs = True
        hyper_paras = {'learning_rate':model.lr,
         'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
          'entropy_coefficient':0.001, 'value_coefficient':0.5, 'intr_coeff':model.intr_coeff,
        'extr_coeff':model.extr_coeff, 'init_obs_steps':init_obs_steps}
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def validate(self,env,num_ep,max_steps,render=False):
        episode_scores = []
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            for t in range(max_steps):
                policy, value_extr, value_intr = self.model.forward(state[np.newaxis])
                #print('policy', policy, 'value_extr', value_extr)
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
        replay_extr_values = np.stack([replay_sample[i][3] for i in range(len(replay_sample))]) 
        replay_dones = np.stack([replay_sample[i][4] for i in range(len(replay_sample))])
        #print('replay_hiddens dones shape', replay_dones.shape)
        
        next_state = self.replay[sample_start+self.nsteps][0] # get state 
        _, replay_extr_last_values, replay_intr_last_values = self.model.forward(next_state)
        replay_R = self.GAE(replay_rewards, replay_extr_values, replay_extr_last_values, replay_dones, gamma=0.99, lambda_=0.95) + replay_extr_values

        if self.model.pixel_control:
            prev_states = self.replay[sample_start-1][0]
            Qaux_value = self.model.get_pixel_control(next_state)
            pixel_rewards = self.pixel_rewards(prev_states, replay_states)
            
            Qaux_target = self.auxiliary_target(pixel_rewards, np.max(Qaux_value, axis=-1), replay_dones)
        else:
            Qaux_target = np.zeros((len(replay_states),2,3,4)) # produce fake Qaux to save writing unecessary code
        
        return replay_states, replay_actions, replay_R, Qaux_target, replay_dones
    
    def sample_reward(self):
        replay_rewards = np.array([self.replay[i][2] for i in range(len(self.replay))])
        worker = np.argmax(np.sum(replay_rewards, axis=0)) # sample experience from best worker
        nonzero_idxs = np.where(np.abs(replay_rewards) > 0)[0] # idxs where |reward| > 0 
        zero_idxs = np.where(replay_rewards == 0)[0] # idxs where reward == 0 
        
        
        if len(nonzero_idxs) ==0 or len(zero_idxs) == 0: # if nonzero or zero idxs do not exist i.e. all rewards same sign 
            idx = np.random.randint(len(replay_rewards))
        elif np.random.uniform() > 0.5: # sample from zero and nonzero rewards equally
            idx = np.random.choice(nonzero_idxs)
        else:
            idx = np.random.choice(zero_idxs)
        
        
        reward_states = self.replay[idx][0][worker]
        sign = int(np.sign(self.replay[idx][2][worker]))
        reward = np.zeros((1,3))
        reward[0,sign] = 1 # catergorical [zero, positive, negative]
    
        return reward_states[np.newaxis], reward
    
    def init_state_obs(self, num_steps):
        states = 0
        for i in range(1, num_steps+1):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            states += next_states
        mean = states / num_steps
        self.runner.state_mean, self.runner.state_std = self.state_rolling.update(mean.mean(axis=0)[None,None])

    
    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        rolling = RunningMeanStd(shape=())
        obs = self.runner.states[0]
        obs = obs[...,-1:] if len(obs.shape) == 3 else obs
        self.state_rolling = rolling_obs(shape=obs.shape)
        self.init_state_obs(self.init_obs_steps)
        self.populate_memory()
        self.runner.states = self.env.reset()
        forward_filter = RewardForwardFilter(0.99)

        # main loop
        start = time.time()
        for t in range(self.t,num_updates+1):
            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, old_policies, dones = self.runner.run()
            policy, extr_last_values, intr_last_values = self.model.forward(next_states[-1])

            self.runner.state_mean, self.runner.state_std = self.state_rolling.update(next_states) # update state normalisation statistics 

            int_rff = np.array([forward_filter.update(intr_rewards[i]) for i in range(len(intr_rewards))])
            R_intr_mean, R_intr_std = rolling.update(int_rff.ravel()) 
            intr_rewards /= R_intr_std # normalise intr reward

            reward_states, sample_rewards = self.sample_reward()
            replay_states, replay_actions, replay_Rextr, Qaux_target, replay_dones = self.sample_replay()

            Adv_extr = self.GAE(extr_rewards, values_extr, extr_last_values, dones, gamma=0.999, lambda_=self.lambda_)
            Adv_intr = self.GAE(intr_rewards, values_intr, intr_last_values, np.zeros_like(dones), gamma=0.99, lambda_=self.lambda_) # non episodic intr reward signal 
            R_extr = Adv_extr + values_extr
            R_intr = Adv_intr + values_intr
            total_Adv = self.model.extr_coeff * Adv_extr + self.model.intr_coeff * Adv_intr

            # perform minibatch gradient descent for K epochs 
            l = 0
            idxs = np.arange(len(states))
            for epoch in range(self.num_epochs):
                mini_batch_size = self.nsteps//self.num_minibatches
                np.random.shuffle(idxs)
                for batch in range(0,len(states), mini_batch_size):
                    batch_idxs = idxs[batch:batch + mini_batch_size]
                    # stack all states, next_states, actions and Rs across all workers into a single batch
                    mb_states, mb_nextstates, mb_actions, mb_Rextr, mb_Rintr, mb_Adv, mb_old_policies = fold_batch(states[batch_idxs]), fold_batch(next_states[batch_idxs]), \
                                                    fold_batch(actions[batch_idxs]), fold_batch(R_extr[batch_idxs]), fold_batch(R_intr[batch_idxs]), \
                                                    fold_batch(total_Adv[batch_idxs]), fold_batch(old_policies[batch_idxs])
                    
                    mb_replay_states, mb_replay_actions, mb_replay_Rextr, mb_Qaux_target, mb_replay_dones = fold_batch(replay_states[batch_idxs]), fold_batch(replay_actions[batch_idxs]), \
                                                                                           fold_batch(replay_Rextr[batch_idxs]), fold_batch(Qaux_target[batch_idxs]), fold_batch(replay_dones[batch_idxs])
                
                    mb_nextstates = mb_nextstates[np.where(np.random.uniform(size=(mini_batch_size)) < self.pred_prob)]
                    mb_nextstates = mb_nextstates[...,-1:] if len(mb_nextstates.shape) == 4 else mb_nextstates
                    
                    mean, std = self.runner.state_mean, self.runner.state_std
                    l += self.model.backprop(mb_states, mb_nextstates, mb_Rextr, mb_Rintr, mb_Adv, mb_actions, mb_old_policies, mean, std,
                                            mb_replay_states, mb_replay_actions, mb_replay_Rextr, mb_Qaux_target, mb_replay_dones, reward_states, sample_rewards)
            
            l /= (self.num_epochs * self.num_minibatches)
        
            if self.render_freq > 0 and t % (self.validate_freq // batch_size * self.render_freq) == 0:
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
            
    
    def get_action(self, state):
        policy, value = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        return action

    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps, replay):
            super().__init__(model, env, num_steps)
            self.replay = replay
            self.state_mean = None
            self.state_std = None
        
        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values_extr, values_intr = self.model.forward(self.states)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)
    
                next_states__ = next_states[...,-1:] if len(next_states.shape) == 4 else next_states
                intr_rewards = self.model.intrinsic_reward(next_states__, self.state_mean, self.state_std)
                #print('intr rewards', intr_rewards)
                rollout.append((self.states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones))
                self.replay.append((self.states, actions, extr_rewards, values_extr, dones)) # add to replay memory
                self.states = next_states

            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones = stack_many(zip(*rollout))
            return states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones
    
    


def main(env_id, Atari=True):
    tf.reset_default_graph()

    num_envs = 32
    nsteps = 128

    env = gym.make(env_id)
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(1)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=84, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    #time.sleep(np.random.uniform(1,30)) # stop processes sharing same log dir
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/RANDAL/' + env_id + '/' + current_time
    model_dir = "models/RANDAL/" + env_id + '/' + current_time
    

    #with tf.device('GPU:3'):
    model = RANDAL(nature_cnn,
                predictor_cnn,
                input_shape = input_size,
                action_size = action_size,
                intr_coeff=1.0,
                extr_coeff=2.0,
                value_coeff=0.5,
                pixel_control=True,
                lr=1e-3,
                grad_clip=0.5,
                policy_args={},
                RND_args={}) #

    

    curiosity = RANDAL_Trainer(envs = envs,
                            model = model,
                            model_dir = model_dir,
                            log_dir = train_log_dir,
                            val_envs = val_envs,
                            train_mode = 'nstep',
                            total_steps = 50e6,
                            nsteps = nsteps,
                            init_obs_steps=128*50,
                            num_epochs=4,
                            num_minibatches=4,
                            validate_freq = 1e6,
                            save_freq = 0,
                            render_freq = 0,
                            num_val_episodes = 50,
                            log_scalars=True,
                            gpu_growth=True)
    curiosity.train()
    
    del curiosity

    


if __name__ == "__main__":
    env_id_list = ['FreewayDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'MontezumaRevengeDeterministic-v4',]
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1' ]
    for env_id in env_id_list:
        main(env_id)
    