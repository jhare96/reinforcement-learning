import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
from collections import deque
from A2C import ActorCritic
from networks import*
from SyncMultiEnvTrainer import SyncMultiEnvTrainer
from VecEnv import*
from ActorCritic import ActorCritic_LSTM

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

        

class Auxiliary(object):
    def __init__(self,  policy_model, input_shape, action_size, num_envs, cell_size=256, forward_model_scale=0.2, intr_scale=1, RP=1, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}):
        self.RP, self.forward_model_scale, self.intr_scale = RP, forward_model_scale, intr_scale
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)

        
        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE):
            self.train_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, num_envs, cell_size, **policy_args)
            self.validate_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, **policy_args)
            enc_state = self.train_policy.dense

            with tf.variable_scope('encoder_network'):
                self.next_state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='next_state')
                enc_next_state = policy_model(self.next_state, **policy_args)
        
        with tf.variable_scope('Forward_Model'):
            state_size = enc_state.get_shape()[1].value
            actions_onehot = tf.one_hot(self.train_policy.actions, action_size)
            concat = tf.concat([enc_state, tf.dtypes.cast(actions_onehot, tf.float32)], 1, name='state-action-concat')
            f1 = mlp_layer(concat, 256, activation=tf.nn.relu, name='foward_model')
            pred_state = mlp_layer(f1, state_size, activation=None, name='pred_state')
            self.intr_reward = 0.5 * tf.reduce_sum(tf.square(pred_state - enc_next_state), axis=1) # l2 distance metric 
            self.forward_loss =  tf.reduce_mean(self.intr_reward) # intr_reward batch loss 
            print('forward_loss', self.forward_loss.get_shape().as_list())
        
        with tf.variable_scope('Inverse_Model'):
            concat = tf.concat([enc_state, enc_next_state], 1, name='state-nextstate-concat')
            f1 = mlp_layer(concat, 256, activation=tf.nn.relu, name='inverse_model')
            pred_action = mlp_layer(f1, action_size, activation=None, name='pred_state')
            self.inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_action, labels=actions_onehot)) # batch inverse loss
        
        self.reward_input = tf.placeholder(tf.float32, shape=[3, *input_shape], name='reward_input')
        with tf.variable_scope('encoder_network'):
                enc_reward_state = policy_model(self.next_state, **policy_args)
        with tf.variable_scope('reward_model'):
            self.reward_target = tf.placeholder(tf.float32, shape=[1])
            pred_reward = mlp_layer(tf.concat(enc_reward_state, axis=1), 1, activation=None)
            reward_loss = 0.5 * tf.reduce_mean(tf.square(self.reward_target - pred_reward))
        
        # with tf.variable_scope('ActorCritic', reuse=True):
        #     self.replay_AC = ActorCritic(policy_model, input_shape, action_size, lr, lr_final, decay_steps, grad_clip, **policy_args)
        # with tf.variable_scope('value_replay'):
        #     self.replay_R = tf.placeholder(dtype=tf.float32, shape=[None])
        #     replay_loss = 0.5 * tf.reduce_mean(tf.square(self.replay_R - self.replay_AC.V))

        
        self.loss = self.train_policy.loss  + intr_scale * ((1-forward_model_scale) * self.inverse_loss + forward_model_scale * self.forward_loss) + RP * reward_loss
        #self.loss = self.auxiliary_loss + self.on_policy_loss
        
        
        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))
        self.train_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)

    def forward(self, state, hidden, validate=False):
        if validate:
            return self.validate_policy.forward(state, hidden)
        else :
            return self.train_policy.forward(fold_batch(state), hidden)

    def backprop(self, states, R, actions, next_states, rewards, reward_states,  hidden):
        feed_dict = {self.train_policy.state:states, self.train_policy.actions:actions, 
                    self.train_policy.R:R, self.reward_target:rewards,
                    self.next_state:next_states, self.reward_input:reward_states, 
                    self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1]}
        _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l

    def get_initial_hidden(self, batch_size):
        return self.train_policy.get_initial_hidden(batch_size)
    
    def reset_batch_hidden(self, hidden, idxs):
        return self.train_policy.reset_batch_hidden(hidden, idxs)

    def set_session(self, sess):
        self.sess = sess
        self.train_policy.set_session(sess)
        self.validate_policy.set_session(sess)


class Auxiliary_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50):
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes)
        
        self.replay = deque([], maxlen=2000)
        self.runner = self.Runner(self.model, self.env, self.nsteps, self.replay)
    
    def populate_memory(self):
        for t in range(500):
            self.runner.run()
    
    def auxiliary_target(self, rewards, values, dones):
        T = len(rewards)
        R = np.zeros((T,*values.shape))
        dones = np.stack([dones for i in range(values.shape[1])], axis=-1)
        rewards = np.stack([rewards for i in range(values.shape[1])], axis=-1)
        #print('R shape', R.shape)
        #print('stack_shape', dones.shape)
        R[-1] = values * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[-1])
        
        return R

    def sample_replay(self):
        sample_start = np.random.randint(0, len(self.replay) -21)
        worker = np.random.randint(0,self.num_envs) # sample from one of n workers
        if self.replay[sample_start][4][worker] == True:
            sample_start += 1
        replay_sample = []
        for i in range(sample_start, sample_start+self.nsteps):
            replay_sample.append(self.replay[i])
        replay_states = np.stack([replay_sample[i][0][worker] for i in range(len(replay_sample))])
        replay_actions = np.stack([replay_sample[i][1][worker] for i in range(len(replay_sample))])
        replay_rewards = np.stack([replay_sample[i][2][worker] for i in range(len(replay_sample))])
        replay_dones = np.stack([replay_sample[i][4][worker] for i in range(len(replay_sample))])

        #print('replay_states', replay_sample[0][3].shape)
        prev_hidden = replay_sample[0][3][:,worker].reshape(2,1,-1)
        last_state = replay_states[-1:]
        #print('last_state shape', last_state.shape)
        _, replay_values, _= self.model.forward(last_state, prev_hidden, validate=True)
        #print('replay_values', replay_values)
        #print('replay_rewards', replay_rewards.shape)
        #print('replay_dones', replay_dones.shape)
        replay_R = self.multistep_target(replay_rewards[:,np.newaxis], replay_values, replay_dones)
        #print('replay_R', replay_R.shape)
        replay_states, replay_R = replay_states, self.fold_batch(replay_R)
        #print('replay_R', replay_R.shape)
        return replay_states, replay_actions, replay_rewards, replay_R
    
    
    def _train_nstep(self):
        '''
            Episodic training loop for synchronous training over multiple environments

            num_steps - number of Total training steps across all environemnts
            nsteps - number of steps performed in environment before updating ActorCritic
        '''
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        self.populate_memory()
        # main loop
        for t in range(1,num_updates+1):
            states, actions, rewards, next_states, hidden_batch, dones, infos, values = self.runner.run()

            R = self.multistep_target(rewards, values, dones, clip=False)


            #sample replay
            replay_states, replay_actions, replay_rewards, replay_R = self.sample_replay()
            #print('replay_R', replay_R.shape)
            #print('replay_states', replay_states.shape)
            
            # stack all states,  actions and Rs across all workers into a single batch
            states, actions, rewards, next_states, R = self.fold_batch(states), self.fold_batch(actions), self.fold_batch(rewards), self.fold_batch(next_states), self.fold_batch(R)
        
            l = self.model.backprop(states, R, actions, next_states, replay_rewards[-1:], replay_states[-3:],  hidden_batch[0])

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
            
    
    def get_action(self, state):
        policy, value = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        #action = np.argmax(policy)
        return action

    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps, replay):
            super().__init__(model, env, num_steps)
            self.replay = replay
            self.prev_hidden = self.model.get_initial_hidden(len(env))

        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values, hidden = self.model.forward(self.states[np.newaxis], self.prev_hidden)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
    
                next_states, rewards, dones, infos = self.env.step(actions)
                rollout.append((self.states, actions, rewards, next_states, self.prev_hidden, dones, infos))
                self.replay.append((self.states, actions, rewards, self.prev_hidden, dones, infos)) # add to replay memory
                self.states = next_states
                self.prev_hidden = self.model.reset_batch_hidden(hidden, 1-dones) # reset hidden state at end of episode
                
            
            states,  actions,  rewards, next_states, hidden_batch, dones, infos = zip(*rollout)
            states,  actions,  rewards, next_states, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)
            return states, actions, rewards,  next_states, hidden_batch, dones, infos, values
    
    def validate(self,env,num_ep,max_steps,render=False):
        episode_scores = []
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            hidden = self.model.get_initial_hidden(1)
            for t in range(max_steps):
                policy, value, hidden = self.model.forward(state[np.newaxis], hidden, validate=True)
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
    #action_size = env.action_space.n
    #input_size = env.reset().shape[0]
    
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(10)]
        envs = BatchEnv(NoRewardEnv, env_id, num_envs, blocking=False)

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=1, episodic=False, reset=reset, clip_reward=False) for i in range(10)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=1, reset=reset, episodic=True, clip_reward=False)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/Auxiliary/' + env_id + '/'
    model_dir = "models/Auxiliary/" + env_id + '/'

    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
   
    ac_mlp_args = {'dense_size':256}


    model = Auxiliary(nature_cnn,
                      input_shape = input_size,
                      action_size = action_size,
                      num_envs = num_envs,
                      cell_size = 256,
                      forward_model_scale = 0.2,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args=ac_mlp_args)

    

    auxiliary = Auxiliary_Trainer(envs = envs,
                                  model = model,
                                  file_loc = [model_dir, train_log_dir],
                                  val_envs = val_envs,
                                  train_mode = 'nstep',
                                  total_steps = 5e6,
                                  nsteps = nsteps,
                                  validate_freq = 1e5,
                                  save_freq = 0,
                                  render_freq = 0,
                                  num_val_episodes = 25)

    
    
    hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':num_envs,
                  'total_steps':auxiliary.total_steps, 'entropy_coefficient':0.01, 'value_coefficient':0.5}
    
    filename = train_log_dir + auxiliary.current_time + '/hyperparameters.txt'
    auxiliary.save_hyperparameters(filename, **hyper_paras)

    auxiliary.train()

    del auxiliary

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = [ 'FreewayDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'PongDeterministic-v4', 'MontezumaRevengeDeterministic-v4']
    #env_id_list = ['CartPole-v1', 'MountainCar-v0']
    for env_id in env_id_list:
        main(env_id)
    