import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
from A2C import ActorCritic
from networks.networks import*
from utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from utils.VecEnv import*
#from .OneNetCuriosity import Curiosity_onenet

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

class ICM(object):
    def __init__(self, model_head, input_size, action_size, reuse=None, **model_head_args):    
            
        print('input_Shape', input_size)
        self.input1 = tf.placeholder("float", shape=[None, *input_size], name="input")
        self.input2 = tf.placeholder("float", shape=[None, *input_size], name="input")
        
        self.action = tf.placeholder(tf.float32, shape=[None, action_size], name="actions")
        if reuse is None:
            name = 'encoder_network'
            reuse = False
        else:
            name = reuse + '/encoder_network'
            reuse = True
       
        with tf.variable_scope(name, reuse=reuse):
            self.phi1 = model_head(self.input1,  **model_head_args)
        with tf.variable_scope(name, reuse=True):
            self.phi2 = model_head(self.input2,  **model_head_args)
        

        with tf.variable_scope('Forward_Model'):
            state_size = self.phi1.get_shape()[1].value
            concat = tf.concat([self.phi1, self.action], 1, name='state-action-concat')
            f1 = mlp_layer(concat, state_size, activation=tf.nn.relu, name='foward_model')
            pred_state = mlp_layer(f1, state_size, activation=None, name='pred_state')
            print('pred_state', pred_state.get_shape().as_list(), 'phi_2', self.phi2.get_shape().as_list())
            self.intr_reward = 0.5 * tf.reduce_sum(tf.square(pred_state - self.phi2), axis=1)  # l2 distance metric ‖ˆφ(st+1)−φ(st+1)‖22
            self.forward_loss =  tf.reduce_mean(self.intr_reward) # intr_reward batch loss 
            print('forward_loss', self.forward_loss.get_shape().as_list())
        
        with tf.variable_scope('Inverse_Model'):
            concat = tf.concat([self.phi1, self.phi2], 1, name='state-nextstate-concat')
            i1 = mlp_layer(concat, state_size, activation=tf.nn.relu, name='inverse_model')
            actions = tf.dtypes.cast(self.action, tf.int32)
            pred_action = mlp_layer(i1, action_size, activation=None, name='pred_state')
            self.inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_action, labels=actions)) # batch inverse loss
            print('pred_action', pred_action.get_shape().as_list(), 'inverse_loss', self.inverse_loss.get_shape().as_list())
        
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        #exit()
        

class Curiosity(object):
    def __init__(self,  policy_model, ICM_model, input_size, action_size, forward_model_scale, policy_importance, reward_scale, prediction_beta, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, ICM_args={}):
        self.reward_scale, self.forward_model_scale, self.policy_importance, self.prediction_beta = reward_scale, forward_model_scale, policy_importance, prediction_beta
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size

        try:
            iterator = iter(input_size)
        except TypeError:
            input_size = (input_size,)
        
        with tf.variable_scope('ICM'):
            self.ICM = ICM(ICM_model, input_size, action_size, **ICM_args)
        
        with tf.variable_scope('ActorCritic'):
            self.AC = ActorCritic(policy_model, input_size, action_size, lr, lr_final, decay_steps, grad_clip, **policy_args)

        self.loss = policy_importance * self.AC.loss + reward_scale * ((1-forward_model_scale) * self.ICM.inverse_loss + forward_model_scale * self.ICM.forward_loss ) 
        
        
        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
        
        weights = self.AC.weights + self.ICM.weights
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))

        self.train_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)

    def forward(self, state):
        return self.AC.forward(state)
    
    def intrinsic_reward(self, state, action, next_state):
        action = one_hot(action, self.action_size)
        forward_loss = self.sess.run(self.ICM.intr_reward, feed_dict={self.ICM.input1:state, self.ICM.action:action, self.ICM.input2:next_state})
        intr_reward = forward_loss * self.prediction_beta 
        return intr_reward
    
    def backprop(self, state, next_state, R, actions):
        actions_onehot = one_hot(actions, self.action_size)
        feed_dict = {self.AC.state:state, self.AC.actions:actions, self.AC.R:R, self.ICM.input1:state, self.ICM.input2:next_state, self.ICM.action:actions_onehot }
        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.AC.set_session(sess)


class Curiosity_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50):
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes)
        self.runner = self.Runner(self.model, self.env, self.nsteps)
    
    
    def _train_nstep(self):
        '''
            Episodic training loop for synchronous training over multiple environments

            num_steps - number of Total training steps across all environemnts
            nsteps - number of steps performed in environment before updating ActorCritic
        '''
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        # main loop
        for t in range(1,num_updates+1):
            #start = time.time()
            states, next_states, actions, rewards, dones, infos, values = self.runner.run()
            #print('nsteps time', time.time() -start)
            R = self.multistep_target(rewards, values, dones, clip=False)  
                
            # stack all states, next_states, actions and Rs across all workers into a single batch
            states, next_states, actions, R = self.fold_batch(states), self.fold_batch(next_states), self.fold_batch(actions), self.fold_batch(R)
            
            #start= time.time()
            l = self.model.backprop(states,next_states,R,actions)
            #print('backprop time', time.time() -start)
            
            #start= time.time()
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
            
            #print('validate time', time.time() -start)
    
    def get_action(self, state):
        policy, value = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        #action = np.argmax(policy)
        return action

    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps):
            super().__init__(model, env, num_steps)

        def run(self,):
            memory = []
            for t in range(self.num_steps):
                start = time.time()
                policies, values = self.model.forward(self.states)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)
                intr_rewards = self.model.intrinsic_reward(self.states, actions, next_states)  * (1-dones)
                #print('intr_rewards', intr_rewards)
                rewards = extr_rewards + intr_rewards
                #print('rewards', rewards)
                memory.append((self.states.copy(), next_states, actions, rewards, dones, infos))
                self.states = next_states
           
            states, next_states, actions, rewards, dones, infos = zip(*memory)
            states, next_states, actions, rewards, dones = np.stack(states), np.stack(next_states), np.stack(actions), np.stack(rewards), np.stack(dones)
            return states, next_states, actions, rewards, dones, infos, values
            

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
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=42, episodic=False, reset=reset, clip_reward=False) for i in range(10)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=42, k=4, reset=reset, episodic=True, clip_reward=True)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/Curiosity/' +env_id + '/hyper_unclipped/'
    model_dir = "models/Curiosity/" + env_id + '/'

    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
   
    ac_mlp_args = {'dense_size':64}


    model = Curiosity(nature_cnn,
                      nature_cnn,
                      input_size = input_size,
                      action_size = action_size,
                      forward_model_scale=0.2,
                      policy_importance=0.1,
                      reward_scale=1,
                      prediction_beta=0.5,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=20.0,
                      policy_args={},
                      ICM_args={})

    

    curiosity = Curiosity_Trainer(envs = envs,
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
                  'total_steps':curiosity.total_steps, 'entropy_coefficient':0.01, 'value_coefficient':0.5, 'reward_scale':model.reward_scale,
                   'forward_model_scale':model.forward_model_scale, 'policy_importance':model.policy_importance, 'prediction_beta':model.prediction_beta}
    
    filename = train_log_dir + curiosity.current_time + '/hyperparameters.txt'
    curiosity.save_hyperparameters(filename, **hyper_paras)

    curiosity.train()

    del curiosity

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = [ 'FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4','SpaceInvadersDeterministic-v4', 'PongDeterministic-v4', 'FreewayDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1' ]
    #for i in range(5):
    for env_id in env_id_list:
        main(env_id)
    