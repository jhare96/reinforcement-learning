import tensorflow as tf 
import numpy as np 
import gym
import threading
from networks import*
from SyncMultiEnvTrainer import SyncMultiEnvTrainer
from VecEnv import*
from Qvalue import Qvalue
import time

main_lock = threading.Lock()

def save_hyperparameters(filename, **kwargs):
    handle = open(filename, "w")
    for key, value in kwargs.items():
        handle.write("{} = {}\n" .format(key, value))
    handle.close()



class DQN(object):
    def __init__(self, model, input_shape, action_size, name, learning_rate=0.00025, grad_clip = 0.5, decay_steps=50e6, learning_rate_final=0, **model_args):
        self.learning_rate = learning_rate
        self.learning_rate_final = learning_rate_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.name = name 
        self.action_size = action_size
        self.sess = None

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)

        with tf.variable_scope(name):
            with tf.variable_scope('encoder_network'):
                self.state = tf.placeholder(tf.float32, shape=[None, *input_shape])
                self.dense = model(self.state, **model_args)
    
            with tf.variable_scope("State_Action"):
                self.Qsa = mlp_layer(self.dense, action_size, activation=None)
                self.R = tf.placeholder("float", shape=[None])
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                actions_onehot = tf.one_hot(self.actions, action_size)
                self.Qvalue = tf.reduce_sum(tf.multiply(self.Qsa, actions_onehot), axis = 1)
                self.loss = tf.reduce_mean(tf.square(self.R - self.Qvalue))
        
        
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=learning_rate_final, power=1.0, cycle=False, name=None)
            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)
            #self.train_op = optimiser.minimize(self.loss, global_step=global_step)
            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)

    
    def forward(self, state):
        return self.sess.run(self.Qsa, feed_dict = {self.state: state})

    def backprop(self, states, R, actions):
        _,l = self.sess.run([self.train_op,self.loss], feed_dict = {self.state:states, self.R:R, self.actions:actions})
        return l
    
    def set_session(self, sess):
        self.sess = sess




class SyncDDQN(SyncMultiEnvTrainer):
    def __init__(self, envs, model, target_model, file_loc, val_envs, action_size,
                     train_mode='nstep', total_steps=1000000, nsteps=5,
                     validate_freq=1e6, save_freq=0, render_freq=0, update_target_freq=10000,
                     epsilon_start=1, epsilon_final=0.01, epsilon_steps = 1e6, epsilon_test=0.01):

        
        super().__init__(envs=envs, model=model, file_loc=file_loc, val_envs=val_envs, train_mode=train_mode, total_steps=total_steps,
                         nsteps=nsteps, validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq, update_target_freq=update_target_freq)
        
        self.target_model = target_model
        self.epsilon = np.array([epsilon_start], dtype=np.float64)
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        schedule = self.linear_schedule(self.epsilon , epsilon_final, epsilon_steps//self.num_envs)
        self.epsilon_test = np.array([epsilon_test], dtype=np.float64)

        self.action_size = action_size
        self.runner = SyncDDQN.Runner(self.model, self.target_model, self.epsilon, schedule, self.env, self.num_envs, self.nsteps, self.action_size)
        
        self.update_weights = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='QTarget'), tf.trainable_variables('Q'))]
        
        self.target_model.set_session(self.sess)
        
    
    def get_action(self, state):
        if np.random.uniform() < self.epsilon_test:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.model.forward(state))
        return action

    def update_target(self):
        self.sess.run(self.update_weights)

            

    
    class Runner(object):
        def __init__(self, Q, TargetQ, epsilon, epsilon_schedule, env, num_envs, num_steps, action_size):
            self.Q = Q
            self.TargetQ = TargetQ
            self.epsilon = epsilon
            self.schedule = epsilon_schedule
            self.env = env
            self.num_envs = num_envs
            self.num_steps = num_steps
            self.action_size = action_size
            
            self.states = self.env.reset()
            
        def one_hot(self,x,n_labels):
            return np.eye(n_labels)[x]
        
        def run(self):
            memory = []
            for t in range(self.num_steps):
                actions = np.argmax(self.Q.forward(self.states), axis=1)
                random = np.random.uniform(size=(self.num_envs))
                random_actions = np.random.randint(self.action_size, size=(self.num_envs))
                actions = np.where(random < self.epsilon, random_actions, actions)
                next_states, rewards, dones, infos = self.env.step(actions)
                memory.append((self.states, actions, rewards, dones, infos))
                self.states = next_states
                self.schedule.step()
            
            states, actions, rewards, dones, infos = zip(*memory)
            states, actions, rewards, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(dones)
            action_values = self.TargetQ.forward(states[-1]) # Q(s,a; theta-1)
            actions_one_hot = self.one_hot(actions[-1],self.action_size)
            values = np.sum(action_values * actions_one_hot, axis=1) # Q(s, argmax_a Q(s,a; theta); theta-1)
            # values = np.argmax(self.TargetQ.forward(self.sess,self.states), axis=1)
            return states, actions, rewards, dones, infos, values
    
    class linear_schedule(object):
        def __init__(self, epsilon, epsilon_final, num_steps=1000000):
            self._counter = 0
            self._epsilon = epsilon
            self._epsilon_final = epsilon_final
            self._step = (epsilon - epsilon_final) / num_steps
            self._num_steps = num_steps
        
        def step(self,):
            if self._counter < self._num_steps :
                self._epsilon -= self._step
                self._counter += 1
            else:
                self._epsilon[:] = self._epsilon_final
        
        def get_epsilon(self,):
            return self._epsilon


def stackFireReset(env):
    return StackEnv(FireResetEnv(env))


def main(env_id, epsilon_final, epsilon_steps):
    num_envs = 32
    nsteps = 5

    train_log_dir = 'logs/SyncDoubleDQN/' + env_id +'/'
    model_dir = "models/SyncDoubleDQN/" + env_id + '/'

    env = gym.make(env_id)
    
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
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False , k=4, reset=reset, episodic=True, clip_reward=True)

    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape

    env.close()
    print('action space', action_size)

    
    
    
    dqn_cnn_args = {'input_shape':[84,84,4], 'action_size':action_size,  'learning_rate':1e-3, 'learning_rate_final':0, 'grad_clip':0.5, 'decay_steps':50e6/(num_envs*nsteps),
                    'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
   
    dqn_mlp_args = {'input_shape':input_size, 'dense_size':64, 'action_size':action_size, 'learning_rate':1e-3, 'grad_clip':0.5, 'decay_steps':5e6//(num_envs*nsteps), 'learning_rate_final':0}

      
    Q = DQN(nature_cnn, **dqn_cnn_args, name='Q')
    TargetQ = DQN(nature_cnn, **dqn_cnn_args, name='QTarget')  

    

    DDQN = SyncDDQN(envs = envs,
                    model = Q,
                    target_model = TargetQ,
                    file_loc = [model_dir, train_log_dir],
                    val_envs = val_envs,
                    action_size = action_size,
                    train_mode ='nstep',
                    total_steps = 50e6,
                    nsteps = nsteps,
                    validate_freq = 1e6,
                    save_freq = 0,
                    render_freq = 0,
                    update_target_freq = 10000,
                    epsilon_start = 1,
                    epsilon_final = epsilon_final,
                    epsilon_steps = epsilon_steps,
                    epsilon_test = 0.01)
    
    hyperparas = {'learning_rate':DDQN.model.learning_rate, 'learning_rate_final':DDQN.model.learning_rate_final, 'lr_decay_steps':DDQN.model.decay_steps , 'grad_clip':DDQN.model.grad_clip, 'nsteps':nsteps, 'num_workers':num_envs,
                  'total_steps':DDQN.total_steps, 'epsilon_start':DDQN.epsilon, 'epsilon_final':DDQN.epsilon_final, 'epsilon_steps':DDQN.epsilon_steps, 'update_freq':10000}
    
    filename = train_log_dir + DDQN.current_time + '/' + 'hyperparameters.txt'
    save_hyperparameters(filename , **hyperparas)
    
     
    
    DDQN.train()

    del DDQN

    tf.reset_default_graph()

if __name__ == "__main__":
    env_id_list = [ 'SpaceInvadersDeterministic-v4', 'PongDeterministic-v4', 'FreewayDeterministic-v4',  'SeaquestDeterministic-v4']
    env_id_list = ['MountainCar-v0', 'Acrobot-v1', ]
    #for i in range(5):
    for env_id in env_id_list:
        main(env_id, 0.01, 2e6)
   # env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v0']
