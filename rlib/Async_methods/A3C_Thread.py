import tensorflow as tf
import numpy as np
import threading, multiprocessing
import queue
import gym
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from ActorCritic import ActorCritic_LSTM
from VecEnv import*
from networks import*

main_lock = threading.Lock()

class ActorCritic_LSTM_(object):
    def __init__(self, model_head, input_shape, action_size, num_envs, cell_size, dense_size=512,
                 lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip = 0.5, opt=False, **model_head_args):
        self.lr, self.lr_final = lr, lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.cell_size = cell_size

        with tf.variable_scope('input'):
            self.state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='time_batch_state') # [time*batch, *input_shape]

        with tf.variable_scope('encoder_network'):
            dense = model_head(self.state, dense_size=dense_size, **model_head_args)
            unfolded_state = tf.expand_dims(dense, 0, name='expanded_state')
        
        with tf.variable_scope('lstm'):
            hidden_in = tf.placeholder(tf.float32, [None, cell_size], name='hidden_in')
            cell_in = tf.placeholder(tf.float32, [None, cell_size], name='cell_in')
            self.hidden_in = (cell_in, hidden_in)
    
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
            state_in = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(cell_in, hidden_in)

            lstm_output, self.hidden_out = tf.compat.v1.nn.dynamic_rnn(lstm_cell, unfolded_state, initial_state=state_in, time_major=False)
            #print('hidden_out', self.hidden_out)
            #print('self.output', lstm_output)
            #self.output, self.hidden_out, self.cell_out = lstm(unfolded_state, [self.hidden_in, self.cell_in])
            lstm_output = tf.squeeze(lstm_output, 0, name='squeezed_lstm_output')

        with tf.variable_scope('critic'):
            self.V = tf.reshape( mlp_layer(lstm_output, 1, name='state_value', activation=None), shape=[-1])

        
        with tf.variable_scope("actor"):
            self.policy_distrib = mlp_layer(lstm_output, action_size, activation=tf.nn.softmax, name='policy_distribution')
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.y = tf.placeholder(dtype=tf.float32, shape=[None])
            Advantage = self.y - self.V
            value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage))

            log_policy = tf.math.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
            log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
            policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(Advantage))

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
    
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        self.loss =  policy_loss + 0.5 * value_loss - 0.01 * entropy

        if opt:
            global_step = tf.Variable(0, trainable=False)
            tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)

            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)

            
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)
    
    def get_initial_hidden(self,batch_size):
        return np.zeros((batch_size, self.cell_size)), np.zeros((batch_size, self.cell_size))
    
    def reset_batch_hidden(self, hidden, idxs):
        return hidden * np.stack([idxs for i in range(self.cell_size)], axis=1)

    def forward(self,sess,state,hidden):
        policy, value, hidden = sess.run([self.policy_distrib, self.V, self.hidden_out], feed_dict = {self.state:state, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1]})
        return policy, value, hidden

    def backprop(self,sess,state,y,a,hidden):
        *_,l = sess.run([self.train_op, self.loss], feed_dict = {self.state : state, self.y : y, self.actions: a, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1]})
        return l

class Actor(ActorCritic_LSTM_):
    def __init__(self, model, optimiser, global_step, input_shape, action_size, dense_size, cell_size, grad_clip=0.5, lr=1-3, **model_args):
        ActorCritic_LSTM_.__init__(self, model, input_shape, action_size, 1, cell_size, dense_size=dense_size, opt=False, **model_args)
        self.sess = None
        if not tf.get_variable_scope().name == 'master':
            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='master')))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)

    def forward(self, state, hidden):
        sess = self.sess
        policy, value, hidden = sess.run([self.policy_distrib, self.V, self.hidden_out], feed_dict = {self.state:state, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1]})
        return policy, value, hidden
    
    def backprop(self, state, y, a, hidden):
        sess = self.sess
        l, _ = sess.run([self.loss, self.train_op], feed_dict = {self.state:state, self.y:y, self.actions:a, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1]})
        return l 


class Worker(threading.Thread):
    def __init__(self, ID, sess, T, episode, device, queue, lock, env_constr, model_constr, daemon, env_args={}, model_args={}, nsteps=20, max_rollouts=5e6):
        threading.Thread.__init__(self, daemon=daemon)
        self.sess = sess
        self.queue = queue
        self.lock = lock
        self.nsteps = nsteps
        self.max_rollouts = int(max_rollouts)
        self.env = env_constr(**env_args)
        if 'gpu' in device.lower():
            with tf.variable_scope('worker_' + str(ID)):
                self.model = model_constr(**model_args)
        else:
            with tf.device(device):
                with tf.variable_scope('worker_' + str(ID)):
                    self.model = model_constr(**model_args)
        
        self.model.sess = self.sess
        self.workerID = ID
        self.T = T
        self.episode = episode
        # self.update_local = [tf.assign(new, old) for (new, old) in 
        #     zip(tf.trainable_variables(scope='worker_' + str(self.workerID) ), tf.trainable_variables('master'))]
        
        network_weights =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='master')
        self.update_local = [v1.assign(v2) for v1, v2 in zip(self.model.weights, network_weights)]
    
    def run(self):
        self.train()
    
    def train(self):
        prev_hidden = self.model.get_initial_hidden(1)
        env = self.env
        state = env.reset().astype(np.float32)
        episode = 0
        epsiode_reward = []
        loss = 0
        frames = 0
        frames_start = time.time()
        while True:
            memory = []
            start = time.time()
            for t in range(self.nsteps):
                #print('state', state.dtype, 'h', prev_hidden.dtype)
                policy, value, hidden = self.model.forward(state[np.newaxis].astype(np.float32), prev_hidden)
                #print('plicy shape', policy)
                action = np.random.choice(policy.shape[1], p=policy[0])

                next_state, reward, done, info = env.step(action)
                epsiode_reward.append(reward)

                # if self.workerID == 0:
                #     with main_lock:
                #         env.render()
            
                memory.append((state, action, reward, next_state, prev_hidden, done, info))
                states_ = next_state
                prev_hidden = hidden

                self.T[:] += 1
                frames += 1
                
                if done:
                    state = env.reset()
                    prev_hidden = self.model.get_initial_hidden(1)
                    time_taken = time.time() - frames_start
                    fps = frames / time_taken
                    if episode % 500 == 0 and episode > 0:
                        print('worker %i, episode %i, total_steps %i, episode reward %f, loss %f, local fps %f' %(self.workerID, episode, self.T, np.sum(epsiode_reward),loss, fps))
                    epsiode_reward = []
                    frames_start = time.time()
                    frames = 0
                    self.episode[:] += 1
                    break
            
            end = time.time()
            #print('nsteps time', end-start)
            states, actions, rewards, next_states, hidden_batch, dones, infos = zip(*memory)

            states = np.stack(states)
            actions = np.stack(actions)
            rewards = np.stack(rewards)
            next_states = np.stack(next_states)
            dones = np.stack(dones)
            rewards = np.clip(rewards, -1, 1)
            T = rewards.shape[0]
            
            # Calculate R for advantage A = R - V 
            R = np.zeros((T), dtype=np.float32)
            v = value.reshape(-1)
    
            R[-1] =  v * (1-dones[-1])
            
            for i in reversed(range(T-1)):
                # restart score if done as wrapped env continues after end of episode
                R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[i])  
            
            #states, actions, R, next_states = fold_batch(states), fold_batch(actions), fold_batch(R), fold_batch(next_states)
            #print('states', states.dtype, 'R shape', R.dtype, 'dones', dones.dtype)
            start2 = time.time()

            loss = self.model.backprop(states, R, actions, hidden_batch[0])

            end2 = time.time()
            #print('backprop time', end2-start2)

            start3 = time.time()
            self.sess.run(self.update_local)
            end3 = time.time()
            #print('set weights time', end3-start3)



def actor_constructor(model, **model_args):
    return Actor(model, **model_args)

def env_constructor(env_constr, env_id, **env_args):
    return env_constr(gym.make(env_id), **env_args)

def main():
    config = tf.compat.v1.ConfigProto() #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.compat.v1.Session(config=config)


    env_id = 'CartPole-v1'
    train_log_dir = 'logs/A3C_Thread/' + env_id + '/'

    env = gym.make(env_id)
    action_size = env.action_space.n
    input_shape = env.reset().shape
    #input_shape = [84,84,1]
    env.close()

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.polynomial_decay(1e-3, global_step, 10e6, end_learning_rate=0, power=1.0, cycle=False, name=None)
    #
    # model, optimiser, global_step, input_shape, action_size, dense_size, cell_size, grad_clip=0.5, **model_args)
    # optimiser = tf.train.RMSPropOptimizer(7e-4, decay=0.99, epsilon=1e-3)
    model_args = {'model':mlp, 'optimiser':None, 'global_step':global_step, 'action_size':action_size, 'input_shape':input_shape, 'cell_size':256, 'dense_size':128, 'grad_clip':0.5, 'lr':lr}
    with tf.device('/cpu:0'):
        with tf.variable_scope('master'):
            master = Actor(**model_args)

    num_workers = 8
    T = np.array([0], dtype=np.int32)
    episode = np.array([0], dtype=np.int32)

    env_args = {'env_constr':DummyEnv, 'env_id':env_id}# 'k':1, 'clip_reward':False}
    env_constr = env_constructor
    model_constr = actor_constructor

    lock = threading.Lock()
    #self, ID, sess, T, device, queue, lock, env_constr, model_constr, daemon, env_args={}, model_args={}, nsteps=20, max_rollouts=5e6)
    workers = [Worker(i, sess, T, episode, '/cpu:0', None, lock, env_constr, model_constr, True, env_args, model_args) for i in range(num_workers)]


    init = tf.global_variables_initializer()
    sess.run(init)

    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    try:
        for w in workers:
            w.start()
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers:
            w.env.close()
            w.join()


if __name__ == "__main__":
    main()