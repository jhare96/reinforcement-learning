import numpy as np 
import tensorflow as tf
import threading
import time
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


from VecEnv import*
from utils import fold_batch



class ActorCritic_LSTM(object):
    def __init__(self, num_actions, cell_size=128):
        self.cell_size = cell_size
        self.num_actions = num_actions
    
    @tf.function
    def loss(self, policies, values, actions, R):
        actions_onehot = tf.one_hot(actions, self.num_actions)
        Advantage = R - values
        value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage))

        log_policy = tf.math.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
        log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
        policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(Advantage))

        entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
        return policy_loss + 0.5 * value_loss - 0.01 * entropy

    def value_loss(self,value,R):
        return 0.5 * tf.reduce_mean(tf.square(R - value))

    def policy_loss(self, policy, adv, action_onehot):
        log_policy = tf.keras.backend.sum(tf.math.log(policy) * action_onehot, axis=1)
        return tf.reduce_mean(-log_policy * tf.stop_gradient(adv))
    
    def entropy_loss(self, policy):
        return -tf.reduce_mean(tf.reduce_sum(tf.math.log(policy) * policy, axis=1))
    
    def get_initial_hidden(self, batch_size):
        return (tf.zeros((batch_size, self.cell_size)))
    
    def reset_batch_hidden(self, hidden, idxs):
        return hidden * tf.stack([tf.convert_to_tensor(idxs, dtype=tf.float32) for i in range(self.cell_size)], axis=1)

class ActorCritic_MLPLSTM(tf.keras.Model):
    def __init__(self, input_shape, action_size, dense_size=64, num_layers=2, cell_size=256, activation=tf.keras.activations.relu):
        tf.keras.Model.__init__(self)
        self.cell_size = cell_size
        self.num_actions = action_size
        self.dense_size = dense_size

        self.mlp_1 = tf.keras.layers.Dense(dense_size, activation=activation, name='mlp_1')
        self.mlp_2 = tf.keras.layers.Dense(dense_size, activation=activation, name='mlp_2')

        #self.lstm = tf.keras.layers.SimpleRNN(cell_size, activation='sigmoid', return_state=True, time_major=False, return_sequences=True)
        self.GRU = tf.keras.layers.GRU(cell_size, activation='sigmoid', return_sequences=True, return_state=True)
        self.policy = tf.keras.layers.Dense(action_size, activation=tf.nn.softmax, dtype=tf.float32, name='policy')
        self.value = tf.keras.layers.Dense(1, activation=None, dtype=tf.float32, name='value')

    #@tf.function
    def call(self, x, prev_hidden):
        # input[time, ...], dtype=int32
        x = self.mlp_1(x)
        x = self.mlp_2(x)
        #print('mlp shape', x.shape)
        x = tf.expand_dims(x, 0) # fake batch dimension
        x, hidden = self.GRU(x, prev_hidden)
        #print('lstm shape', x.shape)
        x = tf.keras.backend.reshape(x, (-1, self.cell_size))
        #print('unfolded shape', x.shape)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value, hidden
    
    
    def loss(self, policies, values, actions, R):
        actions_onehot = tf.keras.backend.one_hot(actions, self.num_actions)
        return self.policy_loss(policies, R-values , actions_onehot) + 0.5 * self.value_loss(values, R) + 0.01 * self.entropy_loss(policies)
    
    
    def value_loss(self,value,R):
        return 0.5 * tf.reduce_mean(tf.square(R - value))
    
    
    def policy_loss(self, policy, adv, action_onehot):
        log_policy = tf.keras.backend.sum(tf.math.log(policy) * action_onehot, axis=1)
        return tf.reduce_mean(-log_policy * tf.stop_gradient(adv))
    
    
    def entropy_loss(self, policy):
        return -tf.reduce_mean(tf.reduce_sum(tf.math.log(policy) * policy, axis=1))
    
    
    def get_initial_hidden(self, batch_size):
        return (tf.zeros((batch_size, self.cell_size)))
    
    
    def reset_batch_hidden(self, hidden, idxs):
        return hidden * tf.stack([tf.convert_to_tensor(idxs, dtype=tf.float32) for i in range(self.cell_size)], axis=1)
    
    


class Worker(threading.Thread):
    def __init__(self, env_id, global_network, T, shared_optimiser, nsteps=5, model_args={}):
        threading.Thread.__init__(self)
        self.env = Env(gym.make(env_id).unwrapped)
        self.global_net = global_network
        with tf.device('/cpu:0'):
            self.local_net = ActorCritic_MLPLSTM(**model_args)
        self.local_net(self.env.reset()[np.newaxis].astype(np.float32), self.local_net.get_initial_hidden(1))
        self.local_net.set_weights(self.global_net.get_weights())
        self.T = T
        self.nsteps = nsteps
        self.optimiser = shared_optimiser 
    
    def run(self):
        self.train()
    
    def train(self):
        prev_hidden = self.local_net.get_initial_hidden(1)
        env = self.env
        state = env.reset().astype(np.float32)
        episode = 0
        epsiode_reward = []
        loss = 0
        while True:
            memory = []
            start = time.time()
            for t in range(self.nsteps):
                #print('state', state.dtype, 'h', prev_hidden.dtype)
                policy, value, hidden = self.local_net(tf.convert_to_tensor(state[np.newaxis], dtype=tf.float32), prev_hidden)
                action = np.random.choice(policy.shape[1], p=policy.numpy()[0])

                next_state, reward, done, info = env.step(action)
                epsiode_reward.append(reward)
            
                memory.append((state, action, reward, next_state, prev_hidden, done, info))
                states_ = next_state
                prev_hidden = hidden

                self.T[:] += 1
                
                if done:
                    state = env.reset()
                    prev_hidden = self.local_net.get_initial_hidden(1)
                    if episode % 1 == 0:
                        print('episode %i, total_steps %i, episode reward %f, loss %f' %(episode, self.T, np.sum(epsiode_reward),loss))
                    epsiode_reward = []
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
            v = value.numpy()
            v = v.reshape(-1)
    
            R[-1] =  v * (1-dones[-1])
            
            for i in reversed(range(T-1)):
                # restart score if done as wrapped env continues after end of episode
                R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[i])  
            
            #states, actions, R, next_states = fold_batch(states), fold_batch(actions), fold_batch(R), fold_batch(next_states)
            #print('states', states.dtype, 'R shape', R.dtype, 'dones', dones.dtype)
            start2 = time.time()
            with tf.GradientTape() as tape:
                policies, values, hidden = self.local_net(states, hidden_batch[0])
                loss = self.local_net.loss(policies, values, actions, R)
            
            vars = self.local_net.trainable_variables
            grads = tape.gradient(loss, vars)
            grads, _ = tf.clip_by_global_norm(grads, 0.5)
            self.optimiser.apply_gradients(zip(grads,self.global_net.trainable_variables))
            end2 = time.time()
            #print('backprop time', end2-start2)

            start3 = time.time()
            self.local_net.set_weights(self.global_net.get_weights())
            end3 = time.time()
            #print('set weights time', end3-start3)

            episode+=1



def main():
    config = tf.compat.v1.ConfigProto() #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


    env_id = 'CartPole-v0'
    env = gym.make(env_id)
    action_size = env.action_space.n
    input_shape = env.reset().shape
    env.close()

    model_args = {'action_size':action_size, 'input_shape':input_shape, 'cell_size':128, 'dense_size':64}
    global_net = ActorCritic_MLPLSTM(**model_args)
    optimiser = tf.keras.optimizers.RMSprop()
    # 'build' model by calling 
    global_net(env.reset()[np.newaxis].astype(np.float32), global_net.get_initial_hidden(1))

    num_workers = 16
    T = np.array([1], dtype=np.int32)

    workers = [Worker(env_id, global_net, T, optimiser, 20, model_args) for i in range(num_workers)]

    try:
        for w in workers:
            w.start()
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers:
            w.join()


if __name__ == "__main__":
    main()