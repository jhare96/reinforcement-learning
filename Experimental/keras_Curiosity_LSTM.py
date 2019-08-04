import tensorflow as tf
import numpy as np 
import gym 
from VecEnv import*
import time
import matplotlib.pyplot as plt
#tf.enable_v2_behavior()
#tf.enable_eager_execution()

@tf.function
def mask(x, idxs):
    mask = np.zeros_like(x)
    print('mask shape', mask.shape)
    for i in range(mask.shape[0]):
        mask[i] = idxs[i]
    return mask



class MaskedRNN(tf.keras.Model):
    def __init__(self, cell, cell_size):
        tf.keras.Model.__init__(self)
        self.cell = cell
        self.cell_size = cell_size
    
    def reset_batch_hidden(self, hidden, idxs, cell_size):
        return hidden * tf.keras.backend.reshape(tf.stack([tf.convert_to_tensor(idxs, dtype=tf.float32) for i in range(cell_size)], axis=1), (-1, idxs.shape[0], cell_size))
    
    @tf.function
    def call(self, x, hidden, masked=None):
        batch_size = x.shape[0]
        #print('hidden shape', hidden.shape)
        #x = tf.transpose(x, perm=[1,0,2]) # [time, batch, units]
        if masked is None:
            masked = np.zeros((x.shape[0],batch_size))
        seqs = []
        for t in range(x.shape[0]):
            output, hidden = self.cell(x[t], hidden) # self.reset_batch_hidden(hidden, masked[t], self.cell_size))
            seqs.append(output)
        #tf.transpose(, perm=[1,0,2])
        return tf.stack(seqs), hidden





class cnn_keras(tf.keras.Model):
    def __init__(self):
        tf.keras.Model.__init__(self)
        self.conv1 = tf.keras.layers.Conv2D(32,[8,8], [4,4], activation=tf.keras.activations.relu, dtype=tf.float32)
        self.conv2 = tf.keras.layers.Conv2D(32,[4,4], [2,2], activation=tf.keras.activations.relu, dtype=tf.float32)
        self.conv3 = tf.keras.layers.Conv2D(32,[3,3], [1,1], activation=tf.keras.activations.relu, dtype=tf.float32)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, tf.keras.activations.relu, dtype=tf.float32)
        
        
    def call(self, x):
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class mlp_keras(tf.keras.Model):
    def __init__(self, dense_size=64, num_layers=2, activation=tf.keras.activations.relu):
        tf.keras.Model.__init__(self)
        self.mlp_layers = []
        for i in range(num_layers):
            layer = tf.keras.layers.Dense(dense_size, activation=activation, dtype=tf.float32)
            self.mlp_layers.append(layer)
    
    def call(self, x):
        for i in range(len(self.mlp_layers)):
            x = self.mlp_layers[i](x)
        return x


def validate(model,env,num_ep,max_steps,render=False):
    episode_scores = []
    for episode in range(num_ep):
        state = env.reset()
        episode_score = []
        hidden = tf.zeros((1,256))
        for t in range(max_steps):
            policy, value, hidden = model(state[np.newaxis,np.newaxis], hidden)
            #policy = tf.keras.backend.get_value(policy)
            policy = policy.numpy()
            #action = np.argmax(policy)
            action = np.random.choice(policy.shape[1], p=policy[0])
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            if render:
                env.render()

            episode_score.append(reward)

            if done or t == max_steps -1:
                tot_reward = np.sum(episode_score)
                episode_scores.append(tot_reward)
                break
    
    avg_score = np.mean(episode_scores)
    return avg_score
    


class ICM(tf.keras.Model):
    def __init__(self, ICM_model, action_size):
        tf.keras.Model.__init__(self)
        self.action_size = action_size
       
        self.dense = ICM_model(288, num_layers=2)

        # forward model
        self.forward1 = tf.keras.layers.Dense(256, tf.keras.activations.relu, dtype=tf.float32)
        self.pred_state = tf.keras.layers.Dense(288, tf.keras.activations.relu, dtype=tf.float32)
        
        # inv model 
        self.inv1 = tf.keras.layers.Dense(256, tf.keras.activations.relu, dtype=tf.float32)
        self.pred_action = tf.keras.layers.Dense(action_size, tf.nn.softmax, dtype=tf.float32)

    @tf.function
    def call(self, state, action, next_state):
        #action = tf.keras.backend.one_hot(action, self.action_size)
    
        h = state / np.array(255.0, dtype=np.float32)
        phi_1 = self.dense(h)

        h = next_state / np.array(255.0, dtype=np.float32)
        phi_2 = self.dense(h)

        f = self.forward1(tf.keras.layers.concatenate([phi_1,phi_2]))
        pred_state = self.pred_state(f)

        i = self.inv1(tf.keras.layers.concatenate([phi_1,action]))
        pred_action = self.pred_action(i)

        return pred_state, pred_action, phi_2

class keras_ActorCritic_MLPLSTM(tf.keras.Model):
    def __init__(self, input_size, action_size, dense_size=64, num_layers=2,  cell_size=256, activation=tf.keras.activations.relu):
        tf.keras.Model.__init__(self)
        self.cell_size = cell_size
        self.input_size = input_size
        self.dense_size = dense_size

        #self.mlp_layers = []
        #for i in range(num_layers):
        self.mlp_1 = tf.keras.layers.Dense(dense_size, activation=activation, name='mlp_1')
        self.mlp_2 = tf.keras.layers.Dense(dense_size, activation=activation, name='mlp_2')

        #self.lstm = tf.keras.layers.SimpleRNN(cell_size, activation='sigmoid', return_state=True, time_major=False, return_sequences=True)
        self.cell = tf.keras.layers.GRUCell(cell_size, activation='sigmoid')
        self.lstm = MaskedRNN(self.cell, cell_size)
        self.policy = tf.keras.layers.Dense(action_size, activation=tf.nn.softmax, dtype=tf.float32, name='policy')
        self.value = tf.keras.layers.Dense(1, activation=None, dtype=tf.float32, name='value')

    @tf.function
    def call(self, x, prev_hidden, mask=None):
        # input[batch, time, ...], dtype=int32
        time = x.shape[0]
        x = keras_fold_batch(x)
        x = self.mlp_1(x)
        x = self.mlp_2(x)
        print('mlp shape', x.shape)
        x = tf.keras.backend.reshape(x, (time, -1, self.dense_size))
        x, hidden = self.lstm(x, prev_hidden, mask)
        print('lstm shape', x.shape)
        x = keras_fold_batch(x)
        print('unfolded shape', x.shape)
        #x = keras_fold_batch(x)
        #print('lstm shape', x.shape)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value, hidden
    
    def get_initial_hidden(self, batch_size):
        return (tf.zeros((1, batch_size, self.cell_size)))
    
    def reset_batch_hidden(self, hidden, idxs):
        return hidden * tf.stack([tf.convert_to_tensor(idxs, dtype=tf.float32) for i in range(self.cell_size)], axis=1)


class keras_ActorCritic_CNNLSTM(tf.keras.Model):
    def __init__(self, input_size, action_size, cell_size=256):
        tf.keras.Model.__init__(self)
        self.cell_size = cell_size
        self.input_size = input_size

        self.conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32,[8,8], [4,4], activation=tf.keras.activations.relu, dtype=tf.float32, name='conv1'))
        self.conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32,[4,4], [2,2], activation=tf.keras.activations.relu, dtype=tf.float32, name='conv1'))
        self.conv3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32,[3,3], [1,1], activation=tf.keras.activations.relu, dtype=tf.float32, name='conv1'))
        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512, tf.keras.activations.relu, dtype=tf.float32))

        self.cell = tf.keras.layers.GRUCell(cell_size, activation='sigmoid')
        self.lstm = MaskedRNN(self.cell, cell_size)
        self.policy = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(action_size, activation=tf.nn.softmax, dtype=tf.float32))
        self.value = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation=None, dtype=tf.float32))

    @tf.function
    def __call__(self, x, prev_hidden, mask=None):
        # input[batch, time, ...], dtype=int32
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x, hidden = self.lstm(x, prev_hidden, mask)
        policy = keras_fold_batch(self.policy(x))
        value = keras_fold_batch(self.value(x))
        return policy, value, hidden
    
    def get_initial_hidden(self, batch_size):
        return tf.zeros((batch_size, self.cell_size))
    
    def reset_batch_hidden(self, hidden, idxs):
        return hidden * tf.stack([tf.convert_to_tensor(idxs, dtype=tf.float32) for i in range(self.cell_size)], axis=1)
    
    

class Curiosity_LSTM_keras(object):
    def __init__(self, policy_model, ICM_model, input_size, action_size, cell_size=256, beta=0.2, instr_coeff=0.01):
        self.action_size = action_size
        self.beta = beta
        self.instr_coeff = instr_coeff
        if policy_model.lower() == 'cnn':
            self.policy_model = keras_ActorCritic_CNNLSTM(input_size, action_size, cell_size)
        elif policy_model.lower() == 'mlp':
            print('mlp')
            self.policy_model = keras_ActorCritic_MLPLSTM(input_size, action_size, cell_size)
        else:
            raise ValueError(policy_model, 'is not a supported type, supported models are "cnn" and "mlp" ')
        self.ICM = None
        self.optimiser = tf.keras.optimizers.Adam()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.cross_ent = tf.keras.losses.CategoricalCrossentropy()
    
    
    def get_initial_hidden(self, batch_size):
        return self.policy_model.get_initial_hidden(batch_size)
    
    def reset_batch_hidden(self, hidden, idxs):
        return self.policy_model.reset_batch_hidden(hidden, idxs)

    @tf.function
    def forward(self, state, hidden, dones):
        policy, value, hidden = self.policy_model(state, hidden)
        return tf.keras.backend.eval(policy), tf.keras.backend.eval(value).reshape(-1), hidden

    @tf.function
    def backprop(self, state, next_state, R, action, hidden, dones):
        # all inputs expected dimensions of [batch, time, ...]
        action_onehot = tf.keras.backend.one_hot(action, self.action_size)
        with tf.GradientTape() as tape:
            #tape.watch(hidden)
            # Policy Forward pass
            policy, value, h = self.policy_model(state, hidden, dones)
            ac_loss = self.policy_loss(policy, R-value, action_onehot) + 0.5 * self.value_loss(value,R) - 0.01 * self.entropy_loss(policy)
            #print('policy shape', policy.shape, 'value shape', value.shape)
            #print('onehot action', action_onehot.shape)
            
            # ICM Forward pass
            #state = keras_fold_batch(state)
            #next_state = keras_fold_batch(next_state)
            #y =  keras_fold_batch(y)
            #action_onehot = keras_fold_batch(action_onehot)
            #pred_state, pred_action, phi_2 = self.ICM(state,next_state,action_onehot)
            
            
            #icm_loss = self.beta * self.forward_loss(pred_state, phi_2) + (1-self.beta) * self.inverse_loss(pred_action, action_onehot)
            loss = ac_loss #+ icm_loss
        
        vars = self.policy_model.trainable_variables #+ self.ICM.trainable_variables
        grads = tape.gradient(loss, vars)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimiser.apply_gradients(zip(grads,vars))
        return loss
   
        
    @tf.function
    def intrinsic_reward(self, states, actions, next_states):
        actions_onehot = tf.keras.backend.one_hot(actions, self.action_size)
        pred_state, pred_action, phi_2 = self.ICM(states, actions_onehot, next_states)
        return self.instr_coeff * self.mse(pred_state, phi_2)


    #@tf.function
    def value_loss(self,value,R):
        return 0.5 * tf.reduce_mean(tf.square(R - value))

    #@tf.function
    def policy_loss(self, policy, adv, action_onehot):
        log_policy = tf.keras.backend.sum(tf.math.log(policy) * action_onehot, axis=1)
        return tf.reduce_mean(-log_policy * tf.stop_gradient(adv))
    
    #@tf.function
    def entropy_loss(self, policy):
        return -tf.reduce_mean(tf.reduce_sum(tf.math.log(policy) * policy, axis=1))
    
    
    def forward_loss(self, phi1, phi2):
        return self.mse(phi1, phi2)
    
    def inverse_loss(self, pred_action, action_onehot):
        l = self.cross_ent(action_onehot, pred_action)
        return l
    
#@tf.function
def fold_batch(x,dtype=np.float32):
    rows, cols = x.shape[0], x.shape[1]
    y = np.reshape(x, (rows*cols,*x.shape[2:]))
    return y

@tf.function
def keras_fold_batch(x):
    time, batch = x.shape[0], x.shape[1]
    return tf.keras.backend.reshape(x, shape=(time*batch, *x.shape[2:]) )

def unfold_batch(x,time):
    batch_size = x.shape[0] / time
    return x.reshape()





def main():
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    config = tf.compat.v1.ConfigProto() #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    num_envs = 32
    nsteps = 20
    env_id = 'LunarLander-v2'

    #writer = tf.summary.create_file_writer('logs/CUR_LSTM/')
    
    #env = BatchEnv(AtariEnv, env_id, num_envs, False, k=1)
    #val_env = AtariEnv(gym.make(env_id), episodic=False, clip_reward=False, k=1)
    env = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)
    val_env = DummyEnv(gym.make(env_id))
    input_shape = val_env.reset().shape
    action_size = val_env.action_space.n
    model = Curiosity_LSTM_keras('mlp', mlp_keras, input_shape, action_size)
    print('gpu aviabliable', tf.test.is_gpu_available())
    print('eagerly', tf.executing_eagerly())
    #print(model.policy_model.summary())
    states_ = env.reset()
    
    #hidden = tf.zeros((num_envs,256), dtype=tf.float32)
    prev_hidden = model.get_initial_hidden(num_envs)
    #hidden = model.policy_model.lstm_cell.cell.get_initial_state(inputs=1,batch_size=32,dtype=tf.float32)
    #hidden = model.policy_model.lstm_cell.cell.zero_state(32,dtype=tf.float32)
    
    # tf.summary.trace_on(graph=True, profiler=True)
    # policies, values, h = model.policy_model(states_, hidden)

    # with writer.as_default():
    #         tf.summary.trace_export(
    #             name="my_func_trace",
    #             step=0,
    #             profiler_outdir='logs/CUR_LSTM/')
    

    start = time.time()
    num_updates =  int(50e6 / (num_envs * nsteps))
    validate_freq = int(1e5 / (num_envs * nsteps))
    print('validate freq ', validate_freq)
    dones = np.ones((num_envs))
    for update in range(1,num_updates+1):
        memory = []
        for t in range(nsteps):
            #print('step', t)
            #print('states', states_.shape, states_.dtype)
            policies, values, hidden = model.policy_model(states_[np.newaxis], prev_hidden)
            #print('policyies', policies)
            policies = policies.numpy()
            actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
            #actions = np.argmax(policies, axis=1).astype(np.int32)
            #print('actions', actions)
            #exit()

            next_states_, extr_rewards, dones, infos = env.step(actions)
            next_states_ = next_states_.reshape(-1,*input_shape)
           
            rewards = extr_rewards #+ model.instr_reward(states_, next_states_)
            memory.append((states_, actions, rewards, next_states_, prev_hidden, dones, infos))
            states_ = next_states_
            prev_hidden = model.reset_batch_hidden(hidden, 1-dones)
        
        states, actions, rewards, next_states, hidden_batch, dones, infos = zip(*memory)

        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        #hidden_batch = np.stack(hidden_batch)
        dones = np.stack(dones)

        #print('states shape', states.shape, 'rewards shape', rewards.shape, 'dones', dones.shape)
        rewards = np.clip(rewards, -1, 1)
        T = rewards.shape[0]
        
        # Calculate R for advantage A = R - V 
        R = np.zeros((T,num_envs), dtype=np.float32)
        v = values.numpy()
        v = v.reshape(-1)
        #print('v', v.shape)
        
        #print('v shape', v.shape, 'rewards shape', rewards.shape)
        R[-1] =  v * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as wrapped env continues after end of episode
            R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[i])  
        
        #print('R.T shape', R.T.shape, 'R[-1]', R.T[:,-1])
        actions, R = keras_fold_batch(actions), keras_fold_batch(R)
        #print('states shape', states.shape, 'R shape', R.shape, 'dones', dones.shape)
        l = model.backprop(states, next_states, R, actions,  hidden_batch[0], dones.astype(np.float32))


    #print('hidden', hidden)
        
        if update % validate_freq == 0:
            tot_steps = update * num_envs * nsteps
            time_taken = time.time() - start
            fps = (num_envs * nsteps *  validate_freq) / time_taken
            score = validate(model.policy_model, val_env, 25, max_steps=10000, render=False)
            print('total steps %i, validation score %f, loss %f, fps %f' %(tot_steps,score,l,fps))
            start = time.time()



if __name__ == "__main__":
    main()