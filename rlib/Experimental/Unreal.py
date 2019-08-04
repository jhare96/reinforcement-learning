import tensorflow as tf
import numpy as np
from VecEnv import*
from ActorCritic import ActorCritic_LSTM
from utils import one_hot, fold_batch
from networks import*
from SyncMultiEnvTrainer import SyncMultiEnvTrainer

#(self, model_head, input_shape, action_size, num_envs, cell_size, dense_size=512,
               #  lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip = 0.5, opt=False, **model_head_args):

class UnrealA2C(object):
    def __init__(self, input_size, action_size, num_envs, PC_coeff, RP_coeff, VR_coeff, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5):
        self.PC_coeff, self.RP_coeff, self.RP_coeff = PC_coeff, RP_coeff, VR_coeff 
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.sess = None

        try:
            iterator = iter(input_size)
        except TypeError:
            input_size = (input_size,)
        
        with tf.variable_scope('Policy', reuse=tf.AUTO_REUSE):
            self.train_policy = ActorCritic_LSTM(model_head=nips_cnn, input_shape=input_size, action_size=action_size, num_envs=num_envs, cell_size=256, opt=False)
            self.validate_policy = ActorCritic_LSTM(model_head=nips_cnn, input_shape=input_size, action_size=action_size, num_envs=1, cell_size=256, opt=False)

        with tf.variable_scope('Pixel_Control'):
            # ignoring cropping from paper hence deconvoluting to size 21x21 feature map (as 84x84 / 4 == 21x21)
            feat_map = mlp_layer(self.train_policy.lstm_output, 32*8*8, activation=tf.nn.relu, name='feat_map_flat')
            feat_map = tf.reshape(feat_map, shape=[-1,8,8,32], name='feature_map')
            batch_size = tf.shape(feat_map)[0]
            deconv1 = conv_transpose_layer(feat_map, output_shape=[batch_size,10,10,32], kernel_size=[3,3], strides=[1,1], padding='VALID', activation=tf.nn.relu)
            deconv_advantage = conv_transpose_layer(deconv1, output_shape=[batch_size,21,21,action_size],
                    kernel_size=[3,3], strides=[2,2], padding='VALID', activation=None, name='deconv_adv')
            deconv_value = conv_transpose_layer(deconv1, output_shape=[batch_size,21,21,1],
                    kernel_size=[3,3], strides=[2,2], padding='VALID', activation=None, name='deconv_value')

            # Auxillary Q value calculated via dueling network 
            # Z. Wang, N. de Freitas, and M. Lanctot. Dueling Network Architectures for Deep ReinforcementLearning. https://arxiv.org/pdf/1511.06581.pdf
            Q_aux = tf.nn.relu(deconv_value + deconv_advantage - tf.reduce_mean(deconv_advantage, axis=3, keep_dims=True))
            
            self.QauxTD_target = tf.placeholder("float", [None, 21, 21]) # temporal difference target for Q_aux
            one_hot_actions = tf.one_hot(self.train_policy.actions, action_size)
            pixel_action = tf.reshape(one_hot_actions, shape=[-1,1,1, action_size], name='pixel_action')
            Q_aux_action = tf.reduce_sum(Q_aux * pixel_action, axis=3)
            pixel_loss = 0.5 * tf.reduce_mean(self.QauxTD_target - Q_aux_action) # l2 loss for Q_aux over all pixels and batch


        self.reward_input = tf.placeholder(tf.float32, shape=[None, *input_size], name='reward_input')
        self.reward_target = tf.placeholder(tf.float32, shape=[None], name='reward_target')
        
        with tf.variable_scope('Policy/encoder_network', reuse=True):
                dense = nips_cnn(self.reward_input)
        with tf.variable_scope('Reward_Prediction'):
            reward_1 = mlp_layer(tf.concat(dense, axis=1), 128, name='reward_layer')
            reward_pred = mlp_layer(reward_1, 3, name='reward_logits') # predicted reward of multinomial distribution [+ve, -ve, zero] reward
            reward_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.reward_target, logits=reward_pred))
        
        
        self.auxiliary_loss = RP_coeff * reward_loss + PC_coeff * pixel_loss + VR_coeff * self.train_policy.loss
        #print('aux loss', self.auxiliary_loss.get_shape().as_list())
        self.policy_loss = self.train_policy.loss     
        
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
        aux_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        with tf.variable_scope('auxiliary_grads'):
            print('aux weights', aux_weights)
            aux_grads = tf.gradients(self.auxiliary_loss, aux_weights, name='gradients_auxiliary')
            print('aux grads', aux_grads)
            aux_grads, _ = tf.clip_by_global_norm(aux_grads, grad_clip)
            aux_grads_vars = list(zip(aux_grads, aux_weights))
            self.aux_train_op = self.optimiser.apply_gradients(aux_grads_vars)
        #print('aux grads and vars', aux_grads_vars)

        with tf.variable_scope('on_policy_grads'):
            policy_weights = self.train_policy.weights
            #print('\n policy_weights weights', policy_weights)
            policy_grads = tf.gradients(self.policy_loss, policy_weights, name='gradients_policy')
            policy_grads, _ = tf.clip_by_global_norm(policy_grads, grad_clip)
            policy_grads_vars = list(zip(policy_grads, policy_weights))
            self.policy_train_op = self.optimiser.apply_gradients(policy_grads_vars)
        

    def forward(self, state, hidden):
        return self.train_policy.forward(state, hidden)
    
    def aux_backprop(self, state, next_state, R, actions, hidden):
        feed_dict = {self.train_policy.state:state, self.train_policy.actions:actions,
                     self.train_policy.R:R, self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1]}
        _, l = self.sess.run([self.train_op, self.auxiliary_loss], feed_dict=feed_dict)
        return l

    def a2c_backprop(self, state, R, actions, hidden):
        actions_onehot = one_hot(actions, self.action_size)
        feed_dict = {self.train_policy.state:state, self.train_policy.actions:actions, self.train_policy.R:R}
        _, l = sess.run([self.train_op, self.policy_loss], feed_dict=feed_dict)
        return l
    
    def get_initial_hidden(self, batch_size):
        return self.train_policy.get_initial_hidden(batch_size)
    
    def reset_batch_hidden(self, hidden, idxs):
        return self.train_policy.reset_batch_hidden(hidden, idxs)
    
    def set_session(self, sess):
        self.sess = sess
        self.train_policy.set_session(sess)
        self.validate_policy.set_session(sess)


    class UnrealTrainer(SyncMultiEnvTrainer):
        def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, update_target_freq=10000, num_val_episodes=50):
            SyncMultiEnvTrainer.__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq, update_target_freq=update_target_freq, num_val_episodes=num_val_episodes)

        class Runner(SyncMultiEnvTrainer.Runner):
            def __init__(self, model, env, num_steps, num_envs, sess):
                super().__init__(model, env, num_steps, sess)
                self.prev_hidden = self.model.get_initial_hidden(num_envs)
            
            def run(self):
                memory = []
                for t in range(self.num_steps):
                    policies, values, hidden = self.model.forward(self.states[np.newaxis], self.prev_hidden)
                    actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                    next_states, rewards, dones, infos = self.env.step(actions)
                    TD_target = rewards + self.gamma * values
                    self.model.a2c_backprop(self.states, TD_target, actions, hidden)
                    memory.append((self.states, next_states, actions, rewards, self.prev_hidden, dones, infos))
                    self.states = next_states
                    self.prev_hidden = self.model.reset_batch_hidden(hidden, 1-dones) # reset hidden state at end of episode
                
                
                states, actions, rewards, dones, infos = zip(*memory)
                states, actions, rewards, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(dones)
                return states, actions, rewards, dones, infos, values


def main():

    Atari = True
    env_id = 'SpaceInvaders-v0'
    num_envs = 32
    nsteps = 20

    env = gym.make(env_id)
    
    if Atari:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=1, episodic=False, reset=reset, clip_reward=False) for i in range(10)]
        input_size = val_envs[0].reset().shape
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False , k=1, reset=reset, episodic=True, clip_reward=True)
    
    else:
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(1)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    
    

    train_log_dir = 'logs/Auxiliary/' + env_id + '/'
    model_dir = "models/Auxiliary/" + env_id + '/'


    unreal = UnrealA2C([84,84,1],
                       action_size=6,
                       num_envs=num_envs,
                       PC_coeff=1,
                       RP_coeff=1, 
                       VR_coeff=1,
                       lr=0.001,
                       lr_final=0.001,
                       decay_steps=600000.0,
                       grad_clip=0.5)

    unreal_agent = SyncMultiEnvTrainer( envs=envs, 
                                        model=unreal,
                                        file_loc = [model_dir, train_log_dir],
                                        val_envs = val_envs,
                                        total_steps = 5e6,
                                        nsteps = nsteps,
                                        validate_freq = 1e5,
                                        save_freq = 0,
                                        render_freq = 0,
                                        num_val_episodes = 25)


if __name__ == "__main__":
    main()