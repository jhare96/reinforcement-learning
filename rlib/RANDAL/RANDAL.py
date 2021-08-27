import numpy as np
import torch 
import torch.nn.functional as F
import gym
import time, datetime

from rlib.RND.RND import PPOIntrinsic, PredictorCNN, PredictorMLP, RewardForwardFilter
from rlib.networks.networks import *
from rlib.utils.VecEnv import*
from rlib.utils.wrappers import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import fastsample, fold_batch, tonumpy, totorch, totorch_many, stack_many, fold_many, RunningMeanStd
from rlib.utils.schedulers import polynomial_sheduler

def sign(x):
    if x < 0:
        return 2
    elif x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        raise ValueError

class RANDAL(torch.nn.Module):
    def __init__(self, policy_model, target_model, input_size, action_size, pixel_control=True, intr_coeff=0.5, extr_coeff=1.0, entropy_coeff=0.001, policy_clip=0.1,
                    lr=1e-4, lr_final=1e-5, decay_steps=6e5, grad_clip=0.5, RP=1, VR=1, PC=1, policy_args={}, RND_args={}, optim=torch.optim.Adam, optim_args={}, device='cuda'):
        super(RANDAL, self).__init__()
        self.lr = lr
        self.entropy_coeff = entropy_coeff
        self.intr_coeff = intr_coeff
        self.extr_coeff = extr_coeff
        self.pixel_control = pixel_control
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.device = device
        self.RP = RP # reward prediction 
        self.VR = VR # value replay
        self.PC = PC # pixel control

        
        self.policy = PPOIntrinsic(policy_model, input_size, action_size, lr=lr, lr_final=lr_final, decay_steps=decay_steps, grad_clip=grad_clip,
                            entropy_coeff=entropy_coeff, policy_clip=policy_clip, extr_coeff=extr_coeff, intr_coeff=intr_coeff, build_optimiser=False, **policy_args)

        target_size = (1, input_size[1], input_size[2]) if len(input_size) == 3 else input_size # only use last frame in frame-stack for convolutions
        
        # randomly weighted and fixed neural network, acts as a random_id for each state
        self.target_model = target_model(target_size, trainable=False, **RND_args).to(device)

        # learns to predict target model 
        # i.e. provides rewards based ability to predict a fixed random function, thus behaves as density map of explored areas
        self.predictor_model = target_model(target_size, trainable=True, **RND_args).to(device)
        
        self.optimiser = optim(self.parameters(), lr, **optim_args)
        self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)

        if pixel_control:
            self.feat_map = torch.nn.Sequential(torch.nn.Linear(self.policy.dense_size, 32*8*8), torch.nn.ReLU()).to(device)
            self.deconv1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(32, 32, kernel_size=[3,3], stride=[1,1]), torch.nn.ReLU()).to(device)
            self.deconv_advantage = torch.nn.ConvTranspose2d(32, action_size, kernel_size=[3,3], stride=[2,2]).to(device)
            self.deconv_value = torch.nn.ConvTranspose2d(32, 1, kernel_size=[3,3], stride=[2,2]).to(device)
                
        # reward model
        self.r1 = torch.nn.Sequential(torch.nn.Linear(self.policy.dense_size, 128), torch.nn.ReLU()).to(device)
        self.r2 = torch.nn.Linear(128, 3).to(device)

        self.optimiser = optim(self.parameters(), lr, **optim_args)
        self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)

    def forward(self, state):
        return self.policy.forward(state)

    def evaluate(self, state):
        return self.policy.evaluate(state)

    def Qaux(self, enc_state):
        # Auxillary Q value calculated via dueling network 
        # Z. Wang, N. de Freitas, and M. Lanctot. Dueling Network Architectures for Deep ReinforcementLearning. https://arxiv.org/pdf/1511.06581.pdf
        batch_size = enc_state.shape[0]
        feat_map = self.feat_map(enc_state).view([batch_size,32,8,8])
        deconv1 = self.deconv1(feat_map)
        deconv_adv = self.deconv_advantage(deconv1)
        deconv_value = self.deconv_value(deconv1)
        qaux = deconv_value + deconv_adv - torch.mean(deconv_adv, dim=1, keepdim=True)
        return qaux

    def get_pixel_control(self, state:np.ndarray):
        with torch.no_grad():
            enc_state = self.policy.model(totorch(state, self.device))
            Qaux = self.Qaux(enc_state)
        return tonumpy(Qaux)

    def pixel_loss(self, Qaux, Qaux_actions, Qaux_target):
        'Qaux_target temporal difference target for Q_aux'
        one_hot_actions = F.one_hot(Qaux_actions.long(), self.action_size)
        pixel_action = one_hot_actions.view([-1,self.action_size,1,1])
        Q_aux_action = torch.sum(Qaux * pixel_action, dim=1)
        pixel_loss = 0.5 * torch.mean(torch.square(Qaux_target - Q_aux_action)) # l2 loss for Q_aux over all pixels and batch
        return pixel_loss

    def reward_loss(self, reward_states, reward_target):
        r1 = self.r1(self.policy.model(reward_states))
        pred_reward = self.r2(r1)
        reward_loss = torch.mean(F.cross_entropy(pred_reward, reward_target.long()))  # cross entropy over caterogical reward
        return reward_loss
    
    def replay_loss(self, R, V):
        return torch.mean(torch.square(R - V))
        
    def forward_loss(self, states, actions, Re, Ri, Adv, old_policy):
        states, actions, Re, Ri, Adv, old_policy = totorch_many(states, actions, Re, Ri, Adv, old_policy, device=self.device)
        actions_onehot = F.one_hot(actions.long(), self.action_size)
        policy, Ve, Vi = self.forward(states)
        forward_loss = self.policy.loss(policy, Re, Ri, Ve, Vi, Adv, actions_onehot, old_policy)
        return forward_loss
    
    def auxiliary_loss(self, reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R):
        reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R = totorch_many(reward_states, rewards, Qaux_target,
                                                                                                    Qaux_actions, replay_states, replay_R, device=self.device)
        
        policy_enc = self.policy.model(replay_states)
        replay_values = self.policy.Ve(policy_enc)
        reward_loss = self.reward_loss(reward_states, rewards)
        replay_loss = self.replay_loss(replay_R, replay_values)
        aux_loss = self.RP * reward_loss + self.VR * replay_loss
        
        Qaux_actions = Qaux_actions.long()
        
        if self.pixel_control:
            Qaux = self.Qaux(policy_enc)
            pixel_loss = self.pixel_loss(Qaux, Qaux_actions, Qaux_target)
            aux_loss += self.PC * pixel_loss
        
        return aux_loss
    
    def predictor_loss(self, next_states, state_mean, state_std):
        'loss for predictor network'
        next_states, state_mean, state_std = totorch_many(next_states, state_mean, state_std, device=self.device)
        predictor_loss = self._intr_reward(next_states, state_mean, state_std).mean()
        return predictor_loss

    def _intr_reward(self, next_state, state_mean, state_std):
        norm_next_state = torch.clip((next_state - state_mean) / state_std, -5, 5)
        intr_reward = torch.square(self.predictor_model(norm_next_state) - self.target_model(norm_next_state).detach()).sum(dim=-1)
        return intr_reward 
    
    def intrinsic_reward(self, next_state:np.ndarray, state_mean:np.ndarray, state_std):
        next_state, state_mean, state_std = totorch_many(next_state, state_mean, state_std, device=self.device)
        with torch.no_grad():
            intr_reward = self._intr_reward(next_state, state_mean, state_std)
        return tonumpy(intr_reward)

    def backprop(self, states, next_states, Re, Ri, Adv, actions, old_policy, reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, state_mean, state_std):
        forward_loss = self.forward_loss(states, actions, Re, Ri, Adv, old_policy)
        aux_losses = self.auxiliary_loss(reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R)
        predictor_loss = self.predictor_loss(next_states, state_mean, state_std)

        loss = forward_loss + aux_losses + predictor_loss
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()







class RANDALTrainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, gamma_extr=0.999, gamma_intr=0.99, lambda_=0.95, 
                    init_obs_steps=600, num_epochs=4, num_minibatches=4, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, max_val_steps=10000, replay_length=2000, norm_pixel_reward=True, log_scalars=True):
        
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, gamma=gamma_extr, lambda_=lambda_,
                            validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, max_val_steps=max_val_steps, log_scalars=log_scalars)
        
        self.gamma_intr = gamma_intr
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.pred_prob = 1 / (self.num_envs / 32.0)
        self.state_obs = RunningMeanStd()
        self.forward_filter = RewardForwardFilter(gamma_intr)
        self.intr_rolling = RunningMeanStd()
        self.init_obs_steps = init_obs_steps
        self.replay = deque([], maxlen=replay_length) # replay length per actor
        self.normalise_obs = norm_pixel_reward
        self.replay_length = replay_length

        hyper_paras = {'learning_rate':model.lr, 'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
                        'entropy_coefficient':model.entropy_coeff, 'value_coefficient':1.0, 'intrinsic_value_coefficient':model.intr_coeff,
                        'extrinsic_value_coefficient':model.extr_coeff, 'init_obs_steps':init_obs_steps, 'gamma_intrinsic':self.gamma_intr, 'gamma_extrinsic':self.gamma,
                        'lambda':self.lambda_, 'predictor_dropout_probability':self.pred_prob, 'replay_length':replay_length, 'normalise_pixel_reward':norm_pixel_reward,
                        'replay_value_coefficient':model.VR, 'pixel_control_coefficient':model.PC, 'reward_prediction_coefficient':model.RP
                        }
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)

    def populate_memory(self):
        for t in range(self.replay_length//self.nsteps):
            states, *_ = self.rollout()
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
        ''' normalise pixel intensity changes by recording min and max pixel observations
            not using per pixel normalisation because expected image is singular greyscale frame
        '''
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
        # states of rank [T, B, channels, 84, 84]
        T = len(states) # time length 
        B = states.shape[1] # batch size
        pixel_rewards = np.zeros((T,B,21,21))
        states = states[:,:,-1,:,:]
        prev_state = prev_state[:,-1,:,:]
        if self.normalise_obs:
            states = self.norm_obs(states)
            #print('states, max', states.max(), 'min', states.min(), 'mean', states.mean())
            prev_state = self.norm_obs(prev_state)
            
        pixel_rewards[0] = np.abs(states[0] - prev_state).reshape(-1,4,4,21,21).mean(axis=(1,2))
        for i in range(1,T):
            pixel_rewards[i] = np.abs(states[i] - states[i-1]).reshape(-1,4,4,21,21).mean(axis=(1,2))
        return pixel_rewards

    def sample_replay(self):
        workers = np.random.choice(self.num_envs, replace=False, size=2) # randomly sample from one of n workers
        sample_start = np.random.randint(1, len(self.replay) - self.nsteps -2)
        replay_sample = []
        for i in range(sample_start, sample_start+self.nsteps):
            replay_sample.append(self.replay[i])
                
        replay_states = np.stack([replay_sample[i][0][workers] for i in range(len(replay_sample))])
        replay_actions = np.stack([replay_sample[i][1][workers] for i in range(len(replay_sample))])
        replay_rewards = np.stack([replay_sample[i][2][workers] for i in range(len(replay_sample))])
        replay_values = np.stack([replay_sample[i][3][workers] for i in range(len(replay_sample))])
        replay_dones = np.stack([replay_sample[i][4][workers] for i in range(len(replay_sample))])
        #print('replay dones shape', replay_dones.shape)
        #print('replay_values shape', replay_values.shape)
        
        next_state = self.replay[sample_start+self.nsteps][0][workers] # get state 
        _, replay_last_values_extr, replay_last_values_intr = self.model.evaluate(next_state)
        replay_R = self.GAE(replay_rewards, replay_values, replay_last_values_extr, replay_dones, gamma=0.99, lambda_=0.95) + replay_values

        if self.model.pixel_control:
            prev_states = self.replay[sample_start-1][0][workers]
            Qaux_value = self.model.get_pixel_control(next_state)
            pixel_rewards = self.pixel_rewards(prev_states, replay_states)
            Qaux_target = self.auxiliary_target(pixel_rewards, np.max(Qaux_value, axis=1), replay_dones)
        else:
            Qaux_target = np.zeros((len(replay_states),1,1,1)) # produce fake Qaux to save writing unecessary code
        
        return replay_states, replay_actions, replay_R, Qaux_target, replay_dones
    
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
        reward = np.array([sign(replay_rewards[idx,worker])]) # source of error
    
        return reward_states[None], reward

    
    def init_state_obs(self, num_steps):
        states = 0
        for i in range(num_steps):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            next_states = next_states[:, -1] if len(next_states.shape) == 4 else next_states # [num_envs, channels, height, width] for convolutions, assume frame stack
            states += next_states
        return states / num_steps
    
    
    def _train_nstep(self):
        # stats for normalising states
        self.state_mean, self.state_std = self.state_obs.update(self.init_state_obs(self.init_obs_steps))
        self.state_min, self.state_max = 0.0, 0.0 
        self.populate_memory() # populate experience replay with random actions 
        self.states = self.env.reset() # reset to state s_0

        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps//self.num_minibatches
        start = time.time()
        # main loop
        for t in range(1,num_updates+1):
            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, last_values_extr, last_values_intr, old_policies, dones = self.rollout()
            # update state normalisation statistics
            self.update_minmax(states)
            self.state_mean, self.state_std = self.state_obs.update(next_states) 
            mean, std = self.state_mean, self.state_std

            replay_states, replay_actions, replay_Re, Qaux_target, replay_dones = self.sample_replay() # sample experience replay 

            int_rff = np.array([self.forward_filter.update(intr_rewards[i]) for i in range(len(intr_rewards))]) 
            R_intr_mean, R_intr_std = self.intr_rolling.update(int_rff.ravel()) # normalise intrinsic rewards
            intr_rewards /= R_intr_std
            
            
            Adv_extr = self.GAE(extr_rewards, values_extr, last_values_extr, dones, gamma=self.gamma, lambda_=self.lambda_)
            Adv_intr = self.GAE(intr_rewards, values_intr, last_values_intr, dones, gamma=self.gamma_intr, lambda_=self.lambda_)
            Re = Adv_extr + values_extr
            Ri = Adv_intr + values_intr
            total_Adv = Adv_extr + Adv_intr
            l = 0
            
            # perform minibatch gradient descent for K epochs 
            idxs = np.arange(len(states))
            for epoch in range(self.num_epochs):
                reward_states, sample_rewards = self.sample_reward() # sample reward from replay memory
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch: batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_nextstates, mb_actions, mb_Re, mb_Ri, mb_Adv, mb_old_policies = fold_many(states[batch_idxs], next_states[batch_idxs], \
                                                                                                                 actions[batch_idxs], Re[batch_idxs], Ri[batch_idxs], \
                                                                                                                 total_Adv[batch_idxs], old_policies[batch_idxs])
                    
                    mb_replay_states, mb_replay_actions, mb_replay_Rextr, mb_Qaux_target = fold_many(replay_states[batch_idxs], replay_actions[batch_idxs], \
                                                                                                                        replay_Re[batch_idxs], Qaux_target[batch_idxs])
                    
                    mb_nextstates = mb_nextstates[np.where(np.random.uniform(size=(mini_batch_size)) < self.pred_prob)]
                    # states, next_states, Re, Ri, Adv, actions, old_policy, reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, state_mean, state_std
                    l += self.model.backprop(mb_states.copy(), mb_nextstates.copy(), mb_Re.copy(), mb_Ri.copy(), mb_Adv.copy(), mb_actions.copy(), mb_old_policies.copy(),
                                                reward_states.copy(), sample_rewards.copy(), mb_Qaux_target.copy(), mb_replay_actions.copy(), mb_replay_states.copy(), mb_replay_Rextr.copy(),
                                                mean.copy(), std.copy())
            
            
            l /= self.num_epochs
           
                    
            if self.render_freq > 0 and t % ((self.validate_freq  // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
        
            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.save(s)
                print('saved model')
            
    
    def get_action(self, states):
        policies, values_extr, values_intr = self.model.evaluate(states)
        actions = fastsample(policies)
        return actions

    def rollout(self):
        rollout = []
        for t in range(self.nsteps):
            policies, values_extr, values_intr = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, extr_rewards, dones, infos = self.env.step(actions)
            
            next_states__ = next_states[:, -1:] if len(next_states.shape) == 4 else next_states # [num_envs, channels, height, width] for convolutions 
            intr_rewards = self.model.intrinsic_reward(next_states__, self.state_mean, self.state_std)
            
            rollout.append((self.states, next_states__, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones))
            self.replay.append((self.states, actions, extr_rewards, values_extr, dones)) # add to replay memory
            self.states = next_states

        states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones = stack_many(*zip(*rollout))
        last_policy, last_values_extr, last_values_intr, = self.model.evaluate(self.states)
        return states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, last_values_extr, last_values_intr, policies, dones


def main(env_id):
    num_envs = 32
    nsteps = 128

    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(10)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)
    
    elif 'ApplePicker' in env_id:
        print('ApplePicker')
        make_args = {'num_objects':300, 'default_reward':0}
        if 'Deterministic' in env_id:
            envs = DummyBatchEnv(apple_pickgame, env_id, num_envs, max_steps=5000, auto_reset=True, k=4, grey_scale=True, make_args=make_args)
            val_envs = DummyBatchEnv(apple_pickgame, env_id, num_envs, max_steps=5000, auto_reset=False, k=4, grey_scale=True, make_args=make_args)
            for i in range(len(envs)):
                val_envs.envs[i].set_locs(envs.envs[i].item_locs_master, envs.envs[i].start_loc)
            val_envs.reset()
        else:
        #val_envs = [apple_pickgame(gym.make(env_id), max_steps=5000, auto_reset=False, k=1) for i in range(16)]
            val_envs = DummyBatchEnv(apple_pickgame, env_id, num_envs, max_steps=5000, auto_reset=False, k=4, grey_scale=True)
            envs = DummyBatchEnv(apple_pickgame, env_id, num_envs, max_steps=5000, auto_reset=True, k=4, grey_scale=True)
        print(val_envs.envs[0])
        print(envs.envs[0])

    else:
        print('Atari')
        env = gym.make(env_id)
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        env.close()
        val_envs = BatchEnv(AtariEnv, env_id, num_envs=16, k=4, blocking=False, episodic=False, reset=reset, clip_reward=False, auto_reset=True)
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=4, reset=reset, episodic=False, clip_reward=True)
        
    
    action_size = val_envs.envs[0].action_space.n
    input_size = val_envs.envs[0].reset().shape

    print('action_size', action_size)
    print('input_size', input_size)
    
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/RANDAL/' + env_id + '/Adam/' + current_time
    model_dir = "models/RANDAL/" + env_id + '/' + current_time
    
    
    model = RANDAL(NatureCNN,
                PredictorCNN,
                input_size=input_size,
                action_size=action_size,
                lr=1e-4,
                lr_final=1e-5,
                decay_steps=200e6//(num_envs*nsteps),
                grad_clip=0.5,
                intr_coeff=1.0,
                extr_coeff=2.0,
                entropy_coeff=0.001,
                optim=torch.optim.Adam,
                optim_args={},
                device='cuda'
                )

    
    randal = RANDALTrainer(envs=envs,
                        model=model,
                        model_dir=model_dir,
                        log_dir=train_log_dir,
                        val_envs=val_envs,
                        train_mode='nstep',
                        total_steps=200e6,
                        nsteps=nsteps,
                        init_obs_steps=128*50,
                        num_epochs=4,
                        num_minibatches=4,
                        validate_freq=1e5,
                        save_freq=0,
                        render_freq=0,
                        num_val_episodes=32,
                        log_scalars=False)


    randal.train()


if __name__ == "__main__":
    env_id_list = ['MontezumaRevengeDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'CartPole-v1' , 'Acrobot-v1', ]
    for env_id in env_id_list:
        main(env_id)
