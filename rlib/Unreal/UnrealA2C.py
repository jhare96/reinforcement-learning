import torch
from torch._C import device
import torch.nn.functional as F
import numpy as np
import gym
import os, time
from collections import deque


from rlib.utils.utils import fastsample, fold_batch, one_hot, totorch, tonumpy, totorch_many, stack_many, tonumpy_many
from rlib.A2C.ActorCritic import ActorCritic_LSTM
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*
from rlib.utils.wrappers import*
from rlib.utils.schedulers import polynomial_sheduler

# A2C version of Unsupervised Reinforcement Learning with Auxiliary Tasks (UNREAL) https://arxiv.org/abs/1611.05397

def sign(x):
    if x < 0:
        return 2
    elif x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        raise ValueError

def concat_action_reward(actions, rewards, num_classes):
    concat = one_hot(actions, num_classes)
    concat[:,-1] = rewards   
    return concat  

def AtariEnv__(env, k=4, episodic=True, reset=True, clip_reward=True, Noop=True, time_limit=None, channels_first=True):
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
    if channels_first:
        env = ChannelsFirstEnv(env)
    return env


class Unreal_ActorCritic_LSTM(torch.nn.Module):
    def __init__(self, model, input_size, action_size, cell_size, entropy_coeff=0.01, value_coeff=0.5, lr=1e-4, lr_final=1e-6, decay_steps=50e6, grad_clip=0.5, optim=torch.optim.Adam, optim_args={}, device='cuda', **model_args):
        super(Unreal_ActorCritic_LSTM, self).__init__()
        self.lr = lr
        self.lr_final = lr_final
        self.input_size = input_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.cell_size = cell_size
        self.action_size = action_size
        self.device = device


        self.model = model(input_size, **model_args).to(device)
        self.dense_size = self.model.dense_size
        # self.lstm = MaskedRNN(MaskedLSTMCell(cell_size, self.dense_size+action_size+1), time_major=True)
        self.lstm = MaskedLSTMBlock(self.dense_size+action_size+1, cell_size, time_major=True).to(device)

        self.policy_distrib = torch.nn.Linear(cell_size, action_size).to(device) # Actor
        self.V = torch.nn.Linear(cell_size, 1).to(device) # Critic

        self.optimiser = optim(self.parameters(), lr, **optim_args)
        self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)

    def loss(self, policy, R, V, actions_onehot):
        Advantage = R - V
        value_loss = 0.5 * torch.mean(torch.square(Advantage))

        log_policy = torch.log(torch.clip(policy, 1e-6, 0.999999))
        log_policy_actions = torch.sum(log_policy * actions_onehot, dim=1)
        policy_loss =  torch.mean(-log_policy_actions * Advantage.detach())

        entropy = torch.mean(torch.sum(policy * -log_policy, dim=1))
        loss =  policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        return loss


    def lstm_forward(self, state, action_reward, hidden=None, done=None):
        T, num_envs = state.shape[:2]
        input_size = state.shape[2:]
        enc_state = self.model(state.view((T*num_envs, *input_size)))
        state_prime = torch.cat([enc_state, action_reward], dim=-1)
        unfolded_state = state_prime.view(T, num_envs, -1)
        lstm_outputs, hidden = self.lstm(unfolded_state, hidden, done)
        return lstm_outputs, hidden

    def forward(self, state, action_reward, hidden=None, done=None):
        lstm_outputs, hidden = self.lstm_forward(state, action_reward, hidden, done)
        policy = F.softmax(self.policy_distrib(lstm_outputs), dim=-1).view(-1, self.action_size)
        value = self.V(lstm_outputs).view(-1)
        return policy, value, hidden
    
    def evaluate(self, state:np.ndarray, action_reward:np.ndarray, hidden=None, done=None):
        state, action_reward = totorch_many(state, action_reward, device=self.device)
        hidden = totorch_many(*hidden, device=self.device) if hidden is not None else None
        with torch.no_grad():
            policy, value, hidden = self.forward(state, action_reward, hidden, done)
        return tonumpy(policy), tonumpy(value), tonumpy_many(*hidden)
    
    def backprop(self, state, R, action, action_reward, hidden, done):
        state, R, action, action_reward, done = totorch_many(state, R, action, action_reward, done, device=self.device)
        hidden = totorch_many(*hidden, device=self.device)
        action_onehot = F.one_hot(action.long(), num_classes=self.action_size)
        policy, value, hidden = self.forward(state, action_reward, hidden, done)
        loss = self.loss(policy, R, value, action_onehot)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()
    
    def get_initial_hidden(self, batch_size):
        return np.zeros((1, batch_size, self.cell_size)), np.zeros((1, batch_size, self.cell_size))
    
    def mask_hidden(self, hidden, dones):
        mask = (1-dones).reshape(-1, 1)
        return (hidden[0]*mask, hidden[1]*mask)


class UnrealA2C(torch.nn.Module):
    def __init__(self, policy_model, input_shape, action_size, cell_size, pixel_control=True, RP=1.0, PC=1.0, VR=1.0, entropy_coeff=0.001, value_coeff=0.5,
                    lr=1e-3, lr_final=1e-6, decay_steps=50e6, grad_clip=0.5, policy_args={}, optim=torch.optim.Adam, optim_args={}, device='cuda'):
        super(UnrealA2C, self).__init__()
        self.RP, self.PC, self.VR = RP, PC, VR
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.entropy_coeff, self.value_coeff = entropy_coeff, value_coeff
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.pixel_control = pixel_control
        self.device = device

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        
        self.policy = Unreal_ActorCritic_LSTM(policy_model, input_shape, action_size, cell_size, entropy_coeff=entropy_coeff, value_coeff=value_coeff,
                                        lr=lr, lr_final=lr, decay_steps=decay_steps, build_optimiser=False, grad_clip=grad_clip, device=device, **policy_args)
        
        if pixel_control:
            self.feat_map = torch.nn.Sequential(torch.nn.Linear(self.policy.cell_size, 32*8*8), torch.nn.ReLU()).to(device)
            self.deconv1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(32, 32, kernel_size=[3,3], stride=[1,1]), torch.nn.ReLU()).to(device)
            self.deconv_advantage = torch.nn.ConvTranspose2d(32, action_size, kernel_size=[3,3], stride=[2,2]).to(device)
            self.deconv_value = torch.nn.ConvTranspose2d(32, 1, kernel_size=[3,3], stride=[2,2]).to(device)
                
        # reward model
        self.r1 = torch.nn.Sequential(torch.nn.Linear(self.policy.dense_size*3, 128), torch.nn.ReLU()).to(device)
        self.r2 = torch.nn.Linear(128, 3).to(device)

        self.optimiser = optim(self.parameters(), lr, **optim_args)
        self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)

    def forward(self, state, action_reward, hidden=None, done=None):
        return self.policy.forward(state, action_reward, hidden, done)

    def evaluate(self, state, action_reward, hidden=None, done=None):
        return self.policy.evaluate(state, action_reward, hidden, done)

    def Qaux(self, enc_state):
        # Auxillary Q value calculated via dueling network 
        # Z. Wang, N. de Freitas, and M. Lanctot. Dueling Network Architectures for Deep ReinforcementLearning. https://arxiv.org/pdf/1511.06581.pdf
        batch_size = enc_state.shape[0]
        feat_map = self.feat_map(enc_state).view([batch_size,32,8,8])
        deconv1 = self.deconv1(feat_map)
        deconv_adv = self.deconv_advantage(deconv1)
        deconv_value = self.deconv_value(deconv1)
        qaux = deconv_value + deconv_adv - torch.mean(deconv_adv, dim=3, keepdim=True)
        return qaux

    def get_pixel_control(self, state:np.ndarray, action_reward, hidden):
        state, action_reward, hidden = totorch(state, self.device), totorch(action_reward, self.device), totorch_many(*hidden, device=self.device)
        with torch.no_grad():
            lstm_state, _ = self.policy.lstm_forward(state, action_reward, hidden, done=None)
            Qaux = self.Qaux(lstm_state)
        return tonumpy(Qaux)

    def pixel_loss(self, Qaux, Qaux_actions, Qaux_target):
        # Qaux_target temporal difference target for Q_aux
        one_hot_actions = F.one_hot(Qaux_actions.long(), self.action_size)
        pixel_action = one_hot_actions.view([-1,self.action_size,1,1])
        Q_aux_action = torch.sum(Qaux * pixel_action, dim=1)
        pixel_loss = 0.5 * torch.mean(torch.square(Qaux_target - Q_aux_action)) # l2 loss for Q_aux over all pixels and batch
        return pixel_loss

    def reward_loss(self, reward_states, reward_target):
        cnn_enc_state = self.policy.model(reward_states).view(1, -1)
        r1 = self.r1(cnn_enc_state)
        pred_reward = self.r2(r1)
        reward_loss = torch.mean(F.cross_entropy(pred_reward, reward_target.long()))  # cross entropy over caterogical reward
        return reward_loss
    
    def replay_loss(self, R, V):
        return torch.mean(torch.square(R - V))
        
    def forward_loss(self, states, R, actions, action_rewards, hidden, dones):
        states, R, actions, action_rewards, dones = totorch_many(states, R, actions, action_rewards, dones, device=self.device)
        hidden = totorch_many(*hidden, device=self.device)
        actions_onehot = F.one_hot(actions.long(), num_classes=self.action_size)
        policies, values, _ = self.forward(states, action_rewards, hidden, dones)
        forward_loss = self.policy.loss(policies, R, values, actions_onehot)
        return forward_loss
    
    def auxiliary_loss(self, reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, replay_hidden, replay_dones, replay_actsrews):
        reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, replay_dones, replay_actsrews = totorch_many(reward_states, rewards, Qaux_target, Qaux_actions,
                                                                                                                        replay_states, replay_R, replay_dones, replay_actsrews, device=self.device)
        replay_hidden = totorch_many(*replay_hidden, device=self.device)
        lstm_state, _ = self.policy.lstm_forward(replay_states, replay_actsrews, replay_hidden, replay_dones)
        replay_values = self.policy.V(lstm_state)
        
        reward_loss = self.reward_loss(reward_states, rewards)
        replay_loss = self.replay_loss(replay_R, replay_values)
        aux_loss = self.RP * reward_loss + self.VR * replay_loss
        
        if self.pixel_control:
            Qaux = self.Qaux(lstm_state)
            pixel_loss = self.pixel_loss(Qaux, Qaux_actions, Qaux_target)
            aux_loss += self.PC * pixel_loss
        
        return aux_loss

    def backprop(self, states, R, actions, hidden, dones, action_rewards,
                reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, replay_hidden, replay_dones, replay_actsrews):
        forward_loss = self.forward_loss(states, R, actions, action_rewards, hidden, dones)

        aux_losses = self.auxiliary_loss(reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, replay_hidden, replay_dones, replay_actsrews)

        loss = forward_loss + aux_losses
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()
    

    def get_initial_hidden(self, batch_size):
        return self.policy.get_initial_hidden(batch_size)
    
    def mask_hidden(self, hidden, dones):
        return self.policy.mask_hidden(hidden, dones)


class Unreal_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, validate_freq=1000000, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars)
        
        self.replay = deque([], maxlen=2000)
        self.action_size = self.model.action_size
        self.prev_hidden = self.model.get_initial_hidden(len(self.env))
        zeros = np.zeros((len(self.env)), dtype=np.int32)
        self.prev_actions_rewards = concat_action_reward(zeros, zeros, self.action_size+1) # start with action 0 and reward 0

        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':self.num_envs,
                  'total_steps':self.total_steps, 'entropy_coefficient':model.entropy_coeff, 'value_coefficient':0.5}
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def populate_memory(self):
        for t in range(2000//self.nsteps):
            self.rollout()
    
    def auxiliary_target(self, prev_state, states, values, dones):
        T = len(states)
        #print('values shape', values.shape)
        R = np.zeros((T,*values.shape))
        dones = dones[:,None,None]
        pixel_rewards = np.zeros_like(R)
        pixel_rewards[0] = np.abs((states[0]/255) - (prev_state/255)).reshape(-1,21,4,21,4,3).mean(axis=(2,4,5))
        for i in range(1,T):
            pixel_rewards[i] = np.abs((states[i]/255) - (states[i-1]/255)).reshape(-1,21,4,21,4,3).mean(axis=(2,4,5))
        #print('pixel reward, max', pixel_rewards.max(), 'mean', pixel_rewards.mean())
        
        R[-1] = values * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = pixel_rewards[i] + 0.99 * R[i+1] * (1-dones[-1])
        
        return R

    def sample_replay(self):
        sample_start = np.random.randint(0, len(self.replay) -self.nsteps -2)
        worker = np.random.randint(0,self.num_envs) # randomly sample from one of n workers
        if self.replay[sample_start][5][worker] == True:
            sample_start += 2
        replay_sample = []
        for i in range(sample_start, sample_start+self.nsteps):
            replay_sample.append(self.replay[i])
            if self.replay[sample_start][5][worker] == True:
                break
                
        replay_states = np.stack([replay_sample[i][0][worker] for i in range(len(replay_sample))])
        replay_actions = np.stack([replay_sample[i][1][worker] for i in range(len(replay_sample))])
        replay_rewards = np.stack([replay_sample[i][2][worker] for i in range(len(replay_sample))])
        replay_hiddens = np.stack([replay_sample[i][3] for i in range(len(replay_sample))])[:,:,:,worker,:]
        replay_actsrews = np.stack([replay_sample[i][4][worker] for i in range(len(replay_sample))])
        replay_dones = np.stack([replay_sample[i][5][worker] for i in range(len(replay_sample))])

        next_state = self.replay[sample_start+self.nsteps][0][worker][None] # get state 
        _, replay_values, *_ = self.model.evaluate(next_state[None], replay_actsrews[-1][None], replay_hiddens[-1].reshape(2,1,1,-1))
        replay_R = self.nstep_return(replay_rewards, replay_values, replay_dones)

        prev_states = self.replay[sample_start-1][0][worker]
        Qaux_value = self.model.get_pixel_control(next_state[None], replay_actsrews[-1][None], replay_hiddens[-1].reshape(2,1,1,-1))[0]
        Qaux_target = self.auxiliary_target(prev_states, replay_states, np.max(Qaux_value, axis=0), replay_dones)
        replay_hidden = (replay_hiddens[0][:1], replay_hiddens[0][1:])

        return replay_states[:,None], replay_actions, replay_R, Qaux_target, \
                     replay_hidden, replay_actsrews, replay_dones
    
    def sample_reward(self):
        # worker = np.random.randint(0,self.num_envs) # randomly sample from one of n workers
        
        replay_rewards = np.array([self.replay[i][2] for i in range(len(self.replay))])[3:]
        worker = np.argmax(np.sum(replay_rewards, axis=0)) # sample experience from best worker
        nonzero_idxs = np.where(np.abs(replay_rewards) > 0)[0] # idxs where |reward| > 0 
        zero_idxs = np.where(replay_rewards == 0)[0] # idxs where reward == 0 
        
        
        if len(nonzero_idxs) ==0 or len(zero_idxs) == 0: # if nonzero or zero idxs do not exist i.e. all rewards same sign 
            idx = np.random.randint(len(replay_rewards))
        elif np.random.uniform() > 0.5: # sample from zero and nonzero rewards equally
            idx = np.random.choice(nonzero_idxs)
        else:
            idx = np.random.choice(zero_idxs)
        
        
        reward_states = np.stack([self.replay[i][0][worker] for i in range(idx-3,idx)])
        reward = np.array([sign(replay_rewards[idx,worker])])
        return reward_states, reward
    
    def _train_nstep(self):
        batch_size = (self.num_envs * self.nsteps)
        start = time.time()
        num_updates = self.total_steps // batch_size
        s = 0
        self.populate_memory()

        # main loop
        for t in range(1,num_updates+1):
            states, actions, rewards, hidden_init, prev_acts_rewards, dones, last_values = self.rollout()

            R = self.nstep_return(rewards, last_values, dones, clip=False)
            # stack all states, actions and Rs across all workers into a single batch
            prev_acts_rewards, actions, rewards, R = fold_batch(prev_acts_rewards), fold_batch(actions), fold_batch(rewards), fold_batch(R)
            
            reward_states, sample_rewards = self.sample_reward()
            replay_states, replay_actions, replay_R, Qaux_target, replay_hidden, replay_actsrews, replay_dones = self.sample_replay()
                        
            l = self.model.backprop(states, R, actions, hidden_init, dones, prev_acts_rewards,
                reward_states, sample_rewards, Qaux_target, replay_actions, replay_states, replay_R, replay_hidden, replay_dones, replay_actsrews)
            
            if self.render_freq > 0 and t % ((self.validate_freq // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq //batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.saver.save(s)
                print('saved model')


    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps, replay):
            super().__init__(model, env, num_steps)
            self.replay = replay
            self.action_size = self.model.action_size
            

    def rollout(self,):
        rollout = []
        first_hidden = self.prev_hidden
        for t in range(self.nsteps):
            policies, values, hidden = self.model.evaluate(self.states[None], self.prev_actions_rewards, self.prev_hidden)
            #Qaux = self.model.get_pixel_control(self.states, self.prev_hidden, self.prev_actions_rewards[None])
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)

            rollout.append((self.states, actions, rewards, self.prev_actions_rewards, dones, infos))
            self.replay.append((self.states, actions, rewards, self.prev_hidden, self.prev_actions_rewards, dones)) # add to replay memory
            self.states = next_states
            self.prev_hidden = self.model.mask_hidden(hidden, dones) # reset hidden state at end of episode
            self.prev_actions_rewards = concat_action_reward(actions, rewards, self.action_size+1)
        
        states, actions, rewards, prev_actions_rewards, dones, infos = stack_many(*zip(*rollout))
        _, last_values, _ = self.model.evaluate(self.states[None], self.prev_actions_rewards, self.prev_hidden)
        return states, actions, rewards, first_hidden, prev_actions_rewards, dones, last_values

    def get_action(self, state):
        policy, value = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        #action = np.argmax(policy)
        return action

    def validate(self,env,num_ep,max_steps,render=False):
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            hidden = self.model.get_initial_hidden(1)
            prev_actrew = concat_action_reward(np.zeros((1),dtype=np.int32), np.zeros((1),dtype=np.int32), self.model.action_size+1)
            for t in range(max_steps):
                policy, value, hidden = self.model.evaluate(state[None, None], prev_actrew, hidden)
                #print('policy', policy, 'value', value)
                action = np.random.choice(policy.shape[1], p=policy[0])
                next_state, reward, done, info = env.step(action)
                state = next_state
                prev_actrew = concat_action_reward(np.array([action]), np.array([reward]), self.model.action_size+1)
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

def main(env_id):
    num_envs = 32
    nsteps = 128

    env = gym.make(env_id)
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(16)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    elif 'ApplePicker' in env_id:
        print('ApplePicker')
        make_args = {'num_objects':100, 'default_reward':-0.01}
        val_envs = [apple_pickgame(gym.make(env_id, **make_args), max_steps=5000, auto_reset=True) for i in range(10)]
        envs = DummyBatchEnv(apple_pickgame, env_id, num_envs, max_steps=5000, auto_reset=True, make_args=make_args)
        print(val_envs[0])
        print(envs.envs[0])

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv__(gym.make(env_id), k=1, episodic=False, reset=reset, clip_reward=False) for i in range(10)]
        envs = BatchEnv(AtariEnv__, env_id, num_envs, blocking=False, k=1, reset=reset, episodic=True, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/UnrealA2C/' + env_id + '/'
    model_dir = "models/UnrealA2C/" + env_id + '/'



    model = UnrealA2C(NatureCNN,
                      input_shape=input_size,
                      action_size=action_size,
                      cell_size=256,
                      PC=0.01,
                      entropy_coeff=0.001,
                      lr=1e-3,
                      lr_final=1e-6,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={'dense_size':256},
                      device='cuda')
    

    auxiliary = Unreal_Trainer(envs=envs,
                                  model=model,
                                  model_dir=model_dir,
                                  log_dir=train_log_dir,
                                  val_envs=val_envs,
                                  train_mode='nstep',
                                  total_steps=50e6,
                                  nsteps=nsteps,
                                  validate_freq=1e6,
                                  save_freq=0,
                                  render_freq=0,
                                  num_val_episodes=10,
                                  log_scalars=False)

    
    auxiliary.train()
    del auxiliary


if __name__ == "__main__":
    import apple_picker
    #env_id_list = ['SpaceInvadersDeterministic-v4', 'PrivateEyeDeterministic-v4', 'FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'PongDeterministic-v4' ]
    #env_id_list = ['MountainCar-v0','CartPole-v1']
    env_id_list = ['ApplePicker-v0']
    for env_id in env_id_list:
        main(env_id)
    