import numpy as np
import gym
import time, datetime
import torch
import torch.nn.functional as F
from rlib.utils.utils import fold_batch, one_hot, Welfords_algorithm, stack_many, RunningMeanStd, tonumpy_many

from rlib.networks.networks import *
from rlib.utils.VecEnv import*
from rlib.utils.wrappers import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import fastsample, fold_batch, tonumpy, totorch, totorch_many, stack_many, fold_many
from rlib.utils.schedulers import polynomial_sheduler


class RewardForwardFilter(object):
    # https://github.com/openai/random-network-distillation
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma
    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems      



class PPOIntrinsic(torch.nn.Module):
    def __init__(self, model, input_size, action_size, lr=1e-3, lr_final=0, decay_steps=6e5, grad_clip=0.5,
                    entropy_coeff=0.01, policy_clip=0.1, extr_coeff=2.0, intr_coeff=1.0,
                    build_optimiser=True, optim=torch.optim.Adam, optim_args={}, device='cuda', **model_args):
        super(PPOIntrinsic, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        
        self.lr = lr
        self.lr_final = lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        
        self.entropy_coeff = entropy_coeff
        self.policy_clip = policy_clip
        self.extr_coeff = extr_coeff
        self.intr_coeff = intr_coeff

        self.device = device

        self.model = model(input_size, **model_args).to(self.device)
        self.dense_size = dense_size = self.model.dense_size
        self.policy = torch.nn.Sequential(torch.nn.Linear(dense_size, action_size), torch.nn.Softmax(dim=-1)).to(self.device) # Actor
        self.Ve = torch.nn.Linear(dense_size, 1).to(self.device) # Critic (Extrinsic)
        self.Vi = torch.nn.Linear(dense_size, 1).to(self.device) # Intrinsic Value i.e. expected instrinsic value of state 

        if build_optimiser:
            self.optimiser = optim(self.parameters(), lr, **optim_args)
            self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)
        
    
    def forward(self, state):
        state_enc = self.model(state)
        policy = self.policy(state_enc)
        value_extr = self.Ve(state_enc).view(-1)
        value_intr = self.Ve(state_enc).view(-1)
        return policy, value_extr, value_intr
    
    def evaluate(self, state):
        with torch.no_grad():
            policy, value_extr, value_intr = self.forward(totorch(state, self.device))
        return tonumpy(policy), tonumpy(value_extr), tonumpy(value_intr)

    
    def loss(self, policy, Re, Ri, Ve, Vi, Adv, action_onehot, old_policy):
        extr_value_loss = 0.5 * torch.mean(torch.square(Re - Ve))
        intr_value_loss = 0.5 * torch.mean(torch.square(Ri - Vi))

        policy_actions = torch.sum(policy * action_onehot, dim=1)
        old_policy_actions = torch.sum(old_policy * action_onehot, dim=1)
        ratio = policy_actions / old_policy_actions
        policy_loss_unclipped = ratio * -Adv
        policy_loss_clipped = torch.clip_(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * -Adv
        policy_loss = torch.mean(torch.maximum(policy_loss_unclipped, policy_loss_clipped))
        entropy = torch.mean(torch.sum(policy * -torch.log(policy), dim=1))

        value_loss = self.extr_coeff * extr_value_loss + self.intr_coeff * intr_value_loss
        loss =  policy_loss + value_loss - self.entropy_coeff * entropy
        return loss

    def backprop(self, state, Re, Ri, Adv, action, old_policy):
        state, action, Re, Ri, Adv, old_policy = totorch_many(state, action, Re, Ri, Adv, old_policy, device=self.device)
        action_onehot = F.one_hot(action.long(), self.action_size)
        policy, Ve, Vi = self.forward(state)
        loss = self.loss(policy, Re, Ri, Ve, Vi, Adv, action_onehot, old_policy)

        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()


class PredictorCNN(torch.nn.Module):
    def __init__(self, input_size, conv1_size=32, conv2_size=64, conv3_size=64, dense_size=512, padding=[0,0], init_scale=np.sqrt(2), scale=True, trainable=True):
        # input_shape [channels, height, width]
        super(PredictorCNN, self).__init__()
        self.scale = scale
        self.dense_size = dense_size
        self.input_size = input_size
        self.init_scale = init_scale
        self.h1 = torch.nn.Sequential(torch.nn.Conv2d(input_size[0], conv1_size, kernel_size=[8,8], stride=[4,4], padding=padding), torch.nn.LeakyReLU())
        self.h2 = torch.nn.Sequential(torch.nn.Conv2d(conv1_size, conv2_size, kernel_size=[4,4], stride=[2,2], padding=padding), torch.nn.LeakyReLU())
        self.h3 = torch.nn.Sequential(torch.nn.Conv2d(conv2_size, conv3_size, kernel_size=[3,3], stride=[1,1], padding=padding), torch.nn.LeakyReLU())
        self.flatten = torch.nn.Flatten()
        c, h, w = self._conv_outsize()
        in_size = h*w*c
        if trainable:
            self.dense = torch.nn.Sequential(
                            torch.nn.Linear(h*w*c, dense_size), torch.nn.ReLU(),
                            torch.nn.Linear(dense_size, dense_size), torch.nn.ReLU(),
                            torch.nn.Linear(dense_size, dense_size)
            )
        else:
            self.dense = torch.nn.Linear(h*w*c, dense_size)
        
        self.init_weights()
        self.set_trainable(trainable)
    
    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def init_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=self.init_scale)

    def _conv_outsize(self):
        _, h, w = self.input_size
        h, w = conv2d_outsize(h, w, self.h1[0].kernel_size, self.h1[0].stride, self.h1[0].padding)
        h, w = conv2d_outsize(h, w, self.h2[0].kernel_size, self.h2[0].stride, self.h2[0].padding)
        h, w = conv2d_outsize(h, w, self.h3[0].kernel_size, self.h3[0].stride, self.h3[0].padding)
        return self.h3[0].out_channels, h, w

    def forward(self, x):
        x = x/255 if self.scale else x
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class PredictorMLP(torch.nn.Module):
    def __init__(self, input_size, num_layers=2, dense_size=64, activation=torch.nn.LeakyReLU, init_scale=np.sqrt(2), trainable=True):
        # input_shape = feature_size
        super(PredictorMLP, self).__init__()
        self.dense_size = dense_size
        self.input_size = input_size
        self.init_scale = init_scale
        layers = []
        in_size = input_size
        for l in range(num_layers):
            layers.append(torch.nn.Linear(in_size, dense_size))
            layers.append(activation())
            in_size = dense_size
        layers.append(torch.nn.Linear(dense_size, dense_size))
        self.layers = torch.nn.ModuleList(layers)
        
        self.init_weights()
        self.set_trainable(trainable)
    
    def set_trainable(self, trainable):
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
    
    def init_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=self.init_scale)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class RND(torch.nn.Module):
    # EXPLORATION BY RANDOM NETWORK DISTILLATION
    # https://arxiv.org/pdf/1810.12894.pdf
    def __init__(self, policy_model, target_model, input_size, action_size, entropy_coeff=0.001,
                 intr_coeff=0.5, extr_coeff=1.0, lr=1e-4, lr_final=0, decay_steps=1e5, grad_clip=0.5, policy_clip=0.1, policy_args={}, RND_args={}, optim=torch.optim.Adam, optim_args={}, device='cuda'):
        super(RND, self).__init__()
        self.intr_coeff = intr_coeff
        self.extr_coeff = extr_coeff
        self.entropy_coeff = entropy_coeff
        self.lr = lr
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.device = device
        
        target_size = (1, input_size[1], input_size[2]) if len(input_size) == 3 else input_size # only use last frame in frame-stack for convolutions

        self.policy = PPOIntrinsic(policy_model, input_size, action_size, lr, lr_final, decay_steps, grad_clip,
                                    entropy_coeff=entropy_coeff, policy_clip=policy_clip, extr_coeff=extr_coeff, intr_coeff=intr_coeff, device=device, build_optimiser=False, **policy_args)
        
        # randomly weighted and fixed neural network, acts as a random_id for each state
        self.target_model = target_model(target_size, trainable=False).to(device)

        # learns to predict target model 
        # i.e. provides rewards based ability to predict a fixed random function, thus behaves as density map of explored areas
        self.predictor_model = target_model(target_size, trainable=True).to(device)
        
        self.optimiser = optim(self.parameters(), lr, **optim_args)
        self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)


    def forward(self, state):
        return self.policy.forward(state)
    
    def evaluate(self, state):
        return self.policy.evaluate(state)
    
    def _intr_reward(self, next_state, state_mean, state_std):
        norm_next_state = torch.clip((next_state - state_mean) / state_std, -5, 5)
        intr_reward = torch.square(self.predictor_model(norm_next_state) - self.target_model(norm_next_state).detach()).sum(dim=-1)
        return intr_reward 
    
    def intrinsic_reward(self, next_state:np.ndarray, state_mean:np.ndarray, state_std):
        next_state, state_mean, state_std = totorch_many(next_state, state_mean, state_std, device=self.device)
        with torch.no_grad():
            intr_reward = self._intr_reward(next_state, state_mean, state_std)
        return tonumpy(intr_reward)
    

    def backprop(self, state, next_state, R_extr, R_intr, Adv, actions, old_policy, state_mean, state_std):
        state, next_state, R_extr, R_intr, Adv, actions, old_policy, state_mean, state_std = totorch_many(state, next_state, R_extr, R_intr,
                                                                                                          Adv, actions, old_policy, state_mean, state_std, device=self.device)
        policy, Ve, Vi = self.policy.forward(state)
        actions_onehot = F.one_hot(actions.long(), self.action_size)
        policy_loss = self.policy.loss(policy, R_extr, R_intr, Ve, Vi, Adv, actions_onehot, old_policy)
        
        predictor_loss = self._intr_reward(next_state, state_mean, state_std).mean()
        loss = policy_loss + predictor_loss
        
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()


class RNDTrainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, gamma_extr=0.999, gamma_intr=0.99, lambda_=0.95, 
                    init_obs_steps=600, num_epochs=4, num_minibatches=4, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, max_val_steps=10000, log_scalars=True):
        
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

        hyper_paras = {'learning_rate':model.lr, 'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
                        'entropy_coefficient':0.001, 'value_coefficient':1.0, 'intrinsic_value_coefficient':model.intr_coeff,
                        'extrinsic_value_coefficient':model.extr_coeff, 'init_obs_steps':init_obs_steps, 'gamma_intrinsic':self.gamma_intr, 'gamma_extrinsic':self.gamma,
                        'lambda':self.lambda_, 'predictor_dropout_probability':self.pred_prob
                        }
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
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
        self.states = self.env.reset() # reset to state s_0

        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps//self.num_minibatches
        start = time.time()
        # main loop
        for t in range(1,num_updates+1):
            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, last_values_extr, last_values_intr, old_policies, dones = self.rollout()
            self.state_mean, self.state_std = self.state_obs.update(next_states) # update state normalisation statistics
            mean, std = self.state_mean, self.state_std

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
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch: batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_nextstates, mb_actions, mb_Re, mb_Ri, mb_Adv, mb_old_policies = fold_many(states[batch_idxs], next_states[batch_idxs], \
                                                                                                                 actions[batch_idxs], Re[batch_idxs], Ri[batch_idxs], \
                                                                                                                 total_Adv[batch_idxs], old_policies[batch_idxs])
                    
                    mb_nextstates = mb_nextstates[np.where(np.random.uniform(size=(mini_batch_size)) < self.pred_prob)]
                    l += self.model.backprop(mb_states.copy(), mb_nextstates.copy(), mb_Re.copy(), mb_Ri.copy(), mb_Adv.copy(), mb_actions.copy(), mb_old_policies.copy(), mean.copy(), std.copy())
            
            
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
    train_log_dir = 'logs/RND/' + env_id + '/Adam/' + current_time
    model_dir = "models/RND/" + env_id + '/' + current_time
    
    
    model = RND(NatureCNN,
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

    
    rnd = RNDTrainer(envs=envs,
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
    rnd.train()


if __name__ == "__main__":
    env_id_list = ['MontezumaRevengeDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'CartPole-v1' , 'Acrobot-v1', ]
    for env_id in env_id_list:
        main(env_id)
    