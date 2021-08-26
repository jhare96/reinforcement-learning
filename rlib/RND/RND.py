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

class RollingObs(object):
    def __init__(self, mean=0):
        self.rolling = RunningMeanStd()
    
    def update(self, x):
        if len(x.shape) == 4: # assume image obs 
            return self.rolling.update(np.mean(x, axis=1, keepdims=True)) #[time*batch,height,width,stack] -> [height, width]
        else:
            return self.rolling.update(x) #[time*batch,*shape] -> [*shape]


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
    def __init__(self, model, input_shape, action_size, lr=1e-3, lr_final=0, decay_steps=6e5, grad_clip=0.5,
                    entropy_coeff=0.01, policy_clip=0.1, extr_coeff=2.0, intr_coeff=1.0,
                    build_optimiser=True, optim=torch.optim.Adam, optim_args={}, device='cuda', **model_args):
        super(PPOIntrinsic, self).__init__()
        self.action_size = action_size
        
        self.lr = lr
        self.lr_final = lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        
        self.entropy_coeff = entropy_coeff
        self.policy_clip = policy_clip
        self.extr_coeff = extr_coeff
        self.intr_coeff = intr_coeff

        self.device = device

        self.model = model(input_shape, **model_args).to(self.device)
        dense_size = self.model.dense_size
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
        self.h1 = torch.nn.Sequential(torch.nn.Conv2d(input_size[0], conv1_size, kernel_size=[8,8], stride=[4,4], padding=padding, requires_grad=trainable), torch.nn.LeakyReLU())
        self.h2 = torch.nn.Sequential(torch.nn.Conv2d(conv1_size, conv2_size, kernel_size=[4,4], stride=[2,2], padding=padding, requires_grad=trainable), torch.nn.LeakyReLU())
        self.h3 = torch.nn.Sequential(torch.nn.Conv2d(conv2_size, conv3_size, kernel_size=[3,3], stride=[1,1], padding=padding, requires_grad=trainable), torch.nn.LeakyReLU())
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
            self.dense = torch.nn.Linear(h*w*c, dense_size, require_grad=False)
        
        self.init_weights()
    
    def init_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal(module.weight, gain=self.init_scale)

    def _conv_outsize(self):
        _, h, w = self.input_shape
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
        self.input_shape = input_size
        self.init_scale = init_scale
        layers = []
        in_size = input_size
        for l in range(num_layers):
            layers.append(torch.nn.Linear(in_size, dense_size, requires_grad=trainable))
            layers.append(activation())
            in_size = dense_size
        layers.append(torch.nn.Linear(dense_size, dense_size, requires_grad=trainable))
        self.layers = torch.nn.ModuleList(layers)
        
        self.init_weights()
    
    def init_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal(module.weight, gain=self.init_scale)

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
        
        target_size = (1, input_size[1:]) if len(input_size) == 3 else input_size # only use last frame in frame-stack for convolutions

        self.policy = PPOIntrinsic(policy_model, input_size, action_size, lr, lr_final, decay_steps, grad_clip,
                                    entropy_coeff=entropy_coeff, policy_clip=policy_clip, extr_coeff=extr_coeff, intr_coeff=intr_coeff, device=device, build_optimiser=False, **policy_args)
        
        # randomly weighted and fixed neural network, acts as a random_id for each state
        self.target_model = target_model(target_size, trainable=False)

        # learns to predict target model 
        # i.e. provides rewards based ability to predict a fixed random function, thus behaves as density map of explored areas
        self.predictor_model = target_model(target_size, trainable=True)   
        
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


class PPO_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, gamma=0.99, lambda_=0.95, 
                    num_epochs=4, num_minibatches=4, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, gamma=gamma, lambda_=lambda_,
                            validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars)

        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.pred_prob = 1 / (self.num_envs / 32.0)
        self.state_obs = RollingObs()

        hyper_paras = {'learning_rate':model.lr, 'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
                        'entropy_coefficient':0.001, 'value_coefficient':1.0, 'intrinsic_value_coefficient':model.intr_coeff,
                        'extrinsic_value_coefficient':model.extr_coeff, 'init_obs_steps':init_obs_steps}
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def init_state_obs(self, num_steps):
        states = 0
        for i in range(num_steps):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            states += next_states
        return states / num_steps
    
    
    def _train_nstep(self):
        # stats for normalising states
        self.state_mean, self.state_std = self.state_obs.update(self.init_state_obs(10000//self.num_envs))
        self.states = self.env.reset() # reset to state s_0

        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps//self.num_minibatches
        start = time.time()
        # main loop
        for t in range(1,num_updates+1):
            states, next_states, actions, rewards, values_extr, values_intr, last_values_extr, last_values_intr, old_policies, dones = self.rollout()
            mean, std = self.state_mean, self.state_std
            
            Adv_extr = self.GAE(rewards, values_extr, last_values_extr, dones, gamma=self.gamma, lambda_=self.lambda_)
            Adv_intr = self.GAE(rewards, values_intr, last_values_intr, dones, gamma=self.gamma, lambda_=self.lambda_)
            Re = Adv_extr + values_extr
            Ri = Adv_intr + values_intr
            total_Adv = Adv_extr + Adv_intr
            l = 0
            
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
            
            
            l /= (self.num_epochs*self.num_minibatches)
           
                    
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
            
            next_states = next_states[:, -1] if len(next_states.shape) == 4 else next_states # [batch, channels, height, width] for convolutions 
            intr_rewards = self.model.intrinsic_reward(next_states, self.state_mean, self.state_std)
            
            rollout.append((self.states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones))
            self.states = next_states

        states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones = stack_many(*zip(*rollout))
        last_policy, last_values_extr, last_values_intr, = self.model.evaluate(next_states)
        return states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, last_values_extr, last_values_intr, policies, dones
    

    def validation_summary(self,t,loss,start,render):
        batch_size = self.num_envs * self.nsteps
        tot_steps = t * batch_size
        time_taken = time.time() - start
        frames_per_update = (self.validate_freq // batch_size) * batch_size
        fps = frames_per_update / time_taken 
        num_val_envs = len(self.val_envs)
        num_val_eps = [self.num_val_episodes//num_val_envs for i in range(num_val_envs)]
        num_val_eps[-1] = num_val_eps[-1] + self.num_val_episodes % self.num_val_episodes//(num_val_envs)
        render_array = np.zeros((len(self.val_envs)))
        render_array[0] = render
        
        score = np.mean(self.validate(self.val_envs, self.num_val_episodes, self.val_steps, render=False))
        print("update %i, validation score %f, total steps %i, loss %f, time taken for %i frames:%fs, fps %f \t\t\t" %(t,score,tot_steps,loss,frames_per_update,time_taken,fps))
        
        if self.log_scalars:
            self.train_writer.add_scalar('validation/score', score, tot_steps)
            self.train_writer.add_scalar('train/loss', loss, tot_steps)
    

    def validate(self, env, num_ep, max_steps, render=False):
        episode_scores = []
        for episode in range(num_ep//len(env)):
            states = env.reset()
            episode_score = []
            for t in range(max_steps):
                action = self.get_action(states)
                next_states, rewards, dones, infos = env.step(action)
                states = next_states
                #print('state', state, 'action', action, 'reward', reward)

                episode_score.append(rewards*(1-dones))
                
                if render:
                    with self.lock:
                        env.render()

                if dones.sum() == self.num_envs or t == max_steps -1:
                    tot_reward = np.sum(np.stack(episode_score), axis=0)
                    episode_scores.append(tot_reward)
        
        return episode_scores


class RNDTrainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, init_obs_steps=128*50, num_epochs=4, num_minibatches=4, validate_freq=1000000.0,
                 save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True, gpu_growth=True):
        
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars,
                            gpu_growth=gpu_growth)


        self.runner = self.Runner(self.model, self.env, self.nsteps)
        self.alpha = 1
        self.pred_prob = 1 / (self.num_envs / 32.0)
        self.lambda_ = 0.95
        self.init_obs_steps = init_obs_steps
        self.num_epochs, self.num_minibatches = num_epochs, num_minibatches
        hyper_paras = {'learning_rate':model.lr,
         'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
          'entropy_coefficient':0.001, 'value_coefficient':0.5, 'intr_coeff':model.intr_coeff,
        'extr_coeff':model.extr_coeff, 'init_obs_steps':init_obs_steps}
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def validate(self,env,num_ep,max_steps,render=False):
        episode_scores = []
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            for t in range(max_steps):
                policy, value_extr, value_intr = self.model.forward(state[np.newaxis])
                #print('policy', policy, 'value_extr', value_extr)
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

    
    def init_state_obs(self, num_steps):
        states = 0
        for i in range(1, num_steps+1):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            states += next_states
        mean = states / num_steps
        self.runner.state_mean, self.runner.state_std = self.state_rolling.update(mean.mean(axis=0)[None,None])

    
    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        rolling = RunningMeanStd(shape=())
        obs = self.runner.states[0]
        obs = obs[...,-1:] if len(obs.shape) == 3 else obs
        self.state_rolling = rolling_obs(shape=obs.shape)
        self.init_state_obs(self.init_obs_steps)
        self.runner.states = self.env.reset()
        forward_filter = RewardForwardFilter(0.99)

        # main loop
        start = time.time()
        for t in range(1,num_updates+1):
            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, old_policies, dones = self.runner.run()
            policy, extr_last_values, intr_last_values = self.model.forward(next_states[-1])

            self.runner.state_mean, self.runner.state_std = self.state_rolling.update(next_states) # update state normalisation statistics 

            int_rff = np.array([forward_filter.update(intr_rewards[i]) for i in range(len(intr_rewards))]) 
            R_intr_mean, R_intr_std = rolling.update(int_rff.ravel()) # normalise intr reward
            intr_rewards /= R_intr_std


            Adv_extr = self.GAE(extr_rewards, values_extr, extr_last_values, dones, gamma=0.999, lambda_=self.lambda_)
            Adv_intr = self.GAE(intr_rewards, values_intr, intr_last_values, np.zeros_like(dones), gamma=0.99, lambda_=self.lambda_) # non episodic intr reward signal 
            R_extr = Adv_extr + values_extr
            R_intr = Adv_intr + values_intr
            total_Adv = self.model.extr_coeff * Adv_extr + self.model.intr_coeff * Adv_intr

            # perform minibatch gradient descent for K epochs 
            l = 0
            idxs = np.arange(len(states))
            for epoch in range(self.num_epochs):
                mini_batch_size = self.nsteps//self.num_minibatches
                np.random.shuffle(idxs)
                for batch in range(0,len(states), mini_batch_size):
                    batch_idxs = idxs[batch:batch + mini_batch_size]
                    # stack all states, next_states, actions and Rs across all workers into a single batch
                    mb_states, mb_nextstates, mb_actions, mb_Rextr, mb_Rintr, mb_Adv, mb_old_policies = fold_batch(states[batch_idxs]), fold_batch(next_states[batch_idxs]), \
                                                    fold_batch(actions[batch_idxs]), fold_batch(R_extr[batch_idxs]), fold_batch(R_intr[batch_idxs]), \
                                                    fold_batch(total_Adv[batch_idxs]), fold_batch(old_policies[batch_idxs])
                
                    mb_nextstates = mb_nextstates[np.where(np.random.uniform(size=(mini_batch_size)) < self.pred_prob)]
                    mb_nextstates = mb_nextstates[...,-1:] if len(mb_nextstates.shape) == 4 else mb_nextstates
                    
                    mean, std = self.runner.state_mean, self.runner.state_std
                    l += self.model.backprop(mb_states, mb_nextstates, mb_Rextr, mb_Rintr, mb_Adv, mb_actions, mb_old_policies, mean, std)
            
            l /= (self.num_epochs * self.num_minibatches)
        
            if self.render_freq > 0 and t % (self.validate_freq // batch_size * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + '/' + str(s) + ".ckpt") )
                print('saved model')
            
    
    def get_action(self, state):
        policy, value = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        return action

    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps):
            super().__init__(model, env, num_steps)
            self.state_mean = None
            self.state_std = None
        
        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values_extr, values_intr = self.model.forward(self.states)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)
    
                next_states__ = next_states[...,-1:] if len(next_states.shape) == 4 else next_states
                intr_rewards = self.model.intrinsic_reward(next_states__, self.state_mean, self.state_std)
                
                rollout.append((self.states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones))
                self.states = next_states

            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones = stack_many(*zip(*rollout))
            return states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones
    
    


def main(env_id, Atari=True):
    num_envs = 64
    nsteps = 128
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(10)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)
    
    elif 'ApplePicker' in env_id:
        print('ApplePicker')
        make_args = {'num_objects':100, 'default_reward':-0.1}
        val_envs = [gym.make(env_id, **make_args) for i in range(10)]
        envs = DummyBatchEnv(apple_pickgame, env_id, num_envs, max_steps=10000, auto_reset=True, make_args=make_args)
        print(val_envs[0])
        print(envs.envs[0])

    else:
        env = gym.make(env_id)
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=84, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)
        env.close()
    
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/RND/' + env_id + '/' + current_time
    model_dir = "models/RND/" + env_id + '/' + current_time

    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
   
    ac_mlp_args = {'dense_size':64}

    #with tf.device('GPU:3'):
    model = RND(nature_cnn,
                predictor_cnn,
                input_shape=input_size,
                action_size=action_size,
                intr_coeff=1.0,
                extr_coeff=2.0,
                value_coeff=0.5,
                entropy_coeff=0.001,
                lr=1e-4,
                grad_clip=0.5,
                policy_args={},
                RND_args={}) #

    

    curiosity = RND_Trainer(envs=envs,
                            model=model,
                            model_dir=model_dir,
                            log_dir=train_log_dir,
                            val_envs=val_envs,
                            train_mode='nstep',
                            total_steps=50e6,
                            nsteps=nsteps,
                            init_obs_steps=128*50,
                            num_epochs=4,
                            num_minibatches=4,
                            validate_freq=1e6,
                            save_freq=0,
                            render_freq=0,
                            num_val_episodes=50,
                            log_scalars=True,
                            gpu_growth=True)
    curiosity.train()
    
    del curiosity

    tf.reset_default_graph()


if __name__ == "__main__":
    #env_id_list = ['MontezumaRevengeDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'CartPole-v1' , 'Acrobot-v1', ]
    env_id_list = ['ApplePickerDeterministic-v0']
    # for i in range(2):
    for env_id in env_id_list:
        main(env_id)
    