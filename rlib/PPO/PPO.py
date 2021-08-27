import torch
import torch.nn.functional as F
import numpy as np
import time, datetime
import gym
import copy
import matplotlib.pyplot as plt

from rlib.networks.networks import *
from rlib.utils.VecEnv import*
from rlib.utils.wrappers import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import fastsample, fold_batch, tonumpy, totorch, totorch_many, stack_many, fold_many
from rlib.utils.schedulers import polynomial_sheduler

class PPO(torch.nn.Module):
    def __init__(self, model, input_shape, action_size, lr=1e-3, lr_final=0, decay_steps=6e5, grad_clip=0.5, value_coeff=1.0, entropy_coeff=0.01, policy_clip=0.1,
                    build_optimiser=True, optim=torch.optim.Adam, optim_args={}, device='cuda', **model_args):
        super(PPO, self).__init__()
        self.lr = lr
        self.lr_final = lr_final
        self.action_size = action_size
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.policy_clip = policy_clip
        self.device = device

        self.model = model(input_shape, **model_args).to(self.device)
        dense_size = self.model.dense_size
        self.policy = torch.nn.Sequential(torch.nn.Linear(dense_size, action_size), torch.nn.Softmax(dim=-1)).to(self.device)
        self.V = torch.nn.Linear(dense_size, 1).to(self.device)

        if build_optimiser:
            self.optimiser = optim(self.parameters(), lr, **optim_args)
            self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)
        
    
    def forward(self, state):
        state_enc = self.model(state)
        policy = self.policy(state_enc)
        value = self.V(state_enc).view(-1)
        return policy, value
    
    def evaluate(self, state):
        with torch.no_grad():
            policy, value = self.forward(totorch(state, self.device))
        return tonumpy(policy), tonumpy(value)

    
    def loss(self, policy, R, V, Adv, action_onehot, old_policy):
        value_loss = 0.5 * torch.mean(torch.square(R - V))

        policy_actions = torch.sum(policy * action_onehot, dim=1)
        old_policy_actions = torch.sum(old_policy * action_onehot, dim=1)
        ratio = policy_actions / old_policy_actions
        policy_loss_unclipped = ratio * -Adv
        policy_loss_clipped = torch.clip_(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * -Adv
        policy_loss = torch.mean(torch.maximum(policy_loss_unclipped, policy_loss_clipped))
        entropy = torch.mean(torch.sum(policy * -torch.log(policy), dim=1))

        loss =  policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        return loss

    def backprop(self, state, R, Adv, action, old_policy):
        state, action, R, Adv, old_policy = totorch_many(state, action, R, Adv, old_policy, device=self.device)
        action_onehot = F.one_hot(action.long(), self.action_size)
        policy, value = self.forward(state)
        loss = self.loss(policy, R, value, Adv, action_onehot, old_policy)

        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()


class PPOTrainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, gamma=0.99, lambda_=0.95, 
                    num_epochs=4, num_minibatches=4, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, max_val_steps=10000, log_scalars=True):
        
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, gamma=gamma, lambda_=lambda_,
                            validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, max_val_steps=max_val_steps, log_scalars=log_scalars)

        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches

        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps,
            'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
            'entropy_coefficient':self.model.entropy_coeff, 'value_coefficient':self.model.value_coeff, 'gamma':self.gamma, 'lambda':self.lambda_}
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    
    
    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        s = 0
        mini_batch_size = self.nsteps//self.num_minibatches
        start = time.time()
        # main loop
        for t in range(1,num_updates+1):
            #rollout_start = time.time()
            states, actions, rewards, values, last_values, old_policies, dones = self.rollout()
            #print('rollout time', time.time()-rollout_start)
            Adv = self.GAE(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_)
            R = Adv + values
            l = 0
            
            #backprop_time = time.time()
            idxs = np.arange(len(states))
            for epoch in range(self.num_epochs):
                np.random.shuffle(idxs)
                for batch in range(0, len(states), mini_batch_size):
                    batch_idxs = idxs[batch: batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_actions, mb_R, mb_Adv, mb_old_policies = fold_many(states[batch_idxs], actions[batch_idxs], 
                                                                                     R[batch_idxs], Adv[batch_idxs],
                                                                                     old_policies[batch_idxs])

                    l += self.model.backprop(mb_states.copy(), mb_R.copy(), mb_Adv.copy(), mb_actions.copy(), mb_old_policies.copy())
            
            #print('backprop time', time.time()-backprop_time)
            l /= self.num_epochs
           
                    
            if self.render_freq > 0 and t % ((self.validate_freq  // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
        
            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                #val_time = time.time()
                self.validation_summary(t,l,start,render)
                #print('validation time', time.time()-val_time)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.save(s)
                print('saved model')
            
    
    def get_action(self, states):
        policies, values = self.model.evaluate(states)
        actions = fastsample(policies)
        return actions

    def rollout(self):
        rollout = []
        for t in range(self.nsteps):
            policies, values = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, policies, dones))
            self.states = next_states

        states, actions, rewards, values, policies, dones = stack_many(*zip(*rollout))
        policy, last_values, = self.model.evaluate(next_states)
        return states, actions, rewards, values, last_values, policies, dones


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
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=4, reset=reset, episodic=False, clip_reward=True)
        
    
    action_size = val_envs.envs[0].action_space.n
    input_size = val_envs.envs[0].reset().shape
    
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/PPO/' + env_id + '/Adam/' + current_time
    model_dir = "models/PPO/" + env_id + '/' + current_time
    
    
    model = PPO(UniverseCNN,
                input_shape=input_size,
                action_size=action_size,
                lr=1e-4,
                lr_final=1e-5,
                decay_steps=200e6//(num_envs*nsteps),
                grad_clip=0.5,
                value_coeff=1.0,
                entropy_coeff=0.01,
                device='cuda'
                ).cuda()

    
    ppo = PPOTrainer(envs=envs,
                            model=model,
                            model_dir=model_dir,
                            log_dir=train_log_dir,
                            val_envs=val_envs,
                            train_mode='nstep',
                            total_steps=200e6,
                            nsteps=nsteps,
                            num_epochs=2,
                            num_minibatches=4,
                            validate_freq=1e5,
                            save_freq=0,
                            render_freq=0,
                            num_val_episodes=32,
                            log_scalars=False)
    ppo.train()
    

if __name__ == "__main__":
    import apple_picker
    #env_id_list = ['SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4']# 'SpaceInvadersDeterministic-v4',]# , ]
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1', ]
    env_id_list = ['ApplePickerDeterministic-v0']
    for env_id in env_id_list:
        main(env_id)
            