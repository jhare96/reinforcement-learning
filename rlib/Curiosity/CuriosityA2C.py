import torch
import torch.nn.functional as F
import numpy as np
import scipy
import gym
import os, time
import threading
from rlib.A2C.A2C import ActorCritic
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*
from rlib.utils.wrappers import*
from rlib.utils.utils import fastsample, fold_batch, one_hot, RunningMeanStd, normalise, stack_many, totorch_many
from rlib.utils.schedulers import polynomial_sheduler

class RollingObs(object):
    def __init__(self, mean=0):
        self.rolling = RunningMeanStd()
    
    def update(self, x):
        if len(x.shape) == 4: # assume image obs 
            return self.rolling.update(np.mean(x, axis=1, keepdims=True)) #[time*batch,height,width,stack] -> [height, width]
        else:
            return self.rolling.update(x) #[time*batch,*shape] -> [*shape]


class ICM(torch.nn.Module):
    def __init__(self, model_head, input_size, action_size, forward_coeff, device='cuda', **model_head_args):
        super(ICM, self).__init__()
        self.action_size = action_size
        self.forward_coeff = forward_coeff
        self.phi = model_head(input_size, **model_head_args)
        dense_size = self.phi.dense_size
        self.device = device

        # forward model 
        self.forward1 = torch.nn.Sequential(torch.nn.Linear(dense_size + action_size, dense_size), torch.nn.ReLU()).to(device)
        self.pred_state = torch.nn.Linear(dense_size, dense_size).to(device)

        # inverse model
        self.inverse1 = torch.nn.Sequential(torch.nn.Linear(dense_size*2, dense_size), torch.nn.ReLU()).to(device)
        self.pred_action = torch.nn.Sequential(torch.nn.Linear(dense_size*2, dense_size), torch.nn.ReLU()).to(device)

    
    def intr_reward(self, phi, action_onehot, phi_next):
        f1 = self.forward1(torch.cat([phi, action_onehot], dim=1))
        phi_pred = self.pred_state(f1)
        intr_reward = 0.5 * torch.sum(torch.square(phi_pred - phi_next), dim=1) # l2 distance metric ‖ˆφ(st+1)−φ(st+1)‖22
        return intr_reward
    
    def predict_action(self, phi1, phi2):
        phi_cat = torch.cat([phi1, phi2], dim=1)
        pred_action = self.pred_action(phi_cat)
        return pred_action

    def get_intr_reward(self, state, action, next_state):
        state, next_state, action = totorch_many(state, next_state, action, device=self.device)
        action = action.long()
        phi1 = self.phi(state)
        phi2 = self.phi(next_state)
        action_onehot = F.one_hot(action, self.action_size)
        with torch.no_grad():
            intr_reward = self.intr_reward(phi1, action_onehot, phi2)
        return intr_reward.cpu().numpy()

    def get_pred_action(self, state, next_state):
        state, next_state = totorch_many(state, next_state, device=self.device)
        return self.pred_action(state, next_state)

    def loss(self, state, action, next_state):
        action = action.long()
        phi1 = self.phi(state)
        phi2 = self.phi(next_state)
        action_onehot = F.one_hot(action, self.action_size)

        forward_loss = torch.mean(self.intr_reward(phi1, action_onehot, phi2))
        inverse_loss = F.cross_entropy(self.predict_action(phi1, phi2), action)
        return (1-self.forward_coeff) * inverse_loss + self.forward_coeff * forward_loss
        

class Curiosity(torch.nn.Module):
    def __init__(self,  policy_model, ICM_model, input_size, action_size, forward_coeff, policy_importance, reward_scale, entropy_coeff, value_coeff=0.5,
                    lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip=0.5, policy_args={}, ICM_args={}, device='cuda'):
        super(Curiosity, self).__init__()
        self.reward_scale, self.forward_coeff, self.policy_importance, self.entropy_coeff = reward_scale, forward_coeff, policy_importance, entropy_coeff
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.device = device

        try:
            iterator = iter(input_size)
        except TypeError:
            input_size = (input_size,)
        
        self.ICM = ICM(ICM_model, input_size, action_size, forward_coeff, device=device, **ICM_args)
        self.AC = ActorCritic(policy_model, input_size, action_size, entropy_coeff, value_coeff, lr, lr_final, decay_steps, grad_clip, build_optimiser=False, device=device, **policy_args)
        
        self.optimiser = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)

    def forward(self, state):
        return self.AC.forward(state)
    
    def evaluate(self, state):
        return self.AC.evaluate(state)
    
    def intrinsic_reward(self, state, action, next_state):
        return self.ICM.get_intr_reward(state, action, next_state)
    
    def backprop(self, state, next_state, R, Adv, action, state_mean, state_std):
        state, next_state, R, Adv, action, state_mean, state_std = totorch_many(state, next_state, R, Adv,
                                                                        action, state_mean, state_std, device=self.device)
        policy, value = self.AC.forward(state)
        action_onehot = F.one_hot(action.long(), self.action_size)
        policy_loss = self.AC.loss(policy, R, value, action_onehot)
        ICM_loss = self.ICM.loss((state-state_mean)/state_std, action, (next_state-state_mean)/state_std)
        loss = self.policy_importance * policy_loss + self.reward_scale * ICM_loss
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()
    



class Curiosity_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', total_steps=1000000, nsteps=5, validate_freq=1000000, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        super().__init__(envs, model, val_envs, train_mode=train_mode, return_type='nstep', log_dir=log_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes,log_scalars=log_scalars)

        self.state_obs = RollingObs()
        self.state_mean = None
        self.state_std = None
        
        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps,
         'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
          'entropy_coefficient':0.01, 'value_coefficient':0.5, 'reward_scale':model.reward_scale,
          'forward_model_scale':model.forward_coeff, 'policy_importance':model.policy_importance}
    
        if self.log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
        
        self.lambda_ = 0.95
    
    def init_state_obs(self, num_steps):
        states = 0
        for i in range(num_steps):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            states += next_states
        return states / num_steps
    
    
    def _train_nstep(self):
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        self.state_mean, self.state_std = self.state_obs.update(self.init_state_obs(10000//self.num_envs))
        self.states = self.env.reset()
        print(self.state_mean.shape, self.state_std.shape)
        start = time.time()
        # main loop
        batch_size = self.num_envs * self.nsteps
        for t in range(1,num_updates+1):
            states, next_states, actions, rewards, dones, values = self.rollout()
            _, last_values = self.model.evaluate(next_states[-1])

            R = self.nstep_return(rewards, last_values, dones)
            Adv = R - values
            #delta = rewards + self.gamma * values[:-1] - values[1:]
            #Adv = self.multistep_target(delta, values[-1], dones, gamma=self.gamma*self.lambda_)
                
            # stack all states, next_states, actions and Rs across all workers into a single batch
            states, next_states, actions, R, Adv = fold_batch(states), fold_batch(next_states), fold_batch(actions), fold_batch(R), fold_batch(Adv)
            mean, std = self.state_mean, self.state_std
            
            l = self.model.backprop(states, next_states, R, Adv, actions, mean, std)
            
            # self.state_mean, self.state_std = self.state_obs.update(states)
            
            if self.render_freq > 0 and t % (self.validate_freq * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + self.current_time + '/' + str(s) + ".ckpt") )
                print('saved model')
            

    
    def get_action(self, state):
        policy, value = self.model.evaluate(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        return action
            

    def rollout(self,):
        rollout = []
        for t in range(self.nsteps):
            start = time.time()
            policies, values = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, extr_rewards, dones, infos = self.env.step(actions)
            
            mean, std = self.state_mean[None], self.state_std[None]
            intr_rewards = self.model.intrinsic_reward((self.states-mean)/std, actions, (next_states-mean)/std)
            rewards = extr_rewards + intr_rewards
            rollout.append((self.states, next_states, actions, rewards, values, dones))
            self.states = next_states
        
        states, next_states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        return states, next_states, actions, rewards, dones, values
            

def main(env_id):
    num_envs = 32
    nsteps = 20
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(1)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    else:
        env = gym.make(env_id)
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(1)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=84, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/Curiosity/' + env_id + '/hyper_unclipped/'

    model = Curiosity(NatureCNN,
                      NatureCNN,
                      input_size=input_size,
                      action_size=action_size,
                      forward_coeff=0.2,
                      policy_importance=1,
                      reward_scale=1.0,
                      entropy_coeff=0.01,
                      #intr_coeff=1,
                      lr=1e-3,
                      lr_final=0,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={},
                      ICM_args={'scale':False}).cuda() 

    

    curiosity = Curiosity_Trainer(envs=envs,
                                  model=model,
                                  val_envs=val_envs,
                                  train_mode='nstep',
                                  total_steps=5e6,
                                  nsteps=nsteps,
                                  validate_freq=1e5,
                                  save_freq=0,
                                  render_freq=0,
                                  num_val_episodes=1,
                                  log_dir=train_log_dir,
                                  log_scalars=False)
    print(env_id)
    curiosity.train()

    del curiosity

if __name__ == "__main__":
    env_id_list = ['SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'PongDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1', ]
    #for i in range(5):
    for env_id in env_id_list:
        main(env_id)
    