import torch
import gym
import numpy as np
import time, datetime

from rlib.networks.networks import*
from rlib.utils.VecEnv import*
from rlib.utils.wrappers import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import stack_many, totorch, fastsample
from rlib.A2C.ActorCritic import ActorCritic

class A2C(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', return_type='nstep', log_dir='logs/A2C', model_dir='models/A2C', total_steps=10000, nsteps=5, gamma=0.99, lambda_=0.95,
                 validate_freq=1e6, save_freq=0, render_freq=0, num_val_episodes=50, max_val_steps=10000, log_scalars=True):
        
        super().__init__(envs, model, val_envs, log_dir=log_dir, model_dir=model_dir, train_mode=train_mode, return_type=return_type, total_steps=total_steps, nsteps=nsteps,
         gamma=gamma, lambda_=lambda_, validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq,
         num_val_episodes=num_val_episodes, max_val_steps=max_val_steps, log_scalars=log_scalars)

        hyperparas = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':self.num_envs,
                  'total_steps':total_steps, 'entropy_coefficient':model.entropy_coeff, 'value_coefficient':model.value_coeff , 'return type':self.return_type, 'gamma':self.gamma, 'lambda':self.lambda_}
        
        if log_scalars:
            filename = log_dir + '/' + 'hyperparameters.txt'
            self.save_hyperparameters(filename , **hyperparas)

    def get_action(self, state):
        policy, value = self.model.evaluate(state)
        action = int(fastsample(policy))
        return action
    
    def rollout(self,):
        rollout = []
        for t in range(self.nsteps):
            policies, values = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, dones))
            self.states = next_states
        
        states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        _, last_values = self.model.evaluate(next_states)
        return states, actions, rewards, dones, values, last_values
        
    def _train_onestep(self):
        states = self.env.reset()
        y = np.zeros((self.num_envs))
        num_steps = self.total_steps // self.num_envs
        for t in range(1,num_steps+1):
            policies, values = self.model.evaluate(self.states)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            y = rewards + self.gamma * self.model.get_value(next_states) * (1-dones)
            
            l = self.model.backprop(states, y, actions)
            states = next_states
            
            if self.render_freq > 0 and t % ((self.validate_freq // self.num_envs) * self.render_freq) == 0:
                render = True
            else:
                render = False

            if self.validate_freq > 0 and t % (self.validate_freq // self.num_envs) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // self.num_envs) == 0: 
                self.s += 1
                self.save(self.s)
                print('saved model')


       
def main(env_id):
    num_envs = 32
    nsteps = 20
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    train_log_dir = 'logs/A2C/' + env_id +'/GAE/' + current_time 
    model_dir = "models/A2C/" + env_id + '/GAE/' + current_time 
    
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(10)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)
    
    elif 'ApplePicker' in env_id:
        print('ApplePicker')
        make_args = {'num_objects':100, 'default_reward':-0.1}
        val_envs = [gym.make(env_id, **make_args) for i in range(10)]
        envs = DummyBatchEnv(apple_pickgame, env_id, num_envs, max_steps=5000, auto_reset=True, make_args=make_args)
        print(val_envs[0])
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
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    print('input shape', input_size)
    print('action space', action_size)


    
    model = ActorCritic(UniverseCNN,
                        input_size,
                        action_size,
                        lr=1e-3,
                        lr_final=1e-4,
                        entropy_coeff=0.01,
                        decay_steps=50e6//(num_envs*nsteps),
                        grad_clip=0.5)
    

    a2c = A2C(envs=envs,
              model=model,
              model_dir=model_dir,
              log_dir=train_log_dir,
              val_envs=val_envs,
              train_mode='nstep',
              return_type='GAE',
              total_steps=50e6,
              nsteps=nsteps,
              validate_freq=1e5,
              save_freq=0,
              render_freq=0,
              num_val_episodes=50,
              log_scalars=False)

    a2c.train()

    del a2c

    # a2c = A2C.load(A2C, model, envs, val_envs, model_dir + time + '/1.trainer')
    # a2c.train()


if __name__ == "__main__":
    env_id_list = ['SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'PongDeterministic-v4']
    #env_id_list = ['MontezumaRevengeDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1', ]
    #env_id_list = ['ApplePicker-v0']
    for env_id in env_id_list:
        main(env_id)
