import numpy as np
import scipy
import gym
import os, time, datetime
import threading
from rlib.A2C.ActorCritic import ActorCritic_LSTM
from rlib.networks.networks import*
from rlib.utils.utils import fold_batch, stack_many, totorch, fastsample
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*
from rlib.utils.wrappers import*



class A2CLSTM_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', return_type='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=20,
                validate_freq=1e6, save_freq=0, render_freq=0, num_val_episodes=50, max_val_steps=10000, log_scalars=True):
        
        super().__init__(envs, model, val_envs, log_dir=log_dir, model_dir=model_dir, train_mode=train_mode, return_type=return_type,
                        total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq, save_freq=save_freq,
                        render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, max_val_steps=max_val_steps, log_scalars=log_scalars)
        
        
        self.prev_hidden = self.model.get_initial_hidden(self.num_envs)
        
        hyper_params = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs,
                  'total_steps':self.total_steps, 'entropy_coefficient':model.entropy_coeff, 'value_coefficient':model.value_coeff, 'gamma':self.gamma, 'lambda':self.lambda_}
        
        if self.log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_params)
    
    def _train_nstep(self):
        batch_size = (self.num_envs * self.nsteps)
        start = time.time()
        num_updates = self.total_steps // batch_size
        s = 0
        # main loop
        for t in range(1,num_updates+1):
            states, actions, rewards, first_hidden, dones, values, last_values = self.rollout()
            
            if self.return_type == 'nstep':
                R = self.nstep_return(rewards, last_values, dones, gamma=self.gamma)
            elif self.return_type == 'GAE':
                R = self.GAE(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_) + values
            elif self.return_type == 'lambda':
                R = self.lambda_return(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_)
                
            # stack all states, actions and Rs across all workers into a single batch
            actions, R = fold_batch(actions), fold_batch(R)
            l = self.model.backprop(states, R, actions, first_hidden, dones)

            if self.render_freq > 0 and t % ((self.validate_freq // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq //batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir  + str(s) + ".ckpt") )
                print('saved model')
    
    
    def _validate_async(self, env, num_ep, max_steps, render=False):
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            hidden = self.model.get_initial_hidden(1)
            for t in range(max_steps):
                policy, value, hidden = self.model.evaluate(state[None, None], hidden)
                #print('policy', policy, 'value', value)
                action = int(fastsample(policy))
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
    
    def validate_sync(self, render):
        episode_scores = []
        env = self.val_envs
        for episode in range(self.num_val_episodes//len(env)):
            states = env.reset()
            episode_score = []
            prev_hidden = self.model.get_initial_hidden(len(self.val_envs))
            for t in range(self.val_steps):
                policies, values, hidden = self.model.evaluate(states[None], prev_hidden)
                actions = fastsample(policies)
                next_states, rewards, dones, infos = env.step(actions)
                states = next_states

                episode_score.append(rewards*(1-dones))
                
                if render:
                    with self.lock:
                        env.render()

                if dones.sum() == self.num_envs or t == self.val_steps -1:
                    tot_reward = np.sum(np.stack(episode_score), axis=0)
                    episode_scores.append(tot_reward)
                    break
        
        return np.mean(episode_scores)
            
        
    def rollout(self,):
        rollout = []
        first_hidden = self.prev_hidden
        for t in range(self.nsteps):
            policies, values, hidden = self.model.evaluate(self.states[None], self.prev_hidden)
            actions = fastsample(policies)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, values, dones))
            self.states = next_states
            self.prev_hidden = self.model.mask_hidden(hidden, dones) # reset hidden state at end of episode
            
        states, actions, rewards, values, dones = stack_many(*zip(*rollout))
        _, last_values, _ = self.model.evaluate(self.states[None], self.prev_hidden)
        return states, actions, rewards, first_hidden, dones, values, last_values
            

def main(env_id):
    num_envs = 32
    nsteps = 20
    
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
        val_envs = [AtariEnv(gym.make(env_id), k=1, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, rescale=84, blocking=False , k=1, reset=reset, episodic=False, clip_reward=True)
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    

    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/A2C_LSTM/' + env_id +'/' + current_time
    model_dir = "models/A2C_LSTM/" + env_id + '/' + current_time
    

    model = ActorCritic_LSTM(NatureCNN,
                        input_size=input_size,
                        action_size=action_size,
                        cell_size=256,
                        lr=1e-3,
                        lr_final=1e-4,
                        decay_steps=50e6//(num_envs*nsteps),
                        grad_clip=0.5,
                        optim=torch.optim.RMSprop,
                        device='cuda')

    
    a2c_trainer = A2CLSTM_Trainer(envs=envs,
                                  model=model,
                                  model_dir=model_dir,
                                  log_dir=train_log_dir,
                                  val_envs=val_envs,
                                  train_mode='nstep',
                                  return_type='GAE',
                                  total_steps=50e6,
                                  nsteps=nsteps,
                                  validate_freq=1e6,
                                  save_freq=0,
                                  render_freq=0,
                                  num_val_episodes=25,
                                  log_scalars=False)
    print(env_id)
    
    a2c_trainer.train()

    del model

if __name__ == "__main__":
    env_id_list = ['SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'PongDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1']
    #env_id_list = ['SuperMarioBros-1-1-v0']
    for env_id in env_id_list:
        main(env_id)