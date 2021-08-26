import torch
import torch.nn.functional as F
import numpy as np 
import gym
import threading
import time, datetime
from collections import OrderedDict

from rlib.networks.networks import*
from rlib.utils.wrappers import*
from rlib.utils.VecEnv import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import one_hot, fold_batch, unfold_batch, totorch, tonumpy, totorch_many
from rlib.utils.schedulers import polynomial_sheduler


main_lock = threading.Lock()

def save_hyperparameters(filename, **kwargs):
    handle = open(filename, "w")
    for key, value in kwargs.items():
        handle.write("{} = {}\n" .format(key, value))
    handle.close()



class DQN(torch.nn.Module):
    def __init__(self, model, input_shape, action_size, lr=1e-3, lr_final=0, decay_steps=50e6, grad_clip=0.5, optim=torch.optim.Adam, optim_args={}, device='cuda', **model_args):
        super(DQN, self).__init__()
        self.lr = lr
        self.lr_final = lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.device = device

        self.model = model(input_shape, **model_args).to(self.device)
        self.Q = torch.nn.Linear(self.model.dense_size, action_size).to(self.device)
        
        self.optimiser = optim(self.parameters(), lr, **optim_args)
        self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)
    
    def loss(self, Qsa, R, action_onehot):
        Qvalue = torch.sum(Qsa * action_onehot, dim=1)
        loss = torch.mean(torch.square(R - Qvalue))
        return loss

    def backprop(self, state:np.ndarray, R:np.ndarray, action:np.ndarray):
        state, R, action = totorch_many(state, R, action, device=self.device)
        action_onehot = F.one_hot(action.long(), num_classes=self.action_size)
        Qsa = self.forward(state)
        loss = self.loss(Qsa, R, action_onehot)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()
    
    def forward(self, state):
        Qsa = self.Q(self.model(state))
        return Qsa

    def evaluate(self, state):
        with torch.no_grad():
            Qsa = self.forward(totorch(state, self.device))
        return Qsa.cpu().numpy()



class SyncDDQN(SyncMultiEnvTrainer):
    def __init__(self, envs, model, target_model, val_envs, action_size, log_dir='logs/SyncDDQN/', model_dir='models/SyncDDQN/',
                     train_mode='nstep', return_type='nstep', total_steps=1000000, nsteps=5, gamma=0.99, lambda_=0.95,
                     validate_freq=1e6, save_freq=0, render_freq=0, update_target_freq=10000, num_val_episodes=50, log_scalars=True,
                     epsilon_start=1, epsilon_final=0.01, epsilon_steps = 1e6, epsilon_test=0.01):

        
        super().__init__(envs=envs, model=model, val_envs=val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, return_type=return_type, total_steps=total_steps,
                nsteps=nsteps, gamma=gamma, lambda_=lambda_, validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq,
                update_target_freq=update_target_freq, num_val_episodes=num_val_episodes, log_scalars=log_scalars)
        
        self.target_model = self.TargetQ = target_model
        self.Q = self.model # more readable alias
        self.epsilon = np.array([epsilon_start], dtype=np.float64)
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.schedule = self.linear_schedule(self.epsilon , epsilon_final, epsilon_steps//self.num_envs)
        self.epsilon_test = np.array(epsilon_test, dtype=np.float64)

        self.action_size = action_size

        hyper_paras = {'learning_rate':self.model.lr, 'learning_rate_final':self.model.lr_final, 'lr_decay_steps':self.model.decay_steps , 'grad_clip':self.model.grad_clip,
         'nsteps':self.nsteps, 'num_workers':self.num_envs, 'return type':self.return_type, 'total_steps':self.total_steps, 'gamma':gamma, 'lambda':lambda_,
         'epsilon_start':self.epsilon, 'epsilon_final':self.epsilon_final, 'epsilon_steps':self.epsilon_steps, 'update_freq':update_target_freq}
        
        hyper_paras = OrderedDict(hyper_paras)

        if self.log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    
    class linear_schedule(object):
        def __init__(self, epsilon, epsilon_final, num_steps=1000000):
            self._counter = 0
            self._epsilon = epsilon
            self._epsilon_final = epsilon_final
            self._step = (epsilon - epsilon_final) / num_steps
            self._num_steps = num_steps
        
        def step(self,):
            if self._counter < self._num_steps :
                self._epsilon -= self._step
                self._counter += 1
            else:
                self._epsilon[:] = self._epsilon_final
        
        def get_epsilon(self,):
            return self._epsilon
    
    def get_action(self, state):
        if np.random.uniform() < self.epsilon_test:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.model.evaluate(state))
        return action

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    
    def local_attr(self, attr):
        attr['update_target_freq'] = self.target_freq
        return attr
        
    def rollout(self):
        rollout = []
        for t in range(self.nsteps):
            Qsa = self.Q.evaluate(self.states)
            actions = np.argmax(Qsa, axis=1)
            random = np.random.uniform(size=(self.num_envs))
            random_actions = np.random.randint(self.action_size, size=(self.num_envs))
            actions = np.where(random < self.epsilon, random_actions, actions)
            next_states, rewards, dones, infos = self.env.step(actions)
            rollout.append((self.states, actions, rewards, dones, infos))
            self.states = next_states
            self.schedule.step()
        
        states, actions, rewards, dones, infos = zip(*rollout)
        states, actions, rewards, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(dones)
        TargetQsa = unfold_batch(self.TargetQ.evaluate(fold_batch(states)), self.num_steps, self.num_envs) # Q(s,a; theta-1)
        values = np.sum(TargetQsa * one_hot(actions, self.action_size), axis=-1) # Q(s, argmax_a Q(s,a; theta); theta-1)
        
        last_actions = np.argmax(self.Q.evaluate(next_states), axis=1)
        last_TargetQsa = self.TargetQ.evaluate(next_states) # Q(s,a; theta-1)
        last_values = np.sum(last_TargetQsa * one_hot(last_actions, self.action_size), axis=-1) # Q(s, argmax_a Q(s,a; theta); theta-1)
        return states, actions, rewards, dones, values, last_values


def stackFireReset(env):
    return StackEnv(FireResetEnv(env))


def main(env_id):
    num_envs = 32
    nsteps = 128

    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/SyncDDQN/' + env_id + '/n-step/RMSprop/' + current_time
    model_dir = "models/SyncDDQN/" + env_id + '/' + current_time

    env = gym.make(env_id)
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(16)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    elif 'ApplePicker' in env_id:
        print('ApplePicker')
        make_args = {'num_objects':100, 'default_reward':-0.01}
        val_envs = [apple_pickgame(gym.make(env_id, **make_args), max_steps=5000, auto_reset=True, k=1) for i in range(15)]
        envs = BatchEnv(apple_pickgame, env_id, num_envs, max_steps=5000, auto_reset=False, k=1)
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
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(15)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)

    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape

    env.close()
    print('action space', action_size)

    dqn_args = dict(model=NatureCNN,
                    input_shape=input_size,
                    action_size=action_size,
                    lr=1e-3,
                    lr_final=1e-6,
                    grad_clip=0.5,
                    decay_steps=50e6//(num_envs*nsteps),
                    optim=torch.optim.RMSprop,
                    device='cuda')
      
    Q = DQN(**dqn_args)
    TargetQ = DQN(**dqn_args)

    DDQN = SyncDDQN(envs=envs,
                    model=Q,
                    target_model=TargetQ,
                    model_dir=model_dir,
                    log_dir=train_log_dir,
                    val_envs=val_envs,
                    action_size=action_size,
                    train_mode='nstep',
                    return_type='lambda',
                    total_steps=50e6,
                    nsteps=nsteps,
                    gamma=0.99,
                    lambda_=0.95,
                    save_freq=0,
                    render_freq=0,
                    validate_freq=1e5,
                    num_val_episodes=15,
                    update_target_freq=10000,
                    epsilon_start=1,
                    epsilon_final=0.01,
                    epsilon_steps=2e6,
                    epsilon_test=0.01,
                    log_scalars=False)
    
    DDQN.update_target()
    DDQN.train()

if __name__ == "__main__":
    import apple_picker
    env_id_list = [ 'SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4','MontezumaRevengeDeterministic-v4', ]
    #env_id_list = ['MontezumaRevengeDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'CartPole-v1', 'Acrobot-v1', ]
    env_id_list = ['ApplePicker-v0']
    for env_id in env_id_list:
        main(env_id)
