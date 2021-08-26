import torch 
import torch.nn.functional as F 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import threading
import time

from rlib.utils.VecEnv import*
from rlib.utils.wrappers import*
from rlib.utils.utils import fold_batch, stack_many, one_hot, totorch, totorch_many, tonumpy

class VINCNN(torch.nn.Module):
    def __init__(self, input_size, action_size, k=10, lr=1e-3, device='cuda'):
        super(VINCNN, self).__init__()
        channels, height, width = input_size
        self.action_size = action_size
        self.conv_enc = torch.nn.Conv2d(channels, 150, kernel_size=[3,3], stride=[1,1], padding=1).to(device) # φ(s)
        self.R_bar = torch.nn.Conv2d(150, 1, kernel_size=[1,1], stride=[1,1], padding=0, bias=False).to(device)
        self.Q_bar = torch.nn.Conv2d(1, action_size, kernel_size=[3,3], stride=[1,1], padding=1, bias=False).to(device)
        self.w = torch.nn.Parameter(torch.zeros(action_size, 1, 3, 3), requires_grad=True).to(device)
        self.Q = torch.nn.Linear(action_size, action_size).to(device)
        self.k = k # nsteps to plan with VIN
        self.optim = torch.optim.RMSprop(params=self.parameters(), lr=lr)
        self.device = device
    
    def forward(self, img, x, y):
        hidden = self.conv_enc(img)
        R_bar = self.R_bar(hidden)
        Q_bar = self.Q_bar(R_bar)
        V_bar, _ = torch.max(Q_bar, dim=1, keepdim=True)
        batch_size = img.shape[0]
        psi = self._plan_ahead(R_bar, V_bar)[torch.arange(batch_size), :, x.long(), y.long()].view(batch_size, self.action_size) # ψ(s)
        Qsa = self.Q(psi)
        return Qsa
    

    def backprop(self, states, locs, R, actions):
        x, y = zip(*locs)
        Qsa = self.forward(totorch(states, self.device), torch.tensor(x).to(self.device), torch.tensor(y)).to(self.device)
        actions_onehot = totorch(one_hot(actions, self.action_size), self.device)
        Qvalue = torch.sum(Qsa * actions_onehot, axis=1)
        loss = torch.mean(torch.square(totorch(R).float().cuda() - Qvalue))
        
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return loss.detach().cpu().numpy()

    
    def value_iteration(self, r, V):
        return F.conv2d(
                # Stack reward with most recent value
                torch.cat([r, V], 1),
                # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
                torch.cat([self.Q_bar.weight, self.w], 1),
                stride=1,
                padding=1)

    def _plan_ahead(self, r, V):
        for i in range(self.k):
            Q = self.value_iteration(r, V)
            V, _ = torch.max(Q, dim=1, keepdim=True)
        
        Q = self.value_iteration(r, V)
        return Q



class VINTrainer(object):
    def __init__(self, model, envs, val_envs, epsilon=0.1, epsilon_final=0.1, epsilon_steps=1000000, epsilon_test=0.1,
                return_type='nstep', log_dir='logs/', model_dir='models/', total_steps=50000000, nsteps=20, gamma=0.99, lambda_=0.95, 
                validate_freq=1e6, save_freq=0, render_freq=0, update_target_freq=0, num_val_episodes=50, log_scalars=True):
        self.model = model
        self.env = envs
        self.num_envs = len(envs)
        self.val_envs = val_envs
        self.total_steps = total_steps
        self.action_size = self.model.action_size
        self.epsilon = epsilon
        self.epsilon_test = epsilon_test
        self.states = self.env.reset()
        self.loc = self.get_locs()
        print('locs', self.loc)

        self.total_steps = int(total_steps)
        self.nsteps = nsteps
        self.return_type = return_type
        self.gamma = gamma
        self.lambda_ = lambda_

        self.validate_freq = int(validate_freq) 
        self.num_val_episodes = num_val_episodes

        self.save_freq = int(save_freq) 
        self.render_freq = render_freq
        self.target_freq = int(update_target_freq)
        self.t=1

        self.validate_rewards = []
        self.lock = threading.Lock()
        self.scheduler = self.linear_schedule(epsilon, epsilon_final, epsilon_steps)

        self.log_scalars = log_scalars
        self.log_dir = log_dir

        if log_scalars:
            # Tensorboard Variables
            train_log_dir = self.log_dir  + '/train'
            self.train_writer = SummaryWriter(train_log_dir)
    
    def nstep_return(self, rewards, last_values, dones, gamma=0.99, clip=False):
        if clip:
            rewards = np.clip(rewards, -1, 1)

        T = len(rewards)
        
        # Calculate R for advantage A = R - V 
        R = np.zeros_like(rewards)
        R[-1] = last_values * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = rewards[i] + gamma * R[i+1] * (1-dones[i])
        
        return R
    
    def lambda_return(self, rewards, values, last_values, dones, gamma=0.99, lambda_=0.8, clip=False):
        if clip:
            rewards = np.clip(rewards, -1, 1)
        T = len(rewards)
        # Calculate eligibility trace R^lambda 
        R = np.zeros_like(rewards)
        R[-1] =  last_values * (1-dones[-1])
        for t in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[t] = rewards[t] + gamma * (lambda_* R[t+1] + (1.0-lambda_) * values[t+1]) * (1-dones[t])
        
        return R

    def GAE(self, rewards, values, last_values, dones, gamma=0.99, lambda_=0.95, clip=False):
        if clip:
            rewards = np.clip(rewards, -1, 1)
        # Generalised Advantage Estimation
        Adv = np.zeros_like(rewards)
        Adv[-1] = rewards[-1] + gamma * last_values * (1-dones[-1]) - values[-1]
        T = len(rewards)
        for t in reversed(range(T-1)):
            delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
            Adv[t] = delta + gamma * lambda_ * Adv[t+1] * (1-dones[t])
        
        return Adv
    
    def get_locs(self):
        locs = []
        for env in self.env.envs:
            locs.append(env.agent_loc)
        return locs

    def train(self):
        self.train_nstep()


    def train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        # main loop
        start = time.time()
        for t in range(self.t,num_updates+1):
            states, locs, actions, rewards, dones, infos, values, last_values = self.rollout()
            if self.return_type == 'nstep':
                R = self.nstep_return(rewards, last_values, dones, gamma=self.gamma)
            elif self.return_type == 'GAE':
                R = self.GAE(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_) + values
            elif self.return_type == 'lambda':
                R = self.lambda_return(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_, clip=False)
            # stack all states, actions and Rs from all workers into a single batch
            states, locs, actions, R = fold_batch(states), fold_batch(locs), fold_batch(actions), fold_batch(R)
            #print('locs', locs.shape)    
            l = self.model.backprop(states, locs, R, actions)
     
            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t,l,start,False)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0: 
                self.s += 1
                self.save(self.s)
                print('saved model')
            
            if self.target_freq > 0 and t % (self.target_freq // batch_size) == 0: # update target network (for value based learning e.g. DQN)
                self.update_target()

            self.t +=1
    
    def eval_state(self, state, loc):
        with torch.no_grad():
            x, y = zip(*loc)
            x, y = torch.tensor(x).to(self.device), torch.tensor(y).to(self.device)
            state_torch = totorch(state, self.device)
            Qsa = self.model(state_torch, x, y)
        return tonumpy(Qsa)

    def rollout(self):
        rollout = []
        for t in range(self.nsteps):
            Qsa = self.eval_state(self.states, self.loc)
            actions = np.argmax(Qsa, axis=1)
            random = np.random.uniform(size=(self.num_envs))
            random_actions = np.random.randint(self.action_size, size=(self.num_envs))
            actions = np.where(random < self.epsilon, random_actions, actions)
            next_states, rewards, dones, infos = self.env.step(actions)
            values = np.sum(Qsa * one_hot(actions, self.action_size), axis=-1)
            rollout.append((self.states, self.loc, actions, rewards, dones, infos, values))
            self.states = next_states
            self.epsilon = self.scheduler.step()
            self.loc = self.get_locs()
            
        states, locs, actions, rewards, dones, infos, values = stack_many(*zip(*rollout))

        last_Qsa = self.eval_state(next_states, self.loc) # Q(s,a|theta)
        last_actions = np.argmax(last_Qsa, axis=1)
        last_values = np.sum(last_Qsa * one_hot(last_actions, self.action_size), axis=-1)
        return states, locs, actions, rewards, dones, infos, values, last_values
    
    def get_action(self, state, loc):
        Qsa = self.eval_state(state, loc)
        if np.random.uniform() < self.epsilon_test:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(Qsa, axis=1)
        return action
    
    def validation_summary(self,t,loss,start,render):
        batch_size = self.num_envs * self.nsteps
        tot_steps = t * batch_size
        time_taken = time.time() - start
        frames_per_update = (self.validate_freq // batch_size) * batch_size
        fps = frames_per_update /time_taken 
        num_val_envs = len(self.val_envs)
        num_val_eps = [self.num_val_episodes//num_val_envs for i in range(num_val_envs)]
        num_val_eps[-1] = num_val_eps[-1] + self.num_val_episodes % self.num_val_episodes//(num_val_envs)
        render_array = np.zeros((len(self.val_envs)))
        render_array[0] = render
        threads = [threading.Thread(daemon=True, target=self.validate, args=(self.val_envs[i], num_val_eps[i], 10000, render_array[i])) for i in range(num_val_envs)]
        try:
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
    
        except KeyboardInterrupt:
            for thread in threads:
                thread.join()
    
            
        score = np.mean(self.validate_rewards)
        self.validate_rewards = []
        print("update %i, validation score %f, total steps %i, loss %f, time taken for %i frames:%fs, fps %f" %(t,score,tot_steps,loss,frames_per_update,time_taken,fps))
        
        if self.log_scalars:
            self.train_writer.add_scalar('Validation/Score', score)
            self.train_writer.add_scalar('Training/Loss', loss)


    def validate(self,env,num_ep,max_steps,render=False):
        episode_scores = []
        for episode in range(num_ep):
            state = env.reset()
            loc = env.agent_loc
            episode_score = []
            for t in range(max_steps):
                action = self.get_action(state[np.newaxis], [loc])
                next_state, reward, done, info = env.step(action)
                state = next_state
                loc = env.agent_loc

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
                self._epsilon = self._epsilon_final
            
            return self._epsilon



def main(env_id):
    num_envs = 32
    nsteps = 1
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    train_log_dir = 'logs/VIN/' + env_id +'/n_step/' + current_time 
    model_dir = "models/VIN/" + env_id + '/n_step/' + current_time 
    
    if 'ApplePicker' in env_id:
        print('ApplePicker')
        make_args = {'num_objects':300, 'default_reward':-0.01}
        val_envs = [apple_pickgame(gym.make('ApplePicker-v0', **make_args)) for i in range(10)]
        envs = DummyBatchEnv(apple_pickgame, 'ApplePicker-v0', num_envs, max_steps=1000, auto_reset=True, make_args=make_args)
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
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(5)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=4, reset=reset, episodic=False, clip_reward=True)
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    print('input shape', input_size)
    print('action space', action_size)


    
    vin = VINCNN(input_size,
                 action_size,
                 k=50,
                 lr=1e-3).cuda()
    

    trainer = VINTrainer(envs=envs,
              model=vin,
              log_dir=train_log_dir,
              val_envs=val_envs,
              return_type='nstep',
              total_steps=10e6,
              nsteps=nsteps,
              validate_freq=1e5,
              save_freq=0,
              render_freq=0,
              num_val_episodes=10,
              log_scalars=False)

    trainer.train()


if __name__ == "__main__":
    import apple_picker
    #env_id_list = ['SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'PongDeterministic-v4']
    #env_id_list = ['MontezumaRevengeDeterministic-v4']
    env_id_list = ['ApplePicker-v0']
    for env_id in env_id_list:
        main(env_id)
