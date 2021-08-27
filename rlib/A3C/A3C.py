import gym
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import math
import time

from rlib.A2C.ActorCritic import ActorCritic
from rlib.networks.networks import NatureCNN
from rlib.utils.wrappers import AtariEnv
from rlib.utils.utils import stack_many, tonumpy, totorch, lambda_return


def train(global_model, model, env, nsteps, num_episodes, ID):
    opt = torch.optim.RMSprop(global_model.parameters(), lr=1e-3)
    episode = 0
    episode_steps = 0
    episode_score = 0
    T = 0
    state = env.reset()
    start = time.time()
    while episode < num_episodes:
        rollout = []
        for t in range(nsteps):
            with torch.no_grad():
                policy, value = model(totorch(state[None], device='cpu'))
                policy, value = tonumpy(policy), tonumpy(value)
            action = np.random.choice(policy.shape[1], p=policy[0])
            next_state, reward, done, info = env.step(action)
            episode_score += reward
            rollout.append((state, action, reward, value, done))
            state = next_state

            T += 1
            episode_steps += 1

            if done or t == nsteps-1:
                states, actions, rewards, values, dones = stack_many(*zip(*rollout))
                with torch.no_grad():
                    _, last_values = model.forward(totorch(next_state[None], device='cpu'))
                    last_values = last_values.cpu().numpy()
                

                    R = lambda_return(rewards, values, last_values, dones, gamma=0.9, lambda_=0.95, clip=False)
                
                loss = update_params(model, global_model, opt, states, actions, R)
                
                #self.T += t

                if done:
                    episode += 1
                    state = env.reset()
                    if episode % 1 == 0:
                        time_taken = time.time() - start 
                        print(f'worker {ID}, total worker steps {T:,} local episode {episode}, episode score {episode_score} episode steps {episode_steps}, time taken {time_taken:,.1f}s, fps {episode_steps/time_taken:.2f}')
                    episode_steps = 0
                    episode_score = 0
                    start = time.time()
                    break
    

def update_params(lm, gm, gopt, states, actions, R):
    states, R, actions = totorch(states, 'cpu'), totorch(R, 'cpu'), totorch(actions, 'cpu')
    actions_onehot = F.one_hot(actions.long(), num_classes=lm.action_size)
    policies, values = lm.forward(states)
    loss = lm.loss(policies, R, values, actions_onehot)

    loss.backward()

    if lm.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(lm.parameters(), lm.grad_clip)
    
    for local_param, global_param in zip(lm.parameters(), gm.parameters()):
        global_param._grad = local_param.grad
    
    gopt.step()
    gopt.zero_grad()
    #self.scheduler.step()

    lm.load_state_dict(gm.state_dict())
    return loss.detach().cpu().numpy()



# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
#                  weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         # State initialization
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)

#                 # share in memory
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()

class SharedAdam(torch.optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss




if __name__ == '__main__':
    env_id = 'SpaceInvadersDeterministic-v4'
    env = AtariEnv(gym.make(env_id), reset=True)
    input_size = env.reset().shape
    action_size = env.action_space.n
    
    print('action_size', action_size)

    global_model = ActorCritic(NatureCNN, input_size, action_size, build_optimiser=False)
    global_model.share_memory()

    #opt = SharedAdam(global_model.parameters(), lr=1e-3)
    #opt.share_memory()

    #actor = ActorCritic(NatureCNN, input_size, action_size)
    env_args = dict(k=4, rescale=84, episodic=True, reset=True, clip_reward=True, Noop=True, time_limit=None, channels_first=True)
    model_args = dict(model=NatureCNN, input_size=input_size, action_size=action_size, build_optimiser=False)

    processes = []
    for rank in range(8):
        p = mp.Process(target=train, args=(global_model, ActorCritic(**model_args), AtariEnv(gym.make(env_id), **env_args), 20, 1000, rank))
        p.start()
        processes.append(p)
        time.sleep(0.5)
    for p in processes:
        p.join()