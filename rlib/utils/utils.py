import torch
import numpy as np

def fastsample(policy:np.ndarray, k=1):
    return torch.multinomial(torch.from_numpy(policy), num_samples=k, replacement=True).view(-1).numpy()

def log_uniform(low=1e-10, high=1, size=()):
    return np.exp(np.random.uniform(low=np.log(low), high=np.log(high), size=size))

def stack_many(*args, stack=np.stack):
    return tuple([stack(arg) for arg in args])

def normalise(x, mean, std):
    return (x-mean)/std

def fold_batch(x):
    rows, cols = x.shape[0], x.shape[1]
    y = x.reshape(rows*cols,*x.shape[2:])
    return y

def unfold_batch(x, length, batch_size):
    return x.reshape(length, batch_size, *x.shape[1:])

def fold_many(*args):
    return tuple([fold_batch(arg) for arg in args])

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

def totorch(x, device='cuda'):
    x = torch.from_numpy(x).float().to(device)
    return x

def tonumpy(x):
    return x.detach().cpu().numpy()

def tonumpy_many(*args):
    return tuple([tonumpy(arg) for arg in args])

def totorch_many(*args, device='cuda'):
    return tuple([totorch(arg, device) for arg in args])

class Welfords_algorithm(object):
    #https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    def __init__(self, mean=0, epsilon=1e-4):
        self.mean = mean
        self.n = epsilon
        self.M2 = 1
    
    def update(self, x):
        return self.update_from_mean(x.mean(axis=0))
    
    def update_from_mean(self, x):
        self.n +=1
        prev_mean = self.mean
        new_mean = prev_mean + ((x - prev_mean) / self.n)
        self.M2 += (x - new_mean) * (x - prev_mean)
        self.var = self.M2 / self.n
        self.mean = new_mean
        return self.mean, np.sqrt(self.var)

#https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float32):
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        return self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        
        return self.mean, np.sqrt(self.var)
    


def nstep_return(rewards, last_values, dones, gamma=0.99, clip=False):
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

def lambda_return(rewards, values, last_values, dones, gamma=0.99, lambda_=0.8, clip=False):
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

def GAE(rewards, values, last_values, dones, gamma=0.99, lambda_=0.95, clip=False):
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