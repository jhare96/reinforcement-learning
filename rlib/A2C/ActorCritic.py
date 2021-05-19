import torch
import torch.nn.functional as F
import numpy as np
from rlib.networks.networks import MaskedLSTMCell, MaskedRNN
from rlib.utils.schedulers import polynomial_sheduler
from rlib.utils.utils import totorch, tonumpy

class ActorCritic(torch.nn.Module):
    def __init__(self, model, input_size, action_size, entropy_coeff=0.01, value_coeff=0.5, lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip=0.5, build_optimiser=True, optim=torch.optim.Adam, optim_args={}, **model_args):
        super(ActorCritic, self).__init__()
        self.lr = lr
        self.lr_final = lr_final
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size

        self.model = model(input_size, **model_args)
        dense_size = self.model.dense_size 
        self.policy_distrib = torch.nn.Linear(dense_size, action_size) # Actor
        self.V = torch.nn.Linear(dense_size, 1) # Critic 
        
        if build_optimiser:
            self.optimiser = optim(self.parameters(), lr, **optim_args)
            self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)
        
    def loss(self, policy, R, V, actions_onehot):
        Advantage = R - V
        value_loss = 0.5 * torch.mean(torch.square(Advantage))

        log_policy = torch.log(torch.clip(policy, 1e-6, 0.999999))
        log_policy_actions = torch.sum(log_policy * actions_onehot, dim=1)
        policy_loss =  torch.mean(-log_policy_actions * Advantage.detach())

        entropy = torch.mean(torch.sum(policy * -log_policy, dim=1))
        loss =  policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        return loss

    def forward(self, state):
        enc_state = self.model(state)
        policy = F.softmax(self.policy_distrib(enc_state), dim=-1)
        value = self.V(enc_state).view(-1)
        return policy, value
    
    def evaluate(self, state:np.ndarray):
        state = totorch(state)
        with torch.no_grad():
            policy, value = self.forward(state)
        return tonumpy(policy), tonumpy(value)
    
    def backprop(self, state, R, action):
        state, R, action = totorch(state), totorch(R), totorch(action)
        action_onehot = F.one_hot(action.long(), num_classes=self.action_size)
        policy, value = self.forward(state)
        loss = self.loss(policy, R, value, action_onehot)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()



class ActorCritic_LSTM(torch.nn.Module):
    def __init__(self, model, input_size, action_size, cell_size, entropy_coeff=0.01, value_coeff=0.5, lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip=0.5, build_optimiser=True, optim=torch.optim.Adam, optim_args={}, **model_args):
        super(ActorCritic_LSTM, self).__init__()
        self.lr = lr
        self.lr_final = lr_final
        self.input_size = input_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.cell_size = cell_size
        self.action_size = action_size


        self.model = model(input_size, **model_args)
        self.dense_size = self.model.dense_size
        self.lstm = MaskedRNN(MaskedLSTMCell(cell_size, self.dense_size), time_major=True)

        self.policy_distrib = torch.nn.Linear(cell_size, action_size) # Actor
        self.V = torch.nn.Linear(cell_size, 1) # Critic 


        if build_optimiser:
            self.optimiser = optim(self.parameters(), lr, **optim_args)
            self.scheduler = polynomial_sheduler(self.optimiser, lr_final, decay_steps, power=1)
        
    def loss(self, policy, R, V, actions_onehot):
        Advantage = R - V
        value_loss = 0.5 * torch.mean(torch.square(Advantage))

        log_policy = torch.log(torch.clip(policy, 1e-6, 0.999999))
        log_policy_actions = torch.sum(log_policy * actions_onehot, dim=1)
        policy_loss =  torch.mean(-log_policy_actions * Advantage.detach())

        entropy = torch.mean(torch.sum(policy * -log_policy, dim=1))
        loss =  policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        return loss

    def forward(self, state, hidden=None, done=None):
        T, num_envs = state.shape[:2]
        folded_state = state.view(-1, *self.input_size)
        enc_state = self.model(folded_state)
        folded_enc_state = enc_state.view(T, num_envs, self.dense_size)
        lstm_outputs, hidden = self.lstm(folded_enc_state, hidden, done)
        policy = F.softmax(self.policy_distrib(lstm_outputs), dim=-1).view(-1, self.action_size)
        value = self.V(lstm_outputs).view(-1)
        return policy, value, hidden
    
    def evaluate(self, state, hidden=None, done=None):
        with torch.no_grad():
            policy, value, hidden = self.forward(state, hidden, done)
        return tonumpy(policy), tonumpy(value), hidden
    
    def backprop(self, state, R, action, hidden, done):
        state, R, action, done = totorch(state), totorch(R), totorch(action), totorch(done)
        action_onehot = F.one_hot(action.long(), num_classes=self.action_size)
        policy, value, hidden = self.forward(state, hidden, done)
        loss = self.loss(policy, R, value, action_onehot)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()
    
    def get_initial_hidden(self, batch_size):
        return torch.zeros((batch_size, self.cell_size)), torch.zeros((batch_size, self.cell_size))
    
    def mask_hidden(self, hidden, dones):
        mask = totorch(1-dones).view(-1, 1)
        return (hidden[0]*mask, hidden[1]*mask)