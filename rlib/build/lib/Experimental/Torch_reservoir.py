import torch
import torch.nn.functional as F
import gym 
import time 
from VecEnv import*


class cnn_lstm_reservoir(torch.nn.Module):
    def __init__(self, cell_size, batch_size=32):
        super(cnn_lstm_reservoir, self).__init__()
        self.cell_size = cell_size
        self.conv1 = torch.nn.Conv2d(1,32,(8,8),(4,4))
        self.conv2 = torch.nn.Conv2d(32,64,(4,4),(2,2))
        self.conv3 = torch.nn.Conv2d(64,64,(3,3),(1,1))
        self.dense = torch.nn.Linear(7*7*64, 512)
        self.rnn = torch.nn.RNN(512, cell_size, 1, nonlinearity='tanh')
        

    def rnn_init(self, batch_size=1):
        return torch.zeros(1, batch_size, self.cell_size)
    
    def forward(self, x, hidden):
        time = x.shape[0]
        batch_size = x.shape[1]
        input_shape = x.shape[2:]
        x = x.reshape(-1, *x.shape[2:])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        flatten = x.reshape(x.shape[0], -1)
        dense = F.relu(self.dense(flatten))
        #print('flat shape', x.shape)
        x = dense.reshape(time, batch_size, dense.shape[1])
        rnn_out, hidden = self.rnn(x, hidden)
        #print('rnn_out', rnn_out.shape)
        rnn_out = rnn_out.reshape(time*batch_size, rnn_out.shape[2])
        return rnn_out, hidden

class ActorCritic(torch.nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.value = torch.nn.Linear(256, 1)
        self.policy = torch.nn.Linear(256,num_actions)
    
    def forward(self, x):
        value = self.value(x)
        policy = F.softmax(self.policy(x))
        return policy, value





def main():
    env  = BatchEnv(AtariEnv, 'SpaceInvaders-v0', 32, False, k=1, episodic=True, reset=True)
    action_size = env.action_space.n
    
    cnn_lstm = cnn_lstm_reservoir(256).cuda()
    for param in cnn_lstm.parameters():
        param.requires_grad = False
        print('parma', param.requires_grad)
    
    actor = ActorCritic(action_size).cuda()
    opt = torch.optim.Adam(actor.parameters(), lr = 0.001)

    states = env.reset().reshape(1,-1,1,84,84)
    prev_hidden = cnn_lstm.rnn_init(len(env)).cuda()
    start = time.time()
    for t in range(1000000):
        x, hidden =  cnn_lstm(torch.from_numpy(states/255.0).cuda().float(), prev_hidden)
        #print('rnn_out', x.shape)
        policies, values = actor(x)
        actions = [np.random.choice(policies.shape[1], p=policies[i].cpu().detach().numpy()) for i in range(len(env))]
        next_states, rewards, dones, infos = env.step(actions)
        next_states = next_states.reshape(1,-1,1,84,84)/255.0

        x, _ = cnn_lstm(torch.from_numpy(next_states).cuda().float(), hidden)
        _, bootstap = actor(x)
        y = torch.from_numpy(rewards).cuda().float() * float(0.99) * bootstap * torch.from_numpy((1-dones)).cuda().float()
        actions_onehot = torch.eye(action_size)[actions].float().cuda()
        adv = y - values
        #adv.requires_grad = False
        #print('values loss', (y - values).pow(2).mean().shape)
        value_loss = (y - values).pow(2).mean()
        loss = value_loss + (torch.sum(torch.mul(policies,actions_onehot),dim=1) * adv.detach()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        states = next_states
        prev_hidden = hidden

        if t % 1000 == 0 and t > 0:
            time_taken = time.time() - start 
            fps = (1000 * len(env)) / time_taken
            print('loss {}, fps {}'.format(loss.detach().cpu().numpy(), fps))
            start = time.time()



    
if __name__ == '__main__':
    main()