from collections import deque
from network import *
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
device = 'cpu'
env_name = 'ring_our0.9'
now = datetime.now()
summary = SummaryWriter(logdir='runs/' + env_name + "_{}-{}-{}-{}".format(now.month, now.day, now.hour, now.minute))


def update_weight(prob, portion):
    prob = np.array(prob)
    portion = (portion ** 2 / sum(portion ** 2)).squeeze()
    prob = np.power(prob,(portion))
    return prob




class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.memory = deque([], maxlen=size)
        self.prob = deque([],maxlen=size)

    def push(self, x):
        self.memory.append(x)

    def push_p(self,x):
        self.prob.append(x)

    def sample(self, batch_size): #TODO : Probabilty에 따라서 샘플링 하기
        if batch_size == 0:
            return np.array([]),np.array([]),np.array([]),np.array([]),np.array([]), np.array([])
        elif len(self.memory) >= batch_size:
            idx = range(len(self.memory))
            batch = np.random.choice(idx, batch_size, p= np.array(self.prob)**2 / np.sum(np.array(self.prob)**2))
            state, action, reward, next_state, done = map(np.stack, zip(*np.array(self.memory)[batch]))
            return state, action, reward, next_state, done, batch
        else:
            return np.array([]),np.array([]),np.array([]),np.array([]),np.array([]), np.array([])


    def get_len(self):
        return len(self.memory)

    def pops(self,idx):
        memory = np.delete(self.memory, idx, 0)
        prob = np.delete(self.prob, idx, 0)
        self.memory = deque(memory, maxlen=self.size)
        self.prob = deque(prob, maxlen=self.size)



class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(30, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory1,memory2, optimizer, p1):
    for i in range(1):

        p = p1 if memory2.get_len() > batch_size * p1 else 0

        states1, actions1, rewards1, next_states1, ndones1, idx1 = memory2.sample( int(batch_size * p)) ## memory2 : used
        states2, actions2, rewards2, next_states2, ndones2, idx2 = memory1.sample( int(batch_size * (1 - p))) ## memory1 : new_one

        states = states1.tolist() + states2.tolist()
        actions = actions1.tolist() + actions2.tolist()
        rewards = rewards1.squeeze().tolist() + rewards2.tolist()
        next_states = next_states1.tolist() + next_states2.tolist()
        ndones = ndones1.squeeze().tolist() + ndones2.tolist()
        if len(idx1) > 0:
            prob = np.array(memory2.prob)[idx1].tolist() + np.array(memory1.prob)[idx2].tolist()
        else:
            prob = np.array(memory1.prob)[idx2].tolist()
        # make sure all are torch tensors
        s = torch.FloatTensor(states).to(device)
        a = torch.LongTensor(actions).to(device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        s_prime = torch.FloatTensor(next_states).to(device)
        done_mask = torch.FloatTensor(np.float32(ndones)).unsqueeze(1).to(device)

        q_out = q(s)
        q_a = q_out.gather(1, a.unsqueeze(-1))
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO : update replay_old / prob
        prob = update_weight(prob, portion = abs(target - q_a).detach().numpy())

        for k in range(int(batch_size * (1 - p1))):
            if device == 'cpu':
                memory2.push((states2[k].squeeze(), int(actions2[k]), float(rewards2[k]), next_states2[k].squeeze(),
                              float(ndones2[k])))
            else:
                memory2.push((states2[k].squeeze(), int(actions2[k]), float(rewards2[k]), next_states2[k].squeeze(),
                              float(ndones2[k])))
            memory2.push_p(prob[k])

        if p != 0:
            memory1.pops(idx2)
            for k in range(int(batch_size * p)):
                memory2.memory[idx1[k]] = states1[k].squeeze(), actions1[k], rewards1[k], next_states1[k].squeeze(), \
                                          ndones1[k]
                memory2.prob[idx1[k]] = prob[k]

    return loss


def normalize(x):
    x = (x-np.mean(x)) / np.std(x)
    return x

env = create_env()
q = Qnet()
q_target = Qnet()
q_target.load_state_dict(q.state_dict())
memory1 = ReplayMemory(buffer_limit)
memory2 = ReplayMemory(batch_size * 50)
device = 'cpu'
print_interval = 20
score = 0.0
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
k = 0
p1 = 0.9
for n_epi in range(15000):
    epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
    go = False
    while not go:
        try:
            env.reset()
            go = True
        except:
            go = False
    env.ready()
    s = env.get_state()
    s = normalize(s)

    for t in range(600):
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        if a == 0:
            s_prime, r, done, info = env.step(0)
            s_prime = normalize(s_prime)
        else:
            s_prime, r, done, info = env.run()
            s_prime = normalize(s_prime)
            print(r)

        done_mask = 0.0 if done else 1.0
        memory1.push((s, a, r / 100.0, s_prime, done_mask))
        memory1.push_p(np.exp(1))

        s = s_prime

        score += r
        if done:
            break

    if memory1.get_len() > 50:
        # break
        k+=1
        loss = train(q, q_target, memory1,memory2, optimizer,p1)
        summary.add_scalar('/home/deepvisions/prac1/dqn/runs/ring/loss', loss, k)

    if n_epi % print_interval == 0 and n_epi != 0:
        q_target.load_state_dict(q.state_dict())
        summary.add_scalar('/home/deepvisions/prac1/dqn/runs/ring/score', score, n_epi)
        print("# of episode :{}, avg score : {:.1f}, buffer1 size : {}, buffer2 size : {}, epsilon : {:.1f}%, k : {}".format(
            n_epi, score / print_interval, memory1.get_len(),memory2.get_len(), epsilon * 100,k))
        score = 0.0
        torch.save(q_target.state_dict(), 'ring'+str(p1)+'.pth')
env.close()

