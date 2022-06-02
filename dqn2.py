import argparse
from collections import deque
import itertools
import random
import time

import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(np.array(x), dtype=torch.float, device=device) for x in zip(*transitions))

class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQN:
    def __init__(self, args, env):
        self._behavior_net = Net(state_dim=env.observation_space.shape[0], 
                                action_dim=env.action_space.n).to(args.device)
        self._target_net = Net(state_dim=env.observation_space.shape[0],
                                action_dim=env.action_space.n).to(args.device)
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr)
        self._memory = ReplayMemory(capacity=args.capacity)

        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, action_space):
        with torch.no_grad():

            state = torch.tensor(np.array([state]), dtype=torch.float, device=self.device)
            Q_value = self._behavior_net(state)
            policy = F.softmax(Q_value / self.alpha, 1)
            m = Categorical(policy)
            action = m.sample()
            return int(action)

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state, [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network()
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def calculate_q_value(self, Q_value, policy, action):
        entropy = -torch.sum(policy * torch.log(policy + 1e-8), 1).reshape(self.batch_size, 1)
        value = torch.sum(policy * Q_value, 1).reshape(self.batch_size, 1)
        pi = policy.gather(1, action.long())
        return self.alpha * (torch.log(pi + 1e-8) + entropy) + value

    def _update_behavior_network(self):
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)

        self._optimizer.zero_grad()

        Q_value = self._behavior_net(state)
        policy = F.softmax(Q_value / self.alpha, 1)
        q_value = self.calculate_q_value(Q_value, policy, action)
        
        with torch.no_grad():
            Q_next = self._behavior_net(next_state).gather(1, action.long())        
            policy_next = F.softmax(Q_next / self.alpha, 1)
            action_next = torch.argmax(policy_next, 1).reshape(self.batch_size, 1)
            q_next = self.calculate_q_value(Q_next, policy_next, action_next)
            q_target = reward + self.gamma * q_next * (1 - done)

        mse_criterion = nn.MSELoss()
        loss = mse_criterion(q_value, q_target)
        loss.backward()
        self._optimizer.step()

    def _update_target_network(self):
        for target_param, param in zip(self._target_net.parameters(), self._behavior_net.parameters()):
            target_param.data.copy_(param.data)

    def save(self, args, checkpoint=False, data=None):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'data': data
            }, f'model/{args.algo_name}/{args.env_name}_checkpoint.pth')
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, f'model/{args.algo_name}/{args.env_name}_model.pth')

    def load(self, args, checkpoint=False):
        if checkpoint:
            model = torch.load(f'model/{args.algo_name}/{args.env_name}_checkpoint.pth')
            self._behavior_net.load_state_dict(model['behavior_net'])
            self._target_net.load_state_dict(model['target_net'])
            return model['data']
        else:
            model = torch.load(f'model/{args.algo_name}/{args.env_name}_model.pth')
            self._behavior_net.load_state_dict(model['behavior_net'])

def train(args, env, agent):
    print('Start Training')
    action_space = env.action_space
    start_episode = 0
    total_steps = 0
    ewma_reward = 0
    best_ewma_reward =0
    accumulate_time = 0

    if args.checkpoint:
        data = agent.load(args, checkpoint=True)
        start_episode = data['start_episode']
        total_steps = data['total_steps']
        ewma_reward = data['ewma_reward']
        best_ewma_reward = data['best_ewma_reward']
        accumulate_time = data['accumulate_time']

    start_time = time.time()
    for episode in itertools.count(start=start_episode + 1):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):

            action = agent.select_action(state, action_space)
            
            next_state, reward, done, _ = env.step(action)
            
            agent.append(state, action, reward, next_state, done)
            if agent._memory.__len__() >= args.batch_size:
                agent.update(total_steps)

            if args.render:
                env.render()

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                
                if ewma_reward > best_ewma_reward:
                    best_ewma_reward = ewma_reward
                    agent.save(args)

                data = {
                    'start_episode': episode,
                    'total_steps': total_steps,
                    'ewma_reward': ewma_reward,
                    'best_ewma_reward': best_ewma_reward,
                    'accumulate_time': time.time()-start_time+accumulate_time
                }
                agent.save(args, checkpoint=True, data=data)

                writer = open(f'./log/{args.algo_name}/{args.env_name}_log.txt', 'a') 
                writer.write('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tTime: {:.2f}s'
                    .format(total_steps, episode, t, total_reward, ewma_reward, time.time()-start_time+accumulate_time) + '\n')
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tTime: {:.2f}s'
                    .format(total_steps, episode, t, total_reward, ewma_reward, time.time()-start_time+accumulate_time))

                break
        if total_steps > args.max_step:
            break
    env.close()

def test(args, env, agent):
    print('Start Testing')
    action_space = env.action_space

    rewards = list()
    for n_episode in range(args.test_episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            action = agent.select_action(state, action_space)
            next_state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            state = next_state
            total_reward += reward
            if done:
                print('Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
                        .format(n_episode, t, total_reward))
                break
        rewards.append(total_reward)
    print('Average Reward', np.mean(rewards))
    env.close()

def main():
    ## arguments ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-e', '--env_name', default='CartPole-v1')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--render', action='store_true')
    # train
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--max_step', default=2000000, type=int)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--alpha', default=.1, type=float)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    # test
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    args = parser.parse_args()

    args.algo_name = 'dqn2'

    ## create dirs
    os.makedirs('model', exist_ok=True)
    os.makedirs('log', exist_ok=True)
    os.makedirs(f'model/{args.algo_name}', exist_ok=True)
    os.makedirs(f'log/{args.algo_name}', exist_ok=True)

    env = gym.make(args.env_name)
    agent = DQN(args, env)

    ## main ##
    if args.train:

        ## remove log file
        if not args.checkpoint:
            if os.path.exists(f'./log/{args.algo_name}/{args.env_name}_log.txt'):
                os.remove(f'./log/{args.algo_name}/{args.env_name}_log.txt')

        train(args, env, agent)

    if args.test:
        agent.load(args, checkpoint=args.checkpoint)
        test(args, env, agent)

if __name__ == '__main__':
    main()