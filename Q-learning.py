from collections import deque
import random
import threading
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=10, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    parser.add_argument('--capacity', default=10000, type=int, help='capacity for replay buffer')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for replay buffer')
    parser.add_argument('--freq', default=10, type=int, help='update frequency for replay buffer')
    parser.add_argument('--alpha', default=.1, type=float, help='parameter of entropy regularization')
    return parser.parse_args()

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: np.array(Image.fromarray(img[35:195].mean(2)).resize((80,80))).astype(np.float32).reshape(1,80,80)/255. # rescale the image

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(np.array(x), dtype=torch.float) for x in zip(*transitions))

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, actions.view(-1,1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    return policy_loss + 0.5 * value_loss

def update_shared_model(shared_model, shared_optimizer, model, loss):
    shared_optimizer.zero_grad() ; loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
    shared_optimizer.step()

def calculate_q_value(value, policy, action, args):
    entropy = -torch.sum(policy * torch.log(policy + 1e-8), 1).reshape(args.batch_size, 1)
    pi = policy.gather(1, action.long())
    return args.alpha * (torch.log(pi + 1e-8) + entropy) + value

def q_update(shared_model, shared_optimizer, model, args, memory):
    state, action, reward, next_state, done, hx = memory.sample(args.batch_size)
    
    value, logit, nhx = model((state, hx))
    policy = torch.exp(F.log_softmax(logit, dim=-1))
    q_value = calculate_q_value(value, policy, action, args)

    with torch.no_grad():
        value_next, logit_next, _ = model((next_state, nhx))
        policy_next = torch.exp(F.log_softmax(logit_next, dim=-1))
        action_next = torch.argmax(policy_next, 1).reshape(args.batch_size, 1)
        q_next = calculate_q_value(value_next, policy_next, action_next, args)
        q_target = reward + args.gamma * q_next * (1 - done)
    
    mse_criterion = nn.MSELoss()
    loss = mse_criterion(q_value, q_target)

    update_shared_model(shared_model, shared_optimizer, model, loss)
    

def train(shared_model, shared_optimizer, rank, args, info, memory):
    env = gym.make(args.env) # make a local (unshared) environment
    env.seed(args.seed + rank) ; torch.manual_seed(args.seed + rank) # seed everything
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions) # a local/unshared model
    state = torch.tensor(prepro(env.reset())) # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    while info['frames'][0] <= 2e7 or args.test: # training 20M steps
        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, nhx = model((state.view(1,1,80,80), hx))
            logp = F.log_softmax(logit, dim=-1)

            entropy = -torch.sum(torch.exp(logp) * logp, 1).reshape(1, 1)
            q_value = args.alpha * (logp + entropy) + value

            action = F.softmax(q_value / args.alpha, 1).multinomial(num_samples=1).data[0] # logp.max(1)[1].data if args.test else
            next_state, reward, done, _ = env.step(action.numpy()[0])
            if args.render: env.render()

            next_state = torch.tensor(prepro(next_state)) ; epr += reward
            reward = np.clip(reward, -1, 1) # reward
            done = done or episode_length >= 1e4 # don't playing one ep for too long

            memory.append(np.array(state), [action], [reward], np.array(next_state), [int(done)], np.array(hx[0].detach()))

            # Q-learning Update
            if step % args.freq == 0 and memory.__len__() >= args.batch_size * 10:
                q_update(shared_model, shared_optimizer, model, args, memory)

            state = next_state
            hx = nhx

            info['frames'].add_(1) ; num_frames = int(info['frames'].item())
            if num_frames % 1e6 == 0: # save every 1M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

            if done: # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 10: # print info ~ every 10 sec
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.3f}M, mean epr {:.3f}'
                    .format(elapsed, info['episodes'].item(), num_frames/1e6, info['run_epr'].item()))
                last_disp_time = time.time()

            if done: # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)

        next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        # loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        # eploss += loss.item()
        # update_shared_model(shared_model, shared_optimizer, model, loss)

if __name__ == "__main__":

    args = get_args()
    args.save_dir = './{}/Q-learning/'.format(args.env.lower()) # keep the directory structure simple
    if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode w one process
    if args.test:  args.lr = 0 # don't train in render mode
    args.num_actions = gym.make(args.env).action_space.n # get the action space of this game
    os.makedirs('./{}/'.format(args.env.lower()), exist_ok=True) # make dir to save
    os.makedirs(args.save_dir, exist_ok=True) # make dir to save models etc.

    torch.manual_seed(args.seed)
    shared_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    memory = ReplayMemory(capacity=args.capacity)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w') # clear log file
    
    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info, memory))
        p.start() ; processes.append(p)
    for p in processes: p.join()