import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from SumTree import SumTree

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0000     # L2 weight decay
BATCH_SIZE = 128         # minibatch size
BUFFER_SIZE = int(1e6)  # replay buffer size

# Prioritized Replay Buffer parameters
E = 0.01
A = 0.6
BETA = 0.4
BETA_INCREMENT_PER_SAMPLING = 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
other good values for params:
A = 0.7, BETA = 0.5
"""

"""This is the Agent with Prioritized Experience Replay
"""

def weighted_mse_loss(input_, target, weight):
    """Weighted MSE loss function,
    taken from
    https://discuss.pytorch.org/t/how-to-implement-weighted-mean-square-error/2547"""
    return torch.mean(weight * (input_ - target) ** 2)


def weighted_huber_loss(input_, target, weight):
    """Weighted Huber Loss, used https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    """
    
    huber_loss_delta = 1.0
    
    cond = weight * torch.abs(input_ - target) < huber_loss_delta
    
    L2 = 0.5 * weight * ((input_ - target)**2)
    L1 = huber_loss_delta * weight *(torch.abs(input_ - target) - 0.5 * huber_loss_delta)
    
    loss = torch.where(cond, L2, L1)
    
    return torch.mean(loss)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = Memory(capacity=BUFFER_SIZE, e=E, a=A, beta=BETA, beta_increment_per_sampling=BETA_INCREMENT_PER_SAMPLING,
                            batch_size=BATCH_SIZE)
        
        return None
    
    def append_sample(self, state, action, reward, next_state, done):
        """Calculate Error and append the tuple to
        Memory
        """
        
        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).float().to(device)
        next_state = torch.from_numpy(next_state).float().to(device)
        
        self.actor_target.eval()
        with torch.no_grad():
            actions_next = self.actor_target(next_state)
        self.actor_target.train()
        
        self.critic_target.eval()
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_state, actions_next)
        self.critic_target.train()
        
        # Compute Q targets for current states (y_i)
        Q_targets = reward + (GAMMA * Q_targets_next * (1 - done))
        
        # Compute critic loss
        self.critic_local.eval()
        with torch.no_grad():
            Q_expected = self.critic_local(state, action)
        self.critic_local.train()
            
        critic_loss = torch.abs(Q_expected - Q_targets).detach().numpy()
        
        # clip loss
        
        critic_loss = min(1.0, critic_loss)

        # add to Memory
        self.memory.add(critic_loss, (state, action, reward, next_state, done))
        
        return None  
        
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #self.memory.add(state, action, reward, next_state, done)
        self.append_sample(state, action, reward, next_state, done)
        
        self.update()
        
        return None
    
    def update(self):
        """Train Agent
        """
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones, idxs, is_weights = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        #critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss = weighted_mse_loss(Q_expected, Q_targets, is_weights)
        
        # update priority in memory
        errors = torch.abs(Q_expected - Q_targets).detach().numpy()
        
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            
            # clip errors
            
            self.memory.update(idx, min(1.0, errors[i]))
        # end of updating priority in memory
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)              

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
class Memory:
    
    """Prioritized Experience Replay Class,
    taken from https://github.com/rlcode/per/blob/master/prioritized_memory.py
    """
    
    # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, capacity, e, a, beta, beta_increment_per_sampling, batch_size):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = e
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.batch_size = batch_size

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self):
        
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        # min and max weights
        
        p_min = min(self.tree.tree) / self.tree.total()
                        
        max_weight = (p_min * self.tree.n_entries) ** (-self.beta)
        # end
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            # reduce s if p == 0
            if p == 0:
                (idx, p, data) = self.tree.get(s * 0.9)
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * np.asarray(sampling_probabilities), -self.beta)
        #is_weight = np.power(self.tree.n_entries * np.asarray(priorities), -self.beta)
        
        is_weight /= max_weight
        
        # set dimensions right
        is_weight = is_weight[:, np.newaxis]
        
        states = torch.from_numpy(np.vstack([b[0] for b in batch])).float().to(device)
        actions = torch.from_numpy(np.vstack([b[1] for b in batch])).float().to(device)
        rewards = torch.from_numpy(np.vstack([b[2] for b in batch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([b[3] for b in batch])).float().to(device)
        dones = torch.from_numpy(np.vstack([b[4] for b in batch]).astype(np.uint8)).float().to(device)
        is_weight = torch.from_numpy(is_weight).float().to(device)

        return (states, actions, rewards, next_states, dones, idxs, is_weight)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
    def __len__(self):
        
        return self.tree.n_entries
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)