import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from src.models.rl_models import Actor, Critic

# Define TD3 Agent
class TD3Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Actor and Critic Networks
        self.actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)

        self.critic_1 = Critic(state_size, action_size)
        self.critic_2 = Critic(state_size, action_size)
        self.target_critic_1 = Critic(state_size, action_size)
        self.target_critic_2 = Critic(state_size, action_size)

        # Synchronize target networks
        self._update_target_networks(tau=1.0)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=0.002)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=0.002)

        # Replay Buffer
        self.memory = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Target network update rate
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5    # Limit for absolute value of noise
        self.policy_delay = 2    # Policy update delay
        self.total_it = 0        # Total iterations for policy delay

    def _update_target_networks(self, tau=0.005):
        """Soft update target networks."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Select action using the actor network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze().numpy()
        
        return np.clip(action, -1., 1.)  # Ensure actions are within valid range

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0) if next_state is not None else None
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            reward_tensor = torch.FloatTensor([reward])
            done_tensor = torch.FloatTensor([done])

            # Add noise to target actions during critic update
            with torch.no_grad():
                noise = (torch.randn_like(action_tensor) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.target_actor(next_state_tensor) + noise).clamp(-1., 1.)

                # Compute target Q-value using both critics
                target_q1 = self.target_critic_1(next_state_tensor, next_action)
                target_q2 = self.target_critic_2(next_state_tensor, next_action)
                target_q_value = reward_tensor + (self.gamma * (1 - done_tensor) * torch.min(target_q1.squeeze(), target_q2.squeeze()))

            # Compute current Q-values using both critics
            current_q1 = self.critic_1(state_tensor, action_tensor)
            current_q2 = self.critic_2(state_tensor, action_tensor)

            # Compute critic losses
            critic_1_loss = nn.MSELoss()(current_q1.squeeze(), target_q_value.detach())
            critic_2_loss = nn.MSELoss()(current_q2.squeeze(), target_q_value.detach())

            # Update both critics
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            # Delayed policy update: Update actor and target networks less frequently
            if self.total_it % self.policy_delay == 0:
                actor_loss = -self.critic_1(state_tensor,
                                            self.actor(state_tensor)).mean()

                # Update actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update target networks
                self._update_target_networks()

            # Increment iteration counter
            self.total_it += 1
