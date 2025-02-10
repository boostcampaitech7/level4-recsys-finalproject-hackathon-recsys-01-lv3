import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.models.rl_models import Actor, Critic
from src.utils.replay_buffer import ReplayBuffer

class HRLAgent:
    """
    Hierarchical Reinforcement Learning (HRL) Agent.

    This agent uses a hierarchical structure with:
        - A high-level policy (DQN) for price-related actions.
        - A low-level policy (TD3) for user engagement actions.

    Args:
        env: The environment the agent interacts with.
        config: Configuration dictionary containing hyperparameters.
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = config["general"]["device"]
        self.batch_size = config["general"]["batch_size"]

        state_dim = env.df.shape[1]
        high_action_dim = len(env.price_action_space)
        low_action_dim = 1

        self.high_policy = Actor(state_dim=state_dim, action_dim=high_action_dim).to(self.device)
        self.high_optimizer = optim.Adam(self.high_policy.parameters(), lr=config["high_policy"]["learning_rate"])

        self.low_actor = Actor(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)
        self.low_critic_1 = Critic(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)
        self.low_critic_2 = Critic(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)

        self.target_low_actor = Actor(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)
        self.target_low_critic_1 = Critic(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)
        self.target_low_critic_2 = Critic(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)

        self._initialize_target_networks()

        self.low_actor_optimizer = optim.Adam(self.low_actor.parameters(), lr=config["low_policy"]["learning_rate"])
        self.low_critic_optimizer_1 = optim.Adam(self.low_critic_1.parameters(), lr=config["low_policy"]["learning_rate"])
        self.low_critic_optimizer_2 = optim.Adam(self.low_critic_2.parameters(), lr=config["low_policy"]["learning_rate"])

        self.replay_buffer = ReplayBuffer(max_size=config["replay_buffer"]["size"])

        self.gamma = config["general"]["gamma"]
        self.tau = config["general"]["tau"]
        self.noise_scale = config["low_policy"]["noise_scale_start"]
        self.noise_min = config["low_policy"]["noise_min"]
        self.epsilon = config["high_policy"]["epsilon_start"]
        self.epsilon_min = config["high_policy"]["epsilon_min"]
        self.decay = config["high_policy"]["epsilon_decay"]

        self.train_step = 0

    def _initialize_target_networks(self):
        """
        Initialize target networks with the same weights as the main networks.
        """
        self.target_low_actor.load_state_dict(self.low_actor.state_dict())
        self.target_low_critic_1.load_state_dict(self.low_critic_1.state_dict())
        self.target_low_critic_2.load_state_dict(self.low_critic_2.state_dict())

    def _soft_update(self, source_net, target_net, tau):
        """
        Perform a soft update of the target network parameters.

        Args:
            source_net: Source network whose parameters will be used.
            target_net: Target network to be updated.
            tau: Soft update factor (0 < tau <= 1).
        
        Formula:
            target_param := tau * source_param + (1 - tau) * target_param
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def select_high_action(self, state):
        """
        Select an action from the high-level policy (DQN).

        Args:
            state: Current state of the environment.

        Returns:
            int: Selected action index.
            
            Exploration vs Exploitation:
                - Random action with probability epsilon (exploration).
                - Greedy action based on Q-values (exploitation).
        """
        if np.random.rand() < self.epsilon:
            return self.env.sample_price_action()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.high_policy(state_tensor)  
                return torch.argmax(q_values).item()     

    def select_low_action(self, state_with_price):
        """
        Select an action from the low-level policy (TD3).

        Args:
            state_with_price (np.ndarray): Current state concatenated with the high-level price action.

        Returns:
            int: Scaled and clipped continuous action in the range [1, 10000].
        """
        state_tensor = torch.FloatTensor(state_with_price).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_continuous = self.low_actor(state_tensor).cpu().numpy()

        noise = np.random.normal(0, self.noise_scale, size=action_continuous.shape)
        action_noisy = action_continuous + noise

        scaled_action = (action_noisy[0][0] + 1) / 2 * (10000 - 1) + 1

        return int(np.clip(scaled_action, 1, 10000))
    
    def train_high_policy(self):
        """
        Train the high-level policy (DQN).

        This method updates the high-level policy using sampled experiences from the replay buffer.
        It calculates the Q-value loss and performs a gradient update.

        Returns:
            float: The loss value for the high-level policy.
        """
        if self.replay_buffer.size_high() < self.batch_size:
            return None

        states, price_actions, rewards, next_states = self.replay_buffer.sample_high(self.batch_size)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(price_actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)

        with torch.no_grad():
            next_q_values = self.high_policy(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards_tensor + self.gamma * max_next_q_values

        current_q_values = self.high_policy(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

        self.high_optimizer.zero_grad()
        loss.backward()
        self.high_optimizer.step()

        return loss.item()

    def train_low_policy(self, policy_update_delay=2):
        """
        Train the low-level policy (TD3).

        This method updates both the critic and actor networks of the low-level policy.
        Critic networks are updated every step, while the actor network and target networks are updated periodically.

        Args:
            policy_update_delay (int): Number of steps between actor updates.

        Returns:
            tuple: Critic losses (critic_loss_1, critic_loss_2) and actor loss (if updated).
        """
        if self.replay_buffer.size_low() < self.batch_size:
            return None, None

        states, actions, rewards, next_states = self.replay_buffer.sample_low(self.batch_size)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)

        with torch.no_grad():
            next_actions = self.target_low_actor(next_states_tensor)
            target_q1 = self.target_low_critic_1(next_states_tensor, next_actions)
            target_q2 = self.target_low_critic_2(next_states_tensor, next_actions)
            target_q_min = torch.min(target_q1, target_q2)
            q_target = rewards_tensor + self.gamma * target_q_min

        q1_predicted = self.low_critic_1(states_tensor, actions_tensor)
        q2_predicted = self.low_critic_2(states_tensor, actions_tensor)

        critic_loss_1 = nn.SmoothL1Loss()(q1_predicted, q_target.detach())
        critic_loss_2 = nn.SmoothL1Loss()(q2_predicted, q_target.detach())

        self.low_critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.low_critic_optimizer_1.step()

        self.low_critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.low_critic_optimizer_2.step()

        actor_loss = None
        if self.train_step % policy_update_delay == 0:
            sampled_actions = self.low_actor(states_tensor)
            actor_loss = -self.low_critic_1(states_tensor, sampled_actions).mean()

            self.low_actor_optimizer.zero_grad()
            actor_loss.backward()
            self.low_actor_optimizer.step()

            self._soft_update(self.low_actor, self.target_low_actor, self.tau)
            self._soft_update(self.low_critic_1, self.target_low_critic_1, self.tau)
            self._soft_update(self.low_critic_2, self.target_low_critic_2, self.tau)

        self.train_step += 1

        return (critic_loss_1.item(), critic_loss_2.item()), actor_loss.item() if actor_loss is not None else None
    
    def train(self, num_episodes=1000, warm_up=10):
        """
        Train the HRL agent using a hierarchical learning approach.

        Args:
            num_episodes (int): Number of episodes to train the agent.
            warm_up (int): Number of initial episodes with random actions for exploration.

        This method trains both the high-level policy (DQN) and the low-level policy (TD3).
        It also tracks performance metrics and periodically saves the best-performing models.
        """
        best_reward = -float('inf')
        save_path = self.config["training"]["save_path"]
        os.makedirs(save_path, exist_ok=True)

        reward_history = []
        high_policy_loss_history = []
        low_policy_critic_loss_history = []
        low_policy_actor_loss_history = []

        start_time = time.time()

        for episode in range(num_episodes):
            episode_start_time = time.time()
            state_high = self.env.reset()
            done = False
            episode_reward = 0
            step_count = 0

            price_action_idx = self.select_high_action(state_high)
            price_action = self.env.price_action_space[price_action_idx]

            while not done:
                state_low = np.concatenate([state_high, [price_action]])

                if episode < warm_up:
                    top_k_action = self.env.sample_top_k_action()
                else:
                    top_k_action = self.select_low_action(state_low)

                next_state_env, reward, done = self.env.step((price_action, top_k_action))
                episode_reward += reward

                next_state_low = np.concatenate([
                    next_state_env if next_state_env is not None else np.zeros_like(state_high),
                    [price_action]
                ])

                if 1 < top_k_action < 10000:
                    self.replay_buffer.add_high(
                        state_high, price_action_idx, reward,
                        next_state_env if next_state_env is not None else np.zeros_like(state_high)
                    )
                    self.replay_buffer.add_low(
                        state_low, top_k_action / 10000.0, reward, next_state_low
                    )

                if len(self.replay_buffer.buffer_low) >= self.batch_size:
                    low_critic_losses, low_actor_loss = self.train_low_policy()
                    
                    if low_critic_losses is not None:
                        low_policy_critic_loss_history.append(low_critic_losses[0])
                        low_policy_critic_loss_history.append(low_critic_losses[1])
                    
                    if low_actor_loss is not None:
                        low_policy_actor_loss_history.append(low_actor_loss)

                step_count += 1
                if step_count % 100 == 0:
                    price_action_idx = self.select_high_action(next_state_env)
                    price_action = self.env.price_action_space[price_action_idx]

                state_high = next_state_env

            if len(self.replay_buffer.buffer_high) >= self.batch_size:
                high_loss = self.train_high_policy()
                if high_loss is not None:
                    high_policy_loss_history.append(high_loss)

            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"New best reward: {best_reward:.2f}. Saving models...")
                self._save_models(save_path)

            if episode >= warm_up:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
                self.noise_scale = max(self.noise_min, self.noise_scale * self.decay)

            reward_history.append(episode_reward)
            self._log_progress(episode, num_episodes, episode_reward, start_time, episode_start_time)

    def _save_models(self, save_path):
        """
        Save the current models to the specified path.

        Args:
            save_path (str): Directory path to save the models.
        """
        torch.save(self.high_policy.state_dict(), os.path.join(save_path, 'best_high_policy.pth'))
        torch.save(self.low_actor.state_dict(), os.path.join(save_path, 'best_low_actor.pth'))
        torch.save(self.low_critic_1.state_dict(), os.path.join(save_path, 'best_low_critic_1.pth'))
        torch.save(self.low_critic_2.state_dict(), os.path.join(save_path, 'best_low_critic_2.pth'))

    def _log_progress(self, episode, num_episodes, episode_reward, start_time, episode_start_time):
        """
        Log training progress and estimated remaining time.

        Args:
            episode (int): Current episode number.
            num_episodes (int): Total number of episodes.
            episode_reward (float): Reward obtained in the current episode.
            start_time (float): Start time of training.
            episode_start_time (float): Start time of the current episode.
        """
        episode_duration = time.time() - episode_start_time
        total_elapsed_time = time.time() - start_time
        estimated_total_time = (total_elapsed_time / (episode + 1)) * num_episodes
        remaining_time = estimated_total_time - total_elapsed_time

        progress_percentage = ((episode + 1) / num_episodes) * 100

        print(
            f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward:.2f}, "
            f"Epsilon: {self.epsilon:.4f}, "
            f"Episode Duration: {episode_duration:.2f}s, "
            f"Progress: {progress_percentage:.2f}%, "
            f"Remaining Time: {remaining_time / 60:.2f} minutes"
        )
