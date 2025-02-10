from collections import deque
import numpy as np
import random

class ReplayBuffer:
    """
    A class to store and sample experiences for reinforcement learning.

    This buffer maintains separate buffers for high-level and low-level policies.
    """

    def __init__(self, max_size=100000):
        """
        Initialize the ReplayBuffer with a maximum size.

        Args:
            max_size (int): Maximum number of experiences to store in each buffer.
        """
        self.buffer_high = deque(maxlen=max_size)
        self.buffer_low = deque(maxlen=max_size)

    def add_high(self, state_high, price_action, reward_high, next_state_high):
        """
        Add a new experience to the high-level policy buffer.

        Args:
            state_high (np.ndarray): Current state for the high-level policy.
            price_action (int): Action taken by the high-level policy.
            reward_high (float): Reward received from the high-level policy.
            next_state_high (np.ndarray): Next state for the high-level policy.
        """
        self.buffer_high.append((state_high, price_action, reward_high, next_state_high))

    def add_low(self, state_low, top_k_action, reward_low, next_state_low):
        """
        Add a new experience to the low-level policy buffer.

        Args:
            state_low (np.ndarray): Current state for the low-level policy.
            top_k_action (float): Action taken by the low-level policy.
            reward_low (float): Reward received from the low-level policy.
            next_state_low (np.ndarray): Next state for the low-level policy.
        """
        self.buffer_low.append((state_low, top_k_action, reward_low, next_state_low))

    def sample_high(self, batch_size):
        """
        Sample a batch of experiences from the high-level policy buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, and next states.
                   Each element is a NumPy array.
        """
        batch = random.sample(self.buffer_high, batch_size)
        states, price_actions, rewards, next_states = zip(*batch)
        return (
            np.array(states),
            np.array(price_actions),
            np.array(rewards),
            np.array(next_states),
        )

    def sample_low(self, batch_size):
        """
        Sample a batch of experiences from the low-level policy buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, and next states.
                   Each element is a NumPy array.
        """
        batch = random.sample(self.buffer_low, batch_size)
        states, top_k_actions, rewards, next_states = zip(*batch)
        return (
            np.array(states),
            np.array(top_k_actions),
            np.array(rewards),
            np.array(next_states),
        )

    def size_high(self):
        """
        Get the current size of the high-level policy buffer.

        Returns:
            int: Number of experiences in the high-level buffer.
        """
        return len(self.buffer_high)

    def size_low(self):
        """
        Get the current size of the low-level policy buffer.

        Returns:
            int: Number of experiences in the low-level buffer.
        """
        return len(self.buffer_low)
