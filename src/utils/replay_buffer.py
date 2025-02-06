from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=100000):
        # Separate buffers for high-level and low-level policies
        self.buffer_high = deque(maxlen=max_size)
        self.buffer_low = deque(maxlen=max_size)

    def add_high(self, state_high, price_action, reward_high, next_state_high):
        """Add a new experience for the high-level policy."""
        self.buffer_high.append((state_high, price_action, reward_high, next_state_high))

    def add_low(self, state_low, top_k_action, reward_low, next_state_low):
        """Add a new experience for the low-level policy."""
        self.buffer_low.append((state_low, top_k_action, reward_low, next_state_low))

    def sample_high(self, batch_size):
        """Sample a batch of experiences for the high-level policy."""
        batch = random.sample(self.buffer_high, batch_size)
        
        # Unpack into separate components
        states, price_actions, rewards, next_states = zip(*batch)
        return (
            np.array(states),
            np.array(price_actions),
            np.array(rewards),
            np.array(next_states),
        )

    def sample_low(self, batch_size):
        """Sample a batch of experiences for the low-level policy."""
        batch = random.sample(self.buffer_low, batch_size)
        
        # Unpack into separate components
        states, top_k_actions, rewards, next_states = zip(*batch)
        return (
            np.array(states),
            np.array(top_k_actions),
            np.array(rewards),
            np.array(next_states),
        )

    def size_high(self):
        """Return the size of the high-level buffer."""
        return len(self.buffer_high)

    def size_low(self):
        """Return the size of the low-level buffer."""
        return len(self.buffer_low)