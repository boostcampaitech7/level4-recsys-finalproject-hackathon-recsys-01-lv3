import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Actor network for policy approximation in reinforcement learning.

    This network outputs actions in the range [-1, 1] using a Tanh activation function.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the Actor network.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
        """
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output action tensor in the range [-1, 1].
        """
        return self.net(state)


class Critic(nn.Module):
    """
    Critic network for value function approximation in reinforcement learning.

    This network estimates the Q-value for a given state-action pair.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the Critic network.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the input action.
        """
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.

        Returns:
            torch.Tensor: Estimated Q-value for the given state-action pair.
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)