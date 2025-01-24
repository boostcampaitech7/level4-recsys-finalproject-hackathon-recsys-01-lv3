import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
from src.models.rl_models import Actor, Critic


class SACAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Actor 및 Critic 네트워크 생성
        self.actor = Actor(state_size, action_size)
        self.critic_1 = Critic(state_size, action_size)
        self.critic_2 = Critic(state_size, action_size)

        # Target Critic 네트워크 생성
        self.target_critic_1 = Critic(state_size, action_size)
        self.target_critic_2 = Critic(state_size, action_size)

        # Target 네트워크 초기화 (동기화)
        self._update_target_networks(tau=1.0)

        # Optimizer 설정
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=0.002)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=0.002)

        # 경험 리플레이 버퍼 설정
        self.memory = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.99  # 할인율
        self.tau = 0.005   # Target 네트워크 업데이트 비율
        self.alpha = 0.2   # 엔트로피 보상 계수

    def _update_target_networks(self, tau=0.005):
        """Soft update target networks."""
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action, reward, next_state, done))

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

            with torch.no_grad():
                # 다음 상태에서의 행동 샘플링
                next_action = self.actor(next_state_tensor)
                next_q1 = self.target_critic_1(next_state_tensor, next_action)
                next_q2 = self.target_critic_2(next_state_tensor, next_action)
                next_q_value = reward_tensor + self.gamma * (1 - done_tensor) * torch.min(next_q1, next_q2)

            # 현재 상태에서의 Q-value 계산
            q1 = self.critic_1(state_tensor, action_tensor)
            q2 = self.critic_2(state_tensor, action_tensor)

            # Critic 손실 계산
            critic_1_loss = torch.nn.MSELoss()(q1.squeeze(), next_q_value.squeeze())
            critic_2_loss = torch.nn.MSELoss()(q2.squeeze(), next_q_value.squeeze())

            # Critic 업데이트
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            # Actor 손실 계산 및 업데이트 (엔트로피 보상 포함)
            predicted_action = self.actor(state_tensor)
            actor_loss = -self.critic_1(state_tensor,
                                        predicted_action).mean() + self.alpha * (predicted_action ** 2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

    def act(self, state):
        """Actor 네트워크를 사용하여 행동 선택."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze().numpy()

        return np.clip(action[0], -1., 1.), int(np.clip(action[1], 5., 3000))
