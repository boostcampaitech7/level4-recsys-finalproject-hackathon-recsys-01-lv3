import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
from src.models.rl_models import Actor, Critic
from src.utils.add_noise import OUNoise

class DDPGAgentWithCQL:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Actor 및 Critic 네트워크 생성
        self.actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        
        self.critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        
        # Target 네트워크 초기화 (동기화)
        self._update_target_networks(tau=1.0)

        # Optimizer 설정
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)

        # 경험 리플레이 버퍼 설정
        self.memory = deque(maxlen=10000)
        
        # Hyperparameters
        self.gamma = 0.99  # 할인율
        self.tau = 0.005   # Target 네트워크 업데이트 비율
        self.alpha = 0.1   # CQL 보수성 하이퍼파라미터
        
        # OU 노이즈 초기화
        self.noise = OUNoise(action_dim=self.action_size)

        
    def _update_target_networks(self, tau=0.005):
        """Soft update target networks."""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
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
                target_action_next = (
                    self.target_actor(next_state_tensor) if next_state_tensor is not None else torch.zeros((1, self.action_size))
                )
                target_q_next = (
                    self.target_critic(next_state_tensor, target_action_next)
                    if next_state_tensor is not None else torch.tensor(0.0)
                )
                target_q_value = reward_tensor + (self.gamma * target_q_next * (1 - done_tensor))

            q_value_predicted = self.critic(state_tensor, action_tensor)

            # Bellman Loss 계산
            bellman_loss = nn.MSELoss()(q_value_predicted.squeeze(), target_q_value.squeeze())

            # CQL Loss 계산 및 Critic 업데이트
            cql_loss = self.critic.compute_cql_loss(state_tensor, action_tensor, alpha=self.alpha)
            
            critic_loss = bellman_loss - cql_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor 업데이트
            actor_loss = -self.critic(state_tensor, self.actor(state_tensor)).mean()
            
            # Loss Update 디버깅

            # print("Critic Loss:", critic_loss.item())
            # print("Actor Loss:", actor_loss.item())
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

    def scale_action(self, value, min_val=1, max_val=3000):
        """
        Tanh 출력값을 [min_val, max_val] 범위로 스케일링.
        :param value: Actor 네트워크의 출력값 (범위: [-1, 1])
        :param min_val: 변환 후 최소값
        :param max_val: 변환 후 최대값
        :return: 스케일링된 값
        """
        return min_val + (max_val - min_val) * (value + 1) / 2
        

    def act(self, state):
        """
        Actor 네트워크를 사용하여 행동 선택 (OU 노이즈 기반 탐험).
        :param state: 현재 상태 벡터
        :param episode: 현재 에피소드 번호
        :return: price_action, top_k_action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            actions = self.actor(state_tensor).squeeze().numpy()
            
        # OU 노이즈 추가
        noise = self.noise.sample()
        noisy_actions = actions + noise

        scale_value = self.scale_action(value=noisy_actions[1], min_val=5, max_val=6000)

        # 액션 범위 제한
        price_action = np.clip(noisy_actions[0], -1, 1)  # 가격 조정 [-1 ~ 1]
        top_k_action = int(np.clip(scale_value, 5, 6000))  # 추천 유저 수 [5 ~ 3000]

        return price_action, top_k_action