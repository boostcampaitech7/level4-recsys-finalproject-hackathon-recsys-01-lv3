import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.rl_models import Actor, Critic
from src.utils.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


# HRL 에이전트 정의
class HRLAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = config["general"]["device"]
        self.batch_size = config["general"]["batch_size"]

        # 환경 차원 정의
        state_dim = env.df.shape[1]
        high_action_dim = len(env.price_action_space)
        low_action_dim = 1

        # High-Level Policy (DQN)
        self.high_policy = Actor(state_dim=state_dim, action_dim=high_action_dim).to(self.device)
        self.high_optimizer = optim.Adam(self.high_policy.parameters(), lr=config["high_policy"]["learning_rate"])

        # Low-Level Policy (TD3)
        self.low_actor = Actor(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)
        self.low_critic_1 = Critic(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)
        self.low_critic_2 = Critic(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)

        # Target Networks for TD3
        self.target_low_actor = Actor(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)
        self.target_low_critic_1 = Critic(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)
        self.target_low_critic_2 = Critic(state_dim=state_dim + 1, action_dim=low_action_dim).to(self.device)

        # Target Networks for TD3 초기화
        self.target_low_actor.load_state_dict(self.low_actor.state_dict())
        self.target_low_critic_1.load_state_dict(self.low_critic_1.state_dict())
        self.target_low_critic_2.load_state_dict(self.low_critic_2.state_dict())

        # Optimizers for Low-Level Policy
        self.low_actor_optimizer = optim.Adam(self.low_actor.parameters(), lr=config["low_policy"]["learning_rate"])
        self.low_critic_optimizer_1 = optim.Adam(self.low_critic_1.parameters(), lr=config["low_policy"]["learning_rate"])
        self.low_critic_optimizer_2 = optim.Adam(self.low_critic_2.parameters(), lr=config["low_policy"]["learning_rate"])

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(max_size=config["replay_buffer"]["size"])

        # Hyperparameters from config
        self.gamma = config["general"]["gamma"]
        self.tau = config["general"]["tau"]
        self.noise_scale = config["low_policy"]["noise_scale_start"]
        self.noise_min = config["low_policy"]["noise_min"]
        self.epsilon = config["high_policy"]["epsilon_start"]
        self.epsilon_min = config["high_policy"]["epsilon_min"]
        self.decay = config["high_policy"]["epsilon_decay"]


        # 학습 스텝 초기화 (TD3에서 Actor 지연 업데이트용)
        self.train_step = 0

    def _soft_update(self, source_net, target_net, tau):
        """소프트 업데이트."""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def select_high_action(self, state):
        """High-Level Policy에서 액션 선택."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.env.price_action_space))  # 탐험: 랜덤 액션 선택
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.high_policy(state_tensor)  # Q값 계산
                return torch.argmax(q_values).item()       # 최대 Q값의 액션 선택

    def select_low_action(self, state_with_price):
        """Low-Level Policy에서 액션 선택."""
        state_tensor = torch.FloatTensor(state_with_price).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Actor 네트워크를 통해 연속형 행동 생성
            action_continuous = self.low_actor(state_tensor).detach().cpu().numpy()
        # 탐험을 위한 가우시안 노이즈 추가
        noise = np.random.normal(0, self.noise_scale, size=action_continuous.shape)
        action_noisy = action_continuous + noise
        # 행동을 [1, 100000] 범위로 스케일링하고 정수화
        return int(np.clip(action_noisy[0][0] * 100000.0, 1, 100000))

    def train_high_policy(self):
        """High-Level Policy 학습 (DQN 사용)."""
        # 상위 정책 버퍼 크기 확인
        if self.replay_buffer.size_high() < self.batch_size:
            return

        # Replay Buffer에서 배치 샘플링 (상위 정책 데이터)
        states, price_actions, rewards, next_states = self.replay_buffer.sample_high(self.batch_size)

        # 텐서 변환 및 디바이스 이동
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(price_actions).to(self.device)  # High-level actions만 사용
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)

        # Target Q값 계산
        with torch.no_grad():
            next_q_values = self.high_policy(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards_tensor + self.gamma * max_next_q_values

        # 현재 Q값 계산
        current_q_values = self.high_policy(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        # 손실 계산 및 네트워크 업데이트
        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.high_optimizer.zero_grad()
        loss.backward()
        self.high_optimizer.step()

        return loss.item()
    
    def train_low_policy(self, policy_update_delay=2):
        """Low-Level Policy 학습 (TD3 사용)."""
        # 하위 정책 버퍼 크기 확인
        if self.replay_buffer.size_low() < self.batch_size:
            return

        # Replay Buffer에서 배치 샘플링 (하위 정책 데이터)
        states, actions, rewards, next_states = self.replay_buffer.sample_low(self.batch_size)

        # 텐서 변환 및 디바이스 이동
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)

        # Critic 네트워크 업데이트
        with torch.no_grad():
            next_actions = self.target_low_actor(next_states_tensor)
            target_q1 = self.target_low_critic_1(next_states_tensor, next_actions)
            target_q2 = self.target_low_critic_2(next_states_tensor, next_actions)
            target_q_min = torch.min(target_q1, target_q2)
            q_target = rewards_tensor + self.gamma * target_q_min

        q1_predicted = self.low_critic_1(states_tensor, actions_tensor)
        q2_predicted = self.low_critic_2(states_tensor, actions_tensor)

        critic_loss_1 = nn.MSELoss()(q1_predicted, q_target.detach())
        critic_loss_2 = nn.MSELoss()(q2_predicted, q_target.detach())

        self.low_critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.low_critic_optimizer_1.step()

        self.low_critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.low_critic_optimizer_2.step()

        # Actor 네트워크 지연 업데이트
        actor_loss = None
        if self.train_step % policy_update_delay == 0:
            sampled_actions = self.low_actor(states_tensor)
            actor_loss = -self.low_critic_1(states_tensor, sampled_actions).mean()

            self.low_actor_optimizer.zero_grad()
            actor_loss.backward()
            self.low_actor_optimizer.step()

            # Target 네트워크 소프트 업데이트
            self._soft_update(self.low_actor, self.target_low_actor, self.tau)
            self._soft_update(self.low_critic_1, self.target_low_critic_1, self.tau)
            self._soft_update(self.low_critic_2, self.target_low_critic_2, self.tau)

        # 학습 스텝 증가
        self.train_step += 1

        return (critic_loss_1.item(), critic_loss_2.item()), actor_loss


    def train(self, num_epochs=1000):
        """
        HRL 에이전트 학습 루프.
        
        Args:
            num_epochs (int): 학습 에포크 수.
        """
        best_reward = -float('inf')

        # 저장 경로 설정 (config에서 가져오기)
        save_path = self.config["training"]["save_path"]
        import os
        os.makedirs(save_path, exist_ok=True)  # 경로가 없으면 생성

        reward_history = []
        high_policy_loss_history = []
        low_policy_critic_loss_history = []
        low_policy_actor_loss_history = []

        plt.ion()  # Interactive 모드 활성화
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        for epoch in range(num_epochs):
            state_high = self.env.reset()  # 초기 상태
            done = False
            episode_reward = 0
            step_count = 0  # 스텝 카운터 초기화

            # 초기 High-Level Policy 액션 선택
            price_action_idx = self.select_high_action(state_high)
            price_action = self.env.price_action_space[price_action_idx]

            first_sample = 1

            while not done:
                # Low-Level Policy용 상태 생성
                state_low = np.concatenate([state_high, [price_action]])

                # Low-Level Policy로 추천 사용자 수(top_k) 액션 선택
                top_k_action = self.select_low_action(state_low)

                if first_sample == 1 and epoch == 0:
                    print(top_k_action)
                    first_sample -= 1

                # 환경과 상호작용하여 다음 상태 및 보상 획득
                next_state_env, reward, done = self.env.step((price_action, top_k_action))
                episode_reward += reward

                # Low-Level Policy의 다음 상태 생성
                next_state_low = np.concatenate([
                    next_state_env if next_state_env is not None else np.zeros_like(state_high),
                    [price_action]
                ])

                # Replay Buffer에 경험 저장
                if 1 < top_k_action < 100000:
                    self.replay_buffer.add_high(state_high, price_action_idx, reward / 1000.0,
                                                next_state_env if next_state_env is not None else np.zeros_like(state_high))
                    self.replay_buffer.add_low(state_low, top_k_action / 100000.0,
                                                reward / 1000.0, next_state_low)

                # Low-Level Policy 학습
                if len(self.replay_buffer.buffer_low) >= self.batch_size:
                    (low_critic_losses, low_actor_loss) = self.train_low_policy()
                
                    if low_critic_losses is not None:
                        low_policy_critic_loss_history.append(low_critic_losses[0])  # Critic1 Loss 기록
                        low_policy_critic_loss_history.append(low_critic_losses[1])  # Critic2 Loss 기록
                    
                    if low_actor_loss is not None:
                        low_policy_actor_loss_history.append(low_actor_loss)  # Actor Loss 기록

                # 주기적 High-Level Policy 갱신
                step_count += 1
                if step_count % 100 == 0:
                    price_action_idx = self.select_high_action(next_state_env)
                    price_action = self.env.price_action_space[price_action_idx]
                    

                state_high = next_state_env

            # High-Level Policy 학습
            if len(self.replay_buffer.buffer_high) >= self.batch_size:
                high_loss = self.train_high_policy()
                if high_loss is not None:
                    high_policy_loss_history.append(high_loss)

            # 최고 보상 갱신 및 모델 저장
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"New best reward: {best_reward:.2f}. Saving models...")

                # 상위 정책 모델 저장
                torch.save(self.high_policy.state_dict(), os.path.join(save_path, 'best_high_policy.pth'))

                # 하위 정책 모델 저장 (Actor와 Critic 모두)
                torch.save(self.low_actor.state_dict(), os.path.join(save_path, 'best_low_actor.pth'))
                torch.save(self.low_critic_1.state_dict(), os.path.join(save_path, 'best_low_critic_1.pth'))
                torch.save(self.low_critic_2.state_dict(), os.path.join(save_path, 'best_low_critic_2.pth'))

            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
            self.noise_scale = max(self.noise_min, self.noise_scale * self.decay)

            reward_history.append(episode_reward)

            print(f"Epoch {epoch + 1}/{num_epochs} - Total Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.4f}")

            # 실시간 그래프 업데이트
            ax[0].clear()
            ax[0].plot(reward_history, label="Reward")
            ax[0].set_title("Reward History")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Reward")
            ax[0].legend()

            ax[1].clear()
            ax[1].plot(high_policy_loss_history, label="High-Level Loss")
            ax[1].set_title("High-Level Loss History")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Loss")
            ax[1].legend()

            ax[2].clear()
            ax[2].plot(low_policy_actor_loss_history, label="Low-Level Actor Loss")
            ax[2].plot(low_policy_critic_loss_history, label="Low-Level Critic Loss")
            ax[2].set_title("Low-Level Loss History")
            ax[2].set_xlabel("Epoch")
            ax[2].set_ylabel("Loss")
            ax[2].legend()

            plt.pause(0.01)  # 그래프 업데이트 대기 시간

        plt.ioff()  # Interactive 모드 종료
        plt.show()