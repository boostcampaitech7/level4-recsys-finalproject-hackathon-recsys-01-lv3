import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh()  # [-1, 1] 범위로 출력 제한
        )

    def forward(self, state):
        return self.model(state)
    
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Q-value 출력
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # 상태와 액션 결합
        return self.model(x)
    
    def compute_cql_loss(self, state, action, alpha=0.1):
        """
        Conservative Q-Learning (CQL) 손실 계산
        :param state: 상태 텐서
        :param action: 액션 텐서
        :param alpha: 보수성 정도를 조절하는 하이퍼파라미터
        :return: CQL 손실 값
        """
        # 랜덤 액션 샘플링
        sampled_actions = torch.rand((10, action.shape[-1]))  # 랜덤 액션 샘플링 (10개)
        logsumexp_q = torch.logsumexp(self.forward(state.repeat(10, 1), sampled_actions), dim=0)
        dataset_q = self.forward(state, action)
        cql_loss = logsumexp_q.mean() - dataset_q.mean()
        return alpha * cql_loss