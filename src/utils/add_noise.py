import numpy as np

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.2, sigma=0.3, sigma_min=0.05, decay_rate=0.99):
        """
        OU 노이즈 초기화.
        :param action_dim: 액션 차원 크기.
        :param mu: 평균값 (기본값: 0).
        :param theta: 평균 복귀 정도 (기본값: 0.2).
        :param sigma: 초기 노이즈 크기 (기본값: 0.3).
        :param sigma_min: 최소 노이즈 크기 (기본값: 0.05).
        :param decay_rate: 에피소드마다 sigma를 감소시키는 비율 (기본값: 0.99).
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay_rate = decay_rate
        self.reset()

    def reset(self):
        """노이즈 상태 초기화."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        """OU 노이즈 샘플링."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

    def decay(self):
        """노이즈 크기를 감소시킴."""
        self.sigma = max(self.sigma_min, self.sigma * self.decay_rate)

class GaussianNoise:
    def __init__(self, action_dim, mean=0.0, std_dev=0.2):
        """
        Gaussian 노이즈 초기화.
        :param action_dim: 액션 차원 크기.
        :param mean: 평균값 (기본값: 0.0).
        :param std_dev: 표준 편차 (기본값: 0.2).
        """
        self.action_dim = action_dim
        self.mean = mean
        self.std_dev = std_dev

    def sample(self):
        """
        Gaussian 노이즈 샘플링.
        :return: 지정된 차원(action_dim)에서 샘플링된 노이즈.
        """
        return np.random.normal(self.mean, self.std_dev, self.action_dim)
