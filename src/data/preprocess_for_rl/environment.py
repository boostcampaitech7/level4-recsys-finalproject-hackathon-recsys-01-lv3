import polars as pl
import numpy as np
from src.util.reward_normalizer import RewardNormalizer

class DynamicPricingEnv:
    def __init__(self, df: pl.DataFrame, interaction_matrix_csr, item_similarity, item_user_scores, true_users_by_product, tau=1):
        """
        Initialize the Dynamic Pricing Environment.
        :param df: Main dataset containing pricing and sales data.
        :param interaction_matrix_csr: Sparse user-item interaction matrix (CSR format).
        :param item_similarity: Precomputed item similarity matrix.
        :param tau: Time interval for comparing revenue conversion rates.
        """
        self.df = df
        self.interaction_matrix_csr = interaction_matrix_csr
        self.item_similarity = item_similarity  # Precomputed item similarity matrix
        self.tau = tau  # Time interval for comparing revenue conversion rates
        self.current_day_idx = None
        self.previous_rcr = None  # Revenue Conversion Rate at t-τ
        self.item_user_scores = item_user_scores
        self.true_users_by_product = true_users_by_product
        self.normalizer = RewardNormalizer()

        # Cache for recommendations to speed up repeated queries
        # self.recommendation_cache = {}

    def reset(self):
        """
        Reset the environment to the initial state.
        :return: Initial state as a NumPy array.
        """
        self.current_day_idx = 0
        self.previous_rcr = None  # Reset previous RCR
        state = self._get_state(self.current_day_idx)
        return state

    def _get_state(self, idx):
        """
        Extract the state features for a given index.
        :param idx: Index of the current day in the dataset.
        :return: State as a NumPy array [mean_price_per_day, purchase_count_per_day, product_id_index].
        """
        row = self.df.row(idx, named=True)  # Use named=True to get a dictionary
        state = np.array([row["mean_price_per_day"], row["purchase_count_per_day"], row["product_id_index"]])
        return state


    def step(self, actions):
        """
        Perform an action in the environment.
        :param actions: Tuple (price adjustment, top_k adjustment).
                        - actions[0]: Continuous price adjustment (e.g., percentage change).
                        - actions[1]: Top-k adjustment for recommendation precision.
        :return: next_state, reward, done.
                - next_state: The next state as a NumPy array.
                - reward: The calculated reward (Revenue + RCR + Precision).
                - done: Boolean indicating whether the episode has ended.
        """
        price_action, top_k_action = actions

        # 현재 상태 가져오기
        current_row = self.df.row(self.current_day_idx, named=True)
        current_price = current_row["mean_price_per_day"]
        current_demand = current_row["demand_per_day"]
        product_id = int(current_row["product_id_index"])

        # 이전 상태를 기반으로 가격 탄력성 계산
        if self.current_day_idx > 0:
            previous_row = self.df.row(self.current_day_idx - 1, named=True)
            previous_price = previous_row["mean_price_per_day"]
            previous_demand = previous_row["demand_per_day"]

            # 실제 데이터 기반 탄력성 계산
            price_change = (current_price - previous_price) / max(previous_price, 1e-6)
            demand_change = (current_demand - previous_demand) / max(previous_demand, 1e-6)

            if price_change != 0:
                new_elasticity = demand_change / price_change
            else:
                new_elasticity = self.price_elasticity  # 이전 값을 유지

            # 이동 평균을 사용한 탄력성 업데이트
            self.price_elasticity = 0.2 * new_elasticity + 0.8 * self.price_elasticity

        else:
            # 첫 번째 단계에서는 기본 탄력성 사용
            self.price_elasticity = -1.0

        # Action을 반영한 가격 조정
        next_price = max(0, current_price * (1 + price_action))

        # Action 수행 후 예상 수요량 계산 (탄력성 적용)
        next_demand = current_demand * (1 + price_action) ** self.price_elasticity

        # Top-K 추천 정확도 계산
        true_users = self.get_true_users(product_id)
        precision_at_k_value = self.precision_at_k(product_id, true_users, top_k=int(top_k_action))

        # 에피소드 종료 여부 확인
        self.current_day_idx += 1
        done = (self.current_day_idx >= len(self.df))

        if done:
            return None, 0, done

        # 다음 상태 가져오기
        next_row = self.df.row(self.current_day_idx, named=True)
        next_unique_visitors = max(1, next_row["unique_visitors"])

        # 수익 계산
        revenue = next_price * next_demand

        # RCR 계산 및 변화량 (Unique Visitor에 대한 예측값 설정 어려움 일단 사용 안함)
        # current_rcr = revenue / next_unique_visitors
        # rcr_change = (current_rcr - self.previous_rcr) if self.previous_rcr is not None else 0

        # 마케팅 비용 계산
        marketing_cost = 1 * int(top_k_action)  # 단위 비용 * Top-K 수

        # 추천 모델 관련 보상
        recsys_value = (next_price * precision_at_k_value * int(top_k_action) * next_demand) - marketing_cost

        # 최종 보상 계산
        reward = max(0.5 * recsys_value + 0.5 * revenue, 0)

        # 이전 RCR 업데이트
        # self.previous_rcr = current_rcr

        # 다음 상태로 이동
        next_state = self._get_state(self.current_day_idx)

        return next_state, reward, done


    def get_true_users(self, product_id):
        """
        주어진 product_id에 대해 상호작용한 실제 사용자 리스트 반환.
        :param product_id: 상품 ID.
        :return: 해당 상품과 상호작용한 사용자 ID 리스트.
        """
        return self.true_users_by_product.get(product_id, [])
        
    def recommend_users(self, product_id, top_k):
        """
        주어진 상품에 대해 추천 사용자 리스트 반환 (미리 계산된 점수 사용).
        :param product_id: 추천 대상 상품 ID.
        :param top_k: 추천할 사용자 수.
        :return: 추천 사용자 ID 리스트.
        """
        if product_id not in self.item_user_scores:
            return []

        # Precomputed user scores에서 top_k 사용자 반환 (slicing)
        return self.item_user_scores[product_id][:top_k]


    def precision_at_k(self, product_id, true_users, top_k):
        """
        Calculate Precision@k using the precomputed item similarity matrix.
        :param product_id: The product ID for which to calculate precision.
        :param true_users: List of true users who interacted with the product.
        :param top_k: Number of top recommendations to consider.
        :return: Precision@k value.
        """
        # Get recommended users for the given product_id
        recommended_users = self.recommend_users(product_id, top_k=top_k)

        # Calculate Precision: |relevant ∩ recommended| / |recommended|
        relevant_and_recommended = len(set(recommended_users) & set(true_users))
        precision = relevant_and_recommended / len(recommended_users) if len(recommended_users) > 0 else 0

        return precision