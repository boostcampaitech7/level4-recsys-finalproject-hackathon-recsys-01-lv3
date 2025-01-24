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

        # Price adjustment logic
        current_row = self.df.row(self.current_day_idx, named=True)
        current_price = current_row["mean_price_per_day"]
        
        next_price = max(0, current_price * (1 + price_action))
                    
        self.current_day_idx += 1
        done = (self.current_day_idx >= len(self.df))
        
        # Extract true_users for the current product_id from interaction_matrix_csr
        product_id = int(current_row["product_id_index"])
        
        true_users = self.get_true_users(product_id)

        # Precision calculation using cached recommendations
        precision_at_k_value = self.precision_at_k(product_id, true_users, top_k=int(top_k_action))
        
        if done:
            return None, 0, done
        
        # Marketing Cost per Man
        cost = 0.1
        
        next_row = self.df.row(self.current_day_idx, named=True)
        
        # Revenue 계산
        revenue = (
            (next_price * next_row["purchase_count_per_day"])
        )
        
        # Unique Visitors 최소값 보장
        uv = max(1, next_row["unique_visitors"])
        
        # RCR 계산
        current_rcr = revenue / uv

        # RCR 변화량 계산
        if self.previous_rcr is not None:
            rcr_change = ((current_rcr - self.previous_rcr))
                        
        else:
            rcr_change = 0

        demand_change = next_row["purchase_count_per_day"] / max(1, current_row["purchase_count_per_day"])
        total_cost = cost * int(top_k_action)

        # Normalize values using RewardNormalizer
        reward = self.normalizer.normalize_reward(
            revenue=revenue,
            rcr_change=rcr_change,
            precision_at_k_value=(next_price * precision_at_k_value * int(top_k_action) * demand_change) - total_cost,
        )

        # 이전 RCR 업데이트
        self.previous_rcr = current_rcr
        
        # 다음 상태 가져오기
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