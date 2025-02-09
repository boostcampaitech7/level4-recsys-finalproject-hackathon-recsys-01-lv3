import torch
import polars as pl
import numpy as np

class DynamicPricingEnv:
    def __init__(self, df: pl.DataFrame, item_user_scores, elasticity_df, raw_df):
        """
        Initialize the Dynamic Pricing Environment.
        """
        self.df = df
        self.current_day_idx = 0
        self.item_user_scores = item_user_scores
        self.elasticity_df = elasticity_df
        self.raw_df = raw_df

        min_price = elasticity_df["price_bucket"].min()
        max_price = elasticity_df["price_bucket"].max()
        self.PRICE_BUCKETS = np.linspace(min_price, max_price, num=21)

        self.price_action_space = np.linspace(-0.3, 0.0, num=7)  # 가격 조정 범위: [-0.3, -0.25, ..., 0.0]
        self.top_k_action_space = range(1, 10001)  # 추천 사용자 수 범위
        self.continuous_columns = ["price"]
        self.extend_columns = ["item_id", "category_id_encoded", "brand_id"]


    def extend_state_with_metadata(self, state_continuous, extend_data):
        """
        상태를 확장하여 연속형 상태와 범주형 데이터를 결합.
        """
        extended_state = np.concatenate((
            np.ravel(state_continuous), 
            np.ravel(extend_data)
        ))
            
        return extended_state
    
    def reset(self):
        """
        Reset the environment to the initial state and return the extended state.
        """
        self.current_day_idx = 0
        
        # 초기 상태 가져오기 (확장된 형태로)
        state = self._get_state(self.current_day_idx)
        return state

    def _get_state(self, idx):
        """
        Extract the state features for a given index and extend with metadata.
        :param idx: Index of the current day in the dataset.
        :return: Extended state as a NumPy array.
        """
        # 연속형 상태 가져오기
        row = self.df.row(idx, named=True)  # Use named=True to get a dictionary
        state_continuous = np.array([row[col] for col in self.continuous_columns])

        # 추가 데이터 (범주형 데이터 등)
        extend_data = np.array([
            row[col] for col in self.extend_columns
        ])

        # 상태 확장
        extended_state = self.extend_state_with_metadata(state_continuous, extend_data)
        return extended_state

    def validate_action(self, price_action, top_k_action):
        # 가격 조정 액션 검증
        if price_action not in self.price_action_space:
            raise ValueError(f"Invalid price action: {price_action}. Must be one of {self.price_action_space}.")
        
        # 추천 사용자 수 액션 검증
        if not isinstance(top_k_action, int):
            raise ValueError(f"Invalid top_k action: {top_k_action}. Must be an integer.")
        if top_k_action not in self.top_k_action_space:
            raise ValueError(f"Invalid top_k action: {top_k_action}. Must be between {min(self.top_k_action_space)} and {max(self.top_k_action_space)}.")

    def discretize_top_k(self, top_k_action):
        # 추천 사용자 수를 정수화하여 가장 가까운 정수 반환
        return int(round(top_k_action))
    
    def sample_price_action(self):
        return np.random.choice(self.price_action_space)

    def sample_top_k_action(self):
        return np.random.randint(min(self.top_k_action_space), max(self.top_k_action_space) + 1)

    def step(self, actions):
        """
        Perform an action in the environment.
        :param actions: Tuple (price adjustment, top_k adjustment).
        :return: next_state, high_level_reward, low_level_reward, done.
        """
        price_action, top_k_action = actions
        top_k_action = self.discretize_top_k(top_k_action)

        # Validate actions before proceeding
        self.validate_action(price_action, top_k_action)

        # 종료 조건 확인
        if self.current_day_idx >= len(self.df):  # 인덱스 초과 방지
            raise IndexError(f"current_day_idx {self.current_day_idx} is out of bounds for DataFrame with length {len(self.df)}")

        current_row = self.df.row(self.current_day_idx, named=True)
        current_price = current_row["price"]
        product_id = int(current_row["item_id"])

        current_price_bucket = self._get_price_bucket(current_price)
        elasticity = self.elasticity_df.filter(pl.col("price_bucket") == current_price_bucket)["elasticity"].to_numpy()[0]

        # 가격 조정 및 수요량 계산
        next_price = max(0, current_price * (1 + price_action))
        demand_change_rate = (1 + (elasticity * price_action))

        # Precision at K 계산
        true_users = self.get_true_users(product_id, next_price)
        precision_at_k_value = self.precision_at_k(product_id, true_users, top_k=int(top_k_action))

        # RecSys Value 계산 (추천 시스템 관련 매출)
        recsys_value = self.calculate_recsys_value(next_price, precision_at_k_value, top_k_action, demand_change_rate)

        # 보상 계산
        low_reward = self.low_level_reward(recsys_value=recsys_value)

        # 다음 상태로 이동
        self.current_day_idx += 1
        done = (self.current_day_idx >= len(self.df))  # 종료 조건

        if done:
            return None, low_reward, done

        next_state = self._get_state(self.current_day_idx)
        
        return next_state, low_reward, done

    def calculate_recsys_value(self, next_price, precision_at_k, top_k_action, next_demand):
        """
        추천 시스템 매출 기여도(RecSys Value) 계산
        :param next_price: 조정된 다음 가격
        :param precision_at_k: Precision@K 값
        :param top_k_action: 추천 Top-K 값
        :param next_demand: 조정된 다음 수요량
        :return: RecSys Value (추천 시스템 관련 매출)
        """
        demand = int(round(precision_at_k * int(top_k_action) * next_demand))
        marketing_cost = (next_price * 0.01) * int(top_k_action)  # 마케팅 비용
        recsys_value = ((next_price * demand) - marketing_cost)
        return recsys_value
    
    def _get_price_bucket(self, price):
        """
        주어진 가격이 속하는 price_bucket을 반환.
        :param price: 현재 가격.
        :return: 해당 price_bucket 값.
        """
        for i in range(len(self.PRICE_BUCKETS) - 1):
            if self.PRICE_BUCKETS[i] <= price < self.PRICE_BUCKETS[i + 1]:
                return self.PRICE_BUCKETS[i]
        
        return self.PRICE_BUCKETS[-1]  # 마지막 구간 처리

    def get_true_users(self, product_id, next_price):
        """
        주어진 product_id에 대해 상호작용한 실제 사용자 리스트 반환.
        :param product_id: 상품 ID.
        :param next_price: 다음 가격 (필터 기준).
        :return: 다음 가격보다 낮은 가격으로 구매한 사용자 ID 리스트.
        """
        # Get all true users for the given product_id
        # true_users = self.true_users_by_product.get(product_id, [])

        user_data = self.raw_df.filter(pl.col("item_id") == product_id)


        # Filter users based on the next_price condition
        filtered_users = (
            user_data.filter(pl.col("price") >= next_price)["user_id"]
            .unique()
            .to_list()
        )
        return filtered_users
        
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

        if isinstance(recommended_users, torch.Tensor):
            recommended_users = recommended_users.tolist()

        # Calculate Precision: |relevant ∩ recommended| / |recommended|
        relevant_and_recommended = len(set(recommended_users) & set(true_users))
        precision = relevant_and_recommended / len(recommended_users) if len(recommended_users) > 0 else 0

        return precision
    
    def low_level_reward(self, recsys_value):
        """
        하위 계층 보상 함수: 추천 시스템의 기여도를 최적화.
        :param recsys_value: 추천 시스템이 기여한 매출 가치.
        :return: 하위 계층 보상 값.
        """
        reward = recsys_value # 정규화된 추천 시스템 가치 (현재 사용 안함.)
        return reward