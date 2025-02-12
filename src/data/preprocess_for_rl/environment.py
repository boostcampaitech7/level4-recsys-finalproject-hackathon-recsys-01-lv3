import torch
import polars as pl
import numpy as np


class DynamicPricingEnv:
    """
    A class to simulate a dynamic pricing environment for reinforcement learning.
    """

    def __init__(self, df: pl.DataFrame, item_user_scores, elasticity_df, raw_df):
        """
        Initialize the Dynamic Pricing Environment.

        Args:
            df (pl.DataFrame): Dataframe containing state information.
            item_user_scores (dict): Precomputed user scores for each item.
            elasticity_df (pl.DataFrame): Dataframe containing price elasticity data.
            raw_df (pl.DataFrame): Raw dataframe for user-product interactions.
        """
        self.df = df
        self.current_day_idx = 0
        self.item_user_scores = item_user_scores
        self.elasticity_df = elasticity_df
        self.raw_df = raw_df

        min_price = elasticity_df["price_bucket"].min()
        max_price = elasticity_df["price_bucket"].max()
        self.PRICE_BUCKETS = np.linspace(min_price, max_price, num=21)

        self.price_action_space = np.linspace(-0.3, 0.0, num=7)
        self.top_k_action_space = range(1, 10001)
        self.continuous_columns = ["price"]
        self.extend_columns = ["item_id", "category_id_encoded", "brand_id"]

    def extend_state_with_metadata(self, state_continuous, extend_data):
        """
        Extend the state with metadata.

        Args:
            state_continuous (np.ndarray): Continuous state variables.
            extend_data (np.ndarray): Additional metadata to extend the state.

        Returns:
            np.ndarray: Extended state.
        """
        return np.concatenate((np.ravel(state_continuous), np.ravel(extend_data)))

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            np.ndarray: Initial extended state.
        """
        self.current_day_idx = 0
        return self._get_state(self.current_day_idx)

    def _get_state(self, idx):
        """
        Extract and extend the state for a given index.

        Args:
            idx (int): Index of the current day in the dataset.

        Returns:
            np.ndarray: Extended state as a NumPy array.
        """
        row = self.df.row(idx, named=True)
        state_continuous = np.array([row[col] for col in self.continuous_columns])
        extend_data = np.array([row[col] for col in self.extend_columns])
        return self.extend_state_with_metadata(state_continuous, extend_data)

    def validate_action(self, price_action, top_k_action):
        """
        Validate the actions before applying them.

        Args:
            price_action (float): Price adjustment action.
            top_k_action (int): Top-K recommendation action.

        Raises:
            ValueError: If actions are invalid.
        """
        if price_action not in self.price_action_space:
            raise ValueError(f"Invalid price action: {price_action}. Must be one of {self.price_action_space}.")
        
        if not isinstance(top_k_action, int):
            raise ValueError(f"Invalid top_k action: {top_k_action}. Must be an integer.")
        
        if top_k_action not in self.top_k_action_space:
            raise ValueError(f"Invalid top_k action: {top_k_action}. Must be between {min(self.top_k_action_space)} and {max(self.top_k_action_space)}.")

    def discretize_top_k(self, top_k_action):
        """
        Discretize the top-K action by rounding to the nearest integer.

        Args:
            top_k_action (float): Continuous top-K action.

        Returns:
            int: Discretized top-K action.
        """
        return int(round(top_k_action))

    def sample_price_action(self):
        """
        Sample a random price adjustment action.

        Returns:
            float: Randomly sampled price adjustment.
        """
        return np.random.choice(self.price_action_space)

    def sample_top_k_action(self):
        """
        Sample a random top-K recommendation action.

        Returns:
            int: Randomly sampled top-K value.
        """
        return np.random.randint(min(self.top_k_action_space), max(self.top_k_action_space) + 1)

    def step(self, actions):
        """
        Perform an action in the environment.

        Args:
            actions (tuple): Tuple containing price adjustment and top-K adjustment actions.

        Returns:
            tuple: 
                - next_state (np.ndarray or None): The next extended state or None if the episode is done.
                - low_level_reward (float): Reward from the low-level policy.
                - done (bool): Whether the episode is finished.

        Raises:
            IndexError: If current_day_idx exceeds dataset bounds.
            ValueError: If actions are invalid.
        """
        price_action, top_k_action = actions
        top_k_action = self.discretize_top_k(top_k_action)

        self.validate_action(price_action, top_k_action)

        if self.current_day_idx >= len(self.df):
            raise IndexError(f"current_day_idx {self.current_day_idx} is out of bounds for DataFrame with length {len(self.df)}")

        current_row = self.df.row(self.current_day_idx, named=True)
        current_price = current_row["price"]
        product_id = int(current_row["item_id"])

        current_price_bucket = self._get_price_bucket(current_price)
        elasticity = self.elasticity_df.filter(pl.col("price_bucket") == current_price_bucket)["elasticity"].to_numpy()[0]

        next_price = max(0, current_price * (1 + price_action))
        demand_change_rate = 1 + (elasticity * price_action)

        true_users = self.get_true_users(product_id, next_price)
        precision_at_k_value = self.precision_at_k(product_id, true_users, top_k=int(top_k_action))

        reward = self.calculate_recsys_value(next_price, precision_at_k_value, top_k_action, demand_change_rate)

        self.current_day_idx += 1
        done = self.current_day_idx >= len(self.df)

        if done:
            return None, reward, done

        next_state = self._get_state(self.current_day_idx)
        return next_state, reward, done

    def calculate_recsys_value(self, next_price, precision_at_k, top_k_action, next_demand):
        """
        Calculate the revenue contribution of the recommendation system (RecSys Value).

        Args:
            next_price (float): Adjusted price for the next step.
            precision_at_k (float): Precision@K value.
            top_k_action (int): Number of top-K recommendations.
            next_demand (float): Adjusted demand for the next step.

        Returns:
            float: RecSys Value representing the revenue contribution.
        """
        demand = int(round(precision_at_k * int(top_k_action) * next_demand))
        marketing_cost = (next_price * 0.01) * int(top_k_action)
        recsys_value = ((next_price * demand) - marketing_cost)
        return recsys_value

    def _get_price_bucket(self, price):
        """
        Determine the price bucket for a given price.

        Args:
            price (float): The current price.

        Returns:
            float: The corresponding price bucket.
        """
        for i in range(len(self.PRICE_BUCKETS) - 1):
            if self.PRICE_BUCKETS[i] <= price < self.PRICE_BUCKETS[i + 1]:
                return self.PRICE_BUCKETS[i]
        return self.PRICE_BUCKETS[-1]

    def get_true_users(self, product_id, next_price):
        """
        Retrieve a list of true users who interacted with a product at or above a given price.

        Args:
            product_id (int): The product ID.
            next_price (float): The threshold price.

        Returns:
            list: List of user IDs who interacted with the product at or above the given price.
        """
        user_data = self.raw_df.filter(pl.col("item_id") == product_id)
        filtered_users = (
            user_data.filter(pl.col("price") >= next_price)["user_id"]
            .unique()
            .to_list()
        )
        return filtered_users

    def recommend_users(self, product_id, top_k):
        """
        Retrieve a list of recommended users for a given product based on precomputed scores.

        Args:
            product_id (int): The product ID for recommendations.
            top_k (int): Number of top recommendations to retrieve.

        Returns:
            list: List of recommended user IDs.
        """
        if product_id not in self.item_user_scores:
            return []
        return self.item_user_scores[product_id][:top_k]

    def precision_at_k(self, product_id, true_users, top_k):
        """
        Calculate Precision@K for a given product and its true users.

        Args:
            product_id (int): The product ID.
            true_users (list): List of true users who interacted with the product.
            top_k (int): Number of top recommendations to consider.

        Returns:
            float: Precision@K value.
        """
        recommended_users = self.recommend_users(product_id, top_k=top_k)
        
        if isinstance(recommended_users, torch.Tensor):
            recommended_users = recommended_users.tolist()

        relevant_and_recommended = len(set(recommended_users) & set(true_users))
        precision = relevant_and_recommended / len(recommended_users) if len(recommended_users) > 0 else 0
        return precision