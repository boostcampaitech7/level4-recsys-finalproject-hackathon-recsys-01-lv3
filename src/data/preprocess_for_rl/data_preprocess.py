import polars as pl
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


class PreprocessRL:
    """
    A class for preprocessing data for reinforcement learning tasks.
    """

    def __init__(self, df, recsyspath):
        """
        Initialize the PreprocessRL class with a dataframe and a path to recommendation system data.

        Args:
            df (pl.DataFrame): The input dataframe containing raw data.
            recsyspath (str): Path to the recommendation system data file.
        """
        self.df = df
        self.recsyspath = recsyspath
        self.encoder = LabelEncoder()

    def true_user_df(self):
        """
        Generate a dictionary of true users for each product based on the input dataframe.

        Returns:
            dict: A dictionary where keys are item IDs and values are lists of user IDs.
        """
        grouped = self.df.group_by("item_id").agg(pl.col("user_id").unique())
        return {row["item_id"]: row["user_id"] for row in grouped.to_dicts()}

    def load_recsys(self):
        """
        Load recommendation system data from a pickle file.

        Returns:
            dict: The loaded recommendation system data.
        """
        with open(self.recsyspath, 'rb') as f:
            return pickle.load(f)

    def make_train_df(self, min_count=500, max_count=5000, selected_columns=None, random=False):
        """
        Preprocess the input dataframe to create a training dataset.

        Args:
            min_count (int): Minimum count threshold for filtering items. Default is 500.
            max_count (int): Maximum count threshold for filtering items. Default is 5000.
            selected_columns (list): List of column names to include in the final training dataframe. 
                                     Default is ["item_id", "price", "category_id_encoded", "brand_id"].
            random (bool): If True, shuffle the training dataset. Default is False.

        Returns:
            pl.DataFrame: A processed training dataframe containing unique items with selected features.
        """
        if selected_columns is None:
            selected_columns = ["item_id", "price", "category_id_encoded", "brand_id"]

        category_encoded_df = self.df.with_columns(
            pl.col("category_id"),
            pl.Series("category_id_encoded", self.encoder.fit_transform(self.df["category_id"].to_numpy()))
        )
        
        filtered_top = category_encoded_df.group_by("item_id").agg(
            pl.count().alias("num_count")
        ).filter((pl.col("num_count") >= min_count) & (pl.col("num_count") <= max_count))
        
        merged_df = filtered_top.join(category_encoded_df, on="item_id", how="left")
        
        train_df = merged_df.select(selected_columns).unique()
        
        if random:
            train_df = train_df.sample(n=len(train_df), seed=np.random.randint(0, 10000))
        
        return train_df