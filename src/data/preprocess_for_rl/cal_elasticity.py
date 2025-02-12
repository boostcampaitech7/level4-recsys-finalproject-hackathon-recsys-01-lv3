import numpy as np
import polars as pl

class ElasticityCalculator:
    """
    A class to calculate price elasticity based on demand and price data.
    """

    def __init__(self, df):
        """
        Initialize the ElasticityCalculator with a dataframe.
        
        Args:
            df (pl.DataFrame): Input dataframe containing price and demand data.
        """
        self.df = df
        self.elasticity_df = None
        self.PRICE_BUCKETS = None
        self.MIDPOINTS = None

    def calculate_demand(self):
        """
        Calculate demand for each price point by grouping and aggregating the data.
        """
        self.elasticity_df = self.df.group_by("price").agg(
            pl.count().alias("demand")
        )

    def define_price_buckets(self):
        """
        Define price buckets and their midpoints for segmentation.
        """
        min_price = self.elasticity_df["price"].min()
        max_price = self.elasticity_df["price"].max()
        self.PRICE_BUCKETS = np.linspace(min_price, max_price, num=21)
        self.MIDPOINTS = [
            (self.PRICE_BUCKETS[i] + self.PRICE_BUCKETS[i + 1]) / 2 
            for i in range(len(self.PRICE_BUCKETS) - 1)
        ]

    def assign_price_bucket(self):
        """
        Assign each price to its corresponding price bucket.
        """
        def assign_bucket(price):
            for i in range(len(self.PRICE_BUCKETS) - 1):
                if self.PRICE_BUCKETS[i] <= price < self.PRICE_BUCKETS[i + 1]:
                    return self.PRICE_BUCKETS[i]
            return self.PRICE_BUCKETS[-1]

        self.elasticity_df = self.elasticity_df.with_columns(
            pl.col("price").map_elements(assign_bucket, return_dtype=pl.Float64).alias("price_bucket")
        )

    def calculate_average_demand(self):
        """
        Calculate average price and total demand for each price bucket.
        """
        self.elasticity_df = (
            self.elasticity_df.group_by("price_bucket")
            .agg([
                pl.col("price").mean().alias("avg_price"),
                pl.col("demand").mean().alias("total_demand")
            ])
            .sort("avg_price", descending=True)
        )

    def calculate_elasticity(self):
        """
        Calculate elasticity based on changes in price and demand.
        """
        self.elasticity_df = self.elasticity_df.with_columns([
            (pl.col("avg_price") - pl.col("avg_price").shift(1)).alias("price_change"),
            (pl.col("total_demand") - pl.col("total_demand").shift(1)).alias("demand_change")
        ])
        
        self.elasticity_df = self.elasticity_df.with_columns(
            (pl.col("demand_change") / pl.col("price_change")).alias("elasticity")
        )
        
        self.elasticity_df = self.elasticity_df.fill_null(self.elasticity_df["elasticity"].mean())

    def smooth_elasticity(self):
        """
        Smooth elasticity values using a rolling mean.
        """
        self.elasticity_df = self.elasticity_df.with_columns(
            pl.col("elasticity").rolling_mean(window_size=3).alias("smoothed_elasticity")
        )
        
        self.elasticity_df = self.elasticity_df.fill_null(self.elasticity_df["smoothed_elasticity"].mean())

    def run(self):
        """
        Execute the entire process to calculate elasticity.

        Returns:
            pl.DataFrame: Final dataframe containing elasticity calculations.
        """
        self.calculate_demand()
        self.define_price_buckets()
        self.assign_price_bucket()
        self.calculate_average_demand()
        self.calculate_elasticity()
        self.smooth_elasticity()
        
        return self.elasticity_df
