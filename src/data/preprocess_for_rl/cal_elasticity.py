import numpy as np
import polars as pl

class ElasticityCalculator:
    def __init__(self, df):
        """
        초기화 메서드. 데이터프레임과 필요한 속성을 초기화합니다.
        """
        self.df = df
        self.elasticity_df = None
        self.PRICE_BUCKETS = None
        self.MIDPOINTS = None

    def calculate_demand(self):
        """
        가격별 수요를 계산합니다.
        """
        self.elasticity_df = self.df.group_by("price").agg(
            pl.count().alias("demand"),
        )

    def define_price_buckets(self):
        """
        가격 구간과 중간값을 정의합니다.
        """
        min_price = self.elasticity_df["price"].min()
        max_price = self.elasticity_df["price"].max()
        self.PRICE_BUCKETS = np.linspace(min_price, max_price, num=21)
        self.MIDPOINTS = [(self.PRICE_BUCKETS[i] + self.PRICE_BUCKETS[i+1]) / 2 for i in range(len(self.PRICE_BUCKETS)-1)]

    def assign_price_bucket(self):
        """
        각 가격에 해당하는 가격 구간을 할당합니다.
        """
        def assign_bucket(price):
            for i in range(len(self.PRICE_BUCKETS) - 1):
                if self.PRICE_BUCKETS[i] <= price < self.PRICE_BUCKETS[i + 1]:
                    return self.PRICE_BUCKETS[i]
            return self.PRICE_BUCKETS[-1]  # 마지막 구간 처리

        self.elasticity_df = self.elasticity_df.with_columns(
            pl.col("price").map_elements(assign_bucket, return_dtype=pl.Float64).alias("price_bucket")
        )

    def calculate_average_demand(self):
        """
        가격 구간별 평균 가격과 총 수요량을 계산합니다.
        """
        self.elasticity_df = (
            self.elasticity_df.group_by("price_bucket")
            .agg([
                pl.col("price").mean().alias("avg_price"),
                pl.col("demand").mean().alias("total_demand"),  # 유저 행동이 구매를 나타낸다고 가정
            ])
        )
        self.elasticity_df = self.elasticity_df.sort("avg_price", descending=True)

    def calculate_elasticity(self):
        """
        가격 변화와 수요 변화로부터 탄력성을 계산합니다.
        """
        self.elasticity_df = self.elasticity_df.with_columns([
            (pl.col("avg_price") - pl.col("avg_price").shift(1)).alias("price_change"),
            (pl.col("total_demand") - pl.col("total_demand").shift(1)).alias("demand_change"),
        ])
        self.elasticity_df = self.elasticity_df.with_columns(
            (pl.col("demand_change") / pl.col("price_change")).alias("elasticity")
        )
        # NaN 값을 평균 탄력성으로 채웁니다.
        self.elasticity_df = self.elasticity_df.fill_null(self.elasticity_df["elasticity"].mean())

    def smooth_elasticity(self):
        """
        탄력성 값을 이동 평균으로 부드럽게 만듭니다.
        """
        self.elasticity_df = self.elasticity_df.with_columns(
            pl.col("elasticity").rolling_mean(window_size=3).alias("smoothed_elasticity")
        )
        # NaN 값을 평균 부드러운 탄력성 값으로 채웁니다.
        self.elasticity_df = self.elasticity_df.fill_null(self.elasticity_df["smoothed_elasticity"].mean())

    def run(self):
        """
        전체 과정을 실행하고 최종 결과를 반환합니다.
        """
        self.calculate_demand()
        self.define_price_buckets()
        self.assign_price_bucket()
        self.calculate_average_demand()
        self.calculate_elasticity()
        self.smooth_elasticity()

        
        return self.elasticity_df