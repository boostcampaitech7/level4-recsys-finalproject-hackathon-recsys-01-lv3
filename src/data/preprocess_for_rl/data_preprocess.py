import polars as pl
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


class PreprocessRL:
    # 초기화
    def __init__(self, df, recsyspath):
        self.df = df
        self.recsyspath = recsyspath
        self.encoder = LabelEncoder()

    # 실제 구매 고객 생성 (현재 사용 안함)
    def true_user_df(self):
        grouped = self.df.group_by("item_id").agg(pl.col("user_id").unique())

        # 딕셔너리 형태로 변환
        true_users_by_product = {row["item_id"]: row["user_id"] for row in grouped.to_dicts()}

        return true_users_by_product
    
    # 추천 데이터 불러오기
    def load_recsys(self):
        with open(self.recsyspath, 'rb') as f:
            return pickle.load(f)
    
    # 학습 데이터 전처리
    def make_train_df(self, random=False):
        category_encoded_df = self.df.with_columns(pl.col("category_id"), pl.Series("category_id_encoded", self.encoder.fit_transform(self.df["category_id"].to_numpy())))
        filtered_top = category_encoded_df.group_by("item_id").agg(
            pl.count().alias("num_count")
        ).filter((pl.col("num_count") >= 500) & (pl.col("num_count") <= 5000))
        merged_df = filtered_top.join(category_encoded_df, on="item_id", how="left")
        train_df = merged_df.select(["item_id", "price", "category_id_encoded", "brand_id"]).unique()

        if random == True:
            train_df = train_df.sample(n=len(train_df), seed=np.random.randint(0, 10000))

        return train_df