import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
from time import time
import pandas as pd
import numpy as np
from src.utils import load_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    raw_path = "~/Hackathon/src/data/raw"
    map_path = "~/Hackathon/src/data/map"
    log_data, user_data, item_data, cat_table, cat_table_kr, brand_table, filter_session = load_file.load_data_lazy(raw_path, preprocess=False)

    log_data = log_data.collect()
    time_data = pl.read_parquet(os.path.join(map_path, 'unique_event_time_index.parquet'))
    user_data = user_data.collect()
    
    # print(log_data)
    # print(time_data)
    # print(user_data)

    # 2020.02 이전 세션 + type이 3인 세션 filter
    reference_date = pl.datetime(2020, 2, 28)
    purchase_data = (log_data
        ).filter(
            pl.col("event_type_index") == 3
        ).join(
            time_data, how = "left", on = "event_time_index"
        ).drop(
            ["event_time_index", "event_type_index", "product_id_index"]
        ).with_columns(
            pl.col("event_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S %Z")
        ).filter(
            pl.col("event_time") <= reference_date
        ).join(
            user_data, on = "user_session_index", how = "left"
        )


    def basic_R(purchase_data: pl.DataFrame, img: bool = False):
        """_summary_
            basic_R 계산 및 이미지 출력 함수
        Args:
            purchase_data (pl.DataFrame): _description_
            img (bool, optional): _description_. Defaults to False.
        """
        R_data = (purchase_data
            ).group_by(
                "user_id_index"
            ).agg(
                pl.col("event_time").max().alias("recent_event_time")
            ).with_columns(
                ((reference_date - pl.col("recent_event_time").dt.truncate("1d")).cast(pl.Int64) // 86400000000).alias("passed_time")
            )
        
        print("\n" + "-"*10 + "basic_R" + "-"*10)
        print(R_data)
        print(f"최대 지난 시간 : {R_data['passed_time'].max()}") # 150일

        if img:
            plt.hist(R_data["passed_time"], bins=np.arange(0, 150, 5), edgecolor='k')
            plt.title('Distribution of Passed Time')
            plt.xlabel('Days Passed')
            plt.ylabel('Frequency')
            plt.savefig("result/basic_img/R_data.png")

        return R_data

    basic_R(purchase_data)

    def basic_F(purchase_data: pl.DataFrame, img: bool = False):
        """_summary_
            basic_F 계산 및 이미지 출력 함수
        Args:
            purchase_data (pl.DataFrame): _description_
            img (bool, optional): _description_. Defaults to False.
        """
        F_data = (purchase_data
            ).group_by(
                "user_id_index"
            ).agg(
                pl.len().alias("purchase_count")
            )

        print("\n" + "-"*10 + "basic_F" + "-"*10)
        print(F_data)
        print(f"최대 구매 횟수 : {F_data['purchase_count'].max()}") # 1950
        print(f"최소 구매 횟수 : {F_data['purchase_count'].min()}") # 1
        print(f"최빈 구매 횟수 : {F_data['purchase_count'].mode()}") # 1
        if img:
            plt.hist(F_data["purchase_count"], bins=np.arange(0, 2000, 1), edgecolor='k')
            plt.yscale('log')
            plt.title('Distribution of Purchase Count (Log Scale)')
            plt.xlabel('Count')
            plt.ylabel('Frequency')
            plt.savefig("result/basic_img/F_data_log_scale.png")
        
        return F_data

    def basic_M(purchase_data: pl.DataFrame, img: bool = False):
        """_summary_
            basic_M 계산 및 이미지 출력 함수
        Args:
            purchase_data (pl.DataFrame): _description_
            img (bool, optional): _description_. Defaults to False.
        """
        M_data = (purchase_data
            ).group_by(
                "user_id_index"
            ).agg(
                pl.col("price").sum().alias("total_spending")
            )

        print("\n" + "-"*10 + "basic_M" + "-"*10)
        print(M_data)
        print(f"최대 총 구매 금액 : {M_data['total_spending'].max()}") # 699547.25
        print(f"최소 총 구매 금액 : {M_data['total_spending'].min()}") # 0.7699999809265137
        if img:
            plt.hist(M_data["total_spending"], bins=np.arange(0, 10_000, 100), edgecolor='k')
            # plt.yscale('log')
            plt.title('Distribution of Total Spending')
            plt.xlabel('Total Purchase Amount')
            plt.ylabel('Frequency')
            plt.savefig("result/basic_img/M_data_expand.png")
        
        return M_data
    
    # 라벨링 함수 정의
    def label_by_quantile(df: pl.DataFrame, col_name: str, higher_is_better=True):
        q1 = df[col_name].quantile(0.25) # 25%
        q3 = df[col_name].quantile(0.75) # 75%
        print("\n" + f"{col_name}'s q1, q3 : {q1}, {q3}")

        # 조건문 생성
        if higher_is_better:
            # 값이 클수록 높은 라벨 (1: 낮음, 3: 높음)
            return (
                pl.when(pl.col(col_name) <= q1).then(1)
                .when(pl.col(col_name) <= q3).then(2)
                .otherwise(3)
                .alias(f"{col_name}_label")
            )
        else:
            # 값이 작을수록 높은 라벨 (3: 낮음, 1: 높음)
            return (
                pl.when(pl.col(col_name) <= q1).then(3)
                .when(pl.col(col_name) <= q3).then(2)
                .otherwise(1)
                .alias(f"{col_name}_label")
            )
    
    def make_basic_RFM(purchase_data, save : bool = False):
        R = basic_R(purchase_data, img = False).drop("recent_event_time")
        F = basic_F(purchase_data, img = False)
        M = basic_M(purchase_data, img = False)
        
        basic_RFM = R.join(F, on="user_id_index", how="inner").join(M, on="user_id_index", how="inner")
        
        basic_RFM = basic_RFM.with_columns(
        label_by_quantile(basic_RFM, "passed_time", higher_is_better=False),  # 짧을수록 좋음
        label_by_quantile(basic_RFM, "purchase_count", higher_is_better=True),  # 클수록 좋음
        label_by_quantile(basic_RFM, "total_spending", higher_is_better=True)  # 클수록 좋음
        )

        print("\n" + "-"*10 + "basic_RFM" + "-"*10)
        print(basic_RFM)
        if save:
            basic_RFM.write_parquet("result/basic_RFM.parquet")

        return basic_RFM
    
    
    
    def one_var_plt(save = False):
        basic_RFM = pl.read_parquet("result/basic_RFM.parquet")
        # 각 라벨별 카운트 계산
        label_counts = {
        "passed_time_label": basic_RFM.group_by("passed_time_label").agg(pl.col("passed_time_label").count().alias("count")),
        "purchase_count_label": basic_RFM.group_by("purchase_count_label").agg(pl.col("purchase_count_label").count().alias("count")),
        "total_spending_label": basic_RFM.group_by("total_spending_label").agg(pl.col("total_spending_label").count().alias("count")),
        }

        # 시각화
        for label, counts in label_counts.items():
            # X축과 Y축 값 추출
            x = counts["passed_time_label" if "passed_time_label" in counts.columns else "purchase_count_label" if "purchase_count_label" in counts.columns else "total_spending_label"].to_list()
            y = counts["count"].to_list()

            # 그래프 생성
            plt.figure(figsize=(6, 4))
            plt.bar(x, y, tick_label=x, color='skyblue', edgecolor='black')
            plt.title(f"Counts for {label}")
            plt.xlabel("Label")
            plt.ylabel("Count")
            plt.xticks(x)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            if save:
                plt.savefig(f"result/basic_img/{label}.png")
    
    def tri_bar_plt(save = False):
        # 3D 플롯 생성
        basic_RFM = pl.read_parquet("result/basic_RFM.parquet")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # X, Y, Z 값 추출
        x = basic_RFM["passed_time"].to_list()  # R
        y = basic_RFM["purchase_count"].to_list()  # F
        z = basic_RFM["total_spending"].to_list()  # M

        # 3D 산점도
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, edgecolor='k', alpha=0.7)

        # 라벨 설정
        ax.set_title("3D RFM Plot", fontsize=14)
        ax.set_xlabel("Recency (R) - passed_time", fontsize=12)
        ax.set_ylabel("Frequency (F) - purchase_count", fontsize=12)
        ax.set_zlabel("Monetary (M) - total_spending", fontsize=12)

        # 색상 바 추가
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label("Monetary (M)", fontsize=12)

        # 결과 저장
        if save:
            plt.savefig("result/basic_img/RFM_3Dplot.png")
            plt.close()

if __name__ == '__main__':
    start = time()
    main()
    print(f"총 걸린 시간: {time() - start:.2f}s") 