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
    item_data = item_data.collect()
    cat_table = cat_table.collect()

    # 카테고리1 리스트
    cat_list = cat_table["category_1"].unique().to_list()
    print("\n-"*10 + "category_1 list" + "-"*10)
    print(cat_list)

    # 아이템에 해당하는 카테고리1 붙여주기
    cat_table = cat_table.select(["category_id", "category_1"])
    item_data = item_data.drop("brand_id")
    item_data = (item_data
        ).join(
            cat_table, on = "category_id", how = "left"
        )
    item_data = item_data.drop("category_id")
    print(item_data)

    # 2020.02 이전 세션 + type이 3인 세션 filter
    reference_date = pl.datetime(2020, 2, 28)
    purchase_data = (log_data
        ).filter(
            pl.col("event_type_index") == 3
        ).join(
            time_data, how = "left", on = "event_time_index"
        ).drop(
            ["event_time_index", "event_type_index"]
        ).with_columns(
            pl.col("event_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S %Z")
        ).filter(
            pl.col("event_time") <= reference_date
        ).join(
            user_data, on = "user_session_index", how = "left"
        ).join(
            item_data, on = "product_id_index", how = "left"
        ).drop(
            ["product_id_index", "user_session_index"]
        )
    
    def make_L(purchase_data: pl.DataFrame, cat: str, img: bool = False):
        if cat == None:
            cat = "basic"
        else:
            purchase_data = (purchase_data
                ).filter(
                    pl.col("category_1") == cat
                )
        
        L_data = (purchase_data
            ).group_by(
                "user_id_index"
            ).agg([
                pl.col("event_time").max().alias("recent_event_time"),  # 가장 최근 날짜
                pl.col("event_time").min().alias("earliest_event_time"),  # 가장 오래된 날짜
            ]).with_columns(
            ((pl.col("recent_event_time").cast(pl.Int64) - pl.col("earliest_event_time").cast(pl.Int64)) // 86_400_000_000).alias("length_days")
            )

        print("\n" + "-"*10 + f"{cat}_L" + "-"*10)
        print(L_data)
        print(f"최대 시간 길이 : {L_data['length_days'].max()}")

        if img:
            plt.hist(L_data["length_days"], edgecolor='k')
            plt.title(f'{cat} - Distribution of Length Days')
            plt.xlabel('Length of Days')
            plt.ylabel('Frequency')
            os.makedirs(f"result/{cat}_img", exist_ok=True)
            plt.savefig(f"result/{cat}_img/{cat}_L_data.png")
            plt.close()
        return L_data
        
    def make_R(purchase_data: pl.DataFrame, cat: str, img: bool = False):
        """_summary_

        Args:
            purchase_data (pl.DataFrame): _description_
            cat (str): _description_
            img (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: pl.DataFrame
        """
        if cat == None:
            cat = "basic"
        else:
            purchase_data = (purchase_data
                ).filter(
                    pl.col("category_1") == cat
                )
        R_data = (purchase_data
            ).group_by(
                "user_id_index"
            ).agg(
                pl.col("event_time").max().alias("recent_event_time")
            ).with_columns(
                ((reference_date - pl.col("recent_event_time").dt.truncate("1d")).cast(pl.Int64) // 86400000000).alias("passed_time")
            )

        print("\n" + "-"*10 + f"{cat}_R" + "-"*10)
        print(R_data)
        print(f"최대 지난 시간 : {R_data['passed_time'].max()}")

        if img:
            plt.hist(R_data["passed_time"], edgecolor='k')
            plt.title(f'{cat} - Distribution of Passed Time')
            plt.xlabel('Days Passed')
            plt.ylabel('Frequency')
            os.makedirs(f"result/{cat}_img", exist_ok=True)
            plt.savefig(f"result/{cat}_img/{cat}_R_data.png")
            plt.close()
        return R_data

    def make_F(purchase_data: pl.DataFrame, cat: str, img: bool = False):
        if cat == None:
            cat = "basic"
        else:
            purchase_data = (purchase_data
                ).filter(
                    pl.col("category_1") == cat
                )
        F_data = (purchase_data
            ).group_by(
                "user_id_index"
            ).agg(
                pl.len().alias("purchase_count")
            )

        print("\n" + "-"*10 + f"{cat}_F" + "-"*10)
        print(F_data)
        print(f"최대 구매 횟수 : {F_data['purchase_count'].max()}")
        print(f"최소 구매 횟수 : {F_data['purchase_count'].min()}")
        print(f"최빈 구매 횟수 : {F_data['purchase_count'].mode()}")
        if img:
            plt.hist(F_data["purchase_count"], edgecolor='k')
            plt.yscale('log')
            plt.title(f'{cat} - Distribution of Purchase Count (Log Scale)')
            plt.xlabel('Count')
            plt.ylabel('Frequency')
            os.makedirs(f"result/{cat}_img", exist_ok=True)
            plt.savefig(f"result/{cat}_img/{cat}_F_data_log_scale.png")
            plt.close()

        return F_data

    def make_M(purchase_data: pl.DataFrame, cat: str, img: bool = False):
        if cat == None:
            cat = "basic"
        else:
            purchase_data = (purchase_data
                ).filter(
                    pl.col("category_1") == cat
                )
        M_data = (purchase_data
            ).group_by(
                "user_id_index"
            ).agg(
                pl.col("price").sum().alias("total_spending")
            )

        print("\n" + "-"*10 + f"{cat}_M" + "-"*10)
        print(M_data)
        print(f"최대 총 구매 금액 : {M_data['total_spending'].max()}")
        print(f"최소 총 구매 금액 : {M_data['total_spending'].min()}")
        if img:
            plt.hist(M_data["total_spending"], edgecolor='k')
            plt.yscale('log')
            plt.title(f'{cat} - Distribution of Total Spending (Log Scale)')
            plt.xlabel('Total Purchase Amount')
            plt.ylabel('Frequency')
            os.makedirs(f"result/{cat}_img", exist_ok=True)
            plt.savefig(f"result/{cat}_img/{cat}_M_data_log_scale.png")
            plt.close()

        return M_data
    
    def make_V(purchase_data: pl.DataFrame, cat: str, img: bool = False):
        if cat == None:
            cat = "basic"
        else:
            purchase_data = (purchase_data
                ).filter(
                    pl.col("category_1") == cat
                )

        # 1. 거래 단위로 그룹화 (유저 ID와 event_time 기준)
        transaction_data = (purchase_data
            ).group_by(
                ["user_id_index", "event_time"]
            ).agg(
                pl.len().alias("transaction_quantity")  # 거래당 총 구매량
            )
        
        # 2. 날짜 단위로 그룹화 (거래 데이터를 날짜별로 집계)
        daily_data = (transaction_data
            ).with_columns(
                pl.col("event_time").dt.truncate("1d").alias("event_date")  # 날짜 단위 변환
            ).group_by(
                ["user_id_index", "event_date"]
            ).agg([
                pl.col("transaction_quantity").sum().alias("daily_quantity"),  # 날짜별 총 구매량(Q)
                pl.col("transaction_quantity").count().alias("transaction_count"),  # 거래 횟수(x)
            ]).with_columns(
                (pl.col("daily_quantity") / pl.col("transaction_count")).alias("avg_quantity_per_transaction")  # Q/x
            )

        # 3. 고객별 평균 구매량(V) 계산
        V_data = (daily_data
            ).group_by(
                "user_id_index"
            ).agg(
                pl.col("avg_quantity_per_transaction").mean().alias("volume")  # V 계산
            )

        print("\n" + "-" * 10 + f"{cat}_V" + "-" * 10)
        print(V_data)
        print(f"최대 V: {V_data['volume'].max()}")
        print(f"최소 V: {V_data['volume'].min()}")

        if img:
            plt.hist(V_data["volume"], edgecolor='k')
            plt.yscale('log')
            plt.title(f'{cat} - Distribution of Avg Daily Purchases (Log Scale)')
            plt.xlabel('Avg Daily Purchases')
            plt.ylabel('Frequency')
            os.makedirs(f"result/{cat}_img", exist_ok=True)
            plt.savefig(f"result/{cat}_img/{cat}_V_data_log_scale.png")
            plt.close()

        return V_data

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
    
    def make_LRFMV(purchase_data: pl.DataFrame, cat:str, save: bool = False, img: bool = False):
        if cat == None:
            pass
        else:
            purchase_data = (purchase_data
                ).filter(
                    pl.col("category_1") == cat
                )
        print("\n" + "-" * 10 + f"{cat}_purchase_data" + "-" * 10)
        print(purchase_data)

        L = make_L(purchase_data, cat, img).drop(["recent_event_time", "earliest_event_time"])
        R = make_R(purchase_data, cat, img).drop("recent_event_time")
        F = make_F(purchase_data, cat, img)
        M = make_M(purchase_data, cat, img)
        V = make_V(purchase_data, cat, img)

        # LRFMV 통합
        LRFMV = (
            L
            .join(R, on="user_id_index", how="inner")
            .join(F, on="user_id_index", how="inner")
            .join(M, on="user_id_index", how="inner")
            .join(V, on="user_id_index", how="inner")
        )

        LRFMV = LRFMV.with_columns(
            label_by_quantile(LRFMV, "length_days", higher_is_better=True),  # L: 길수록 좋음
            label_by_quantile(LRFMV, "passed_time", higher_is_better=False),  # R: 짧을수록 좋음
            label_by_quantile(LRFMV, "purchase_count", higher_is_better=True),  # F: 클수록 좋음
            label_by_quantile(LRFMV, "total_spending", higher_is_better=True),  # M: 클수록 좋음
            label_by_quantile(LRFMV, "volume", higher_is_better=True)  # V: 클수록 좋음
        )
        if cat == None:
            cat = "basic"
        print("\n" + "-" * 10 + f"{cat}_LRFMV" + "-" * 10)
        print(LRFMV)

        if save:
            os.makedirs("result/_LRFMV", exist_ok=True)
            LRFMV.write_parquet(f"result/_LRFMV/{cat}_LRFMV.parquet")

        return LRFMV
    
    def one_var_plt(cat: str, save = False):
        if cat == None:
            cat = "basic"
        LRFMV = pl.read_parquet(f"result/_LRFMV/{cat}_LRFMV.parquet")
        # 각 라벨별 카운트 계산
        label_counts = {
        "L_label": LRFMV.group_by("length_days_label").agg(pl.col("length_days_label").count().alias("count")),
        "R_label": LRFMV.group_by("passed_time_label").agg(pl.col("passed_time_label").count().alias("count")),
        "F_label": LRFMV.group_by("purchase_count_label").agg(pl.col("purchase_count_label").count().alias("count")),
        "M_label": LRFMV.group_by("total_spending_label").agg(pl.col("total_spending_label").count().alias("count")),
        "V_label": LRFMV.group_by("volume_label").agg(pl.col("volume_label").count().alias("count")),
        }

        # 시각화
        for label, counts in label_counts.items():
            # X축과 Y축 값 추출
            label_column = counts.columns[0]  # 첫 번째 열 이름 가져오기
            x = counts[label_column].to_list()  # 첫 번째 열 값을 리스트로 변환
            y = counts["count"].to_list()  # 'count' 열 값을 리스트로 변환

            # 그래프 생성
            plt.figure(figsize=(6, 4))
            plt.bar(x, y, tick_label=x, color='skyblue', edgecolor='black')
            plt.title(f"{cat} - Counts for {label}")
            plt.xlabel("Label")
            plt.ylabel("Count")
            plt.xticks(x)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            if save:
                os.makedirs(f"result/{cat}_img", exist_ok=True)
                plt.savefig(f"result/{cat}_img/{cat}_{label}.png")
                plt.close()
    
    def tri_bar_plt(cat: str, save = False):
        # 3D 플롯 생성
        RFM = pl.read_parquet(f"result/{cat}_RFM.parquet")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # X, Y, Z 값 추출
        x = RFM["passed_time"].to_list()  # R
        y = RFM["purchase_count"].to_list()  # F
        z = RFM["total_spending"].to_list()  # M

        # 3D 산점도
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, edgecolor='k', alpha=0.7)

        # 라벨 설정
        ax.set_title(f"{cat} - 3D RFM Plot", fontsize=14)
        ax.set_xlabel("Recency (R) - passed_time", fontsize=12)
        ax.set_ylabel("Frequency (F) - purchase_count", fontsize=12)
        ax.set_zlabel("Monetary (M) - total_spending", fontsize=12)

        # 색상 바 추가
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label("Monetary (M)", fontsize=12)

        # 결과 저장
        if save:
            plt.savefig(f"result/{cat}_img/{cat}_RFM_3Dplot.png")
            plt.close()
    
    # for cat in cat_list:
    #     make_LRFMV(purchase_data, cat = cat, save = True, img = True)
    #     one_var_plt(cat, save = True)
        # tri_bar_plt(cat, save = True)
    # make_LRFMV(purchase_data, None, save = True, img = True)
    # one_var_plt(None, save = True)

    make_LRFMV(purchase_data, "ReplacementParts", True, True)
if __name__ == '__main__':
    start = time()
    main()
    print(f"총 걸린 시간: {time() - start:.2f}s") 