import polars as pl
import os
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

FILE_PATH = os.path.join(config["data"]["group"], "PP_Group_3.parquet")

print(f"file_path : {FILE_PATH}")

df = pl.read_parquet(FILE_PATH)
df = df.drop("category_codes")

# 표시할 열의 개수를 늘리거나 제한을 해제
pl.Config.set_tbl_cols(n=-1)  # n = -1 이면 모든 열을 표시


"""
데이터 개요
# """
# # print("\n앞 5개")
# # print(df.head())

print("\nShape")
print(df.shape)

# print("\n열 unique 수")
# print(df["product_id_index"].unique())



"""
결측치
"""
# print("\n결측치")
# print(df.null_count())

# print("\n결측 있는 행 출력")
# missing = df.filter(pl.col("user_session_index").is_null())
# print(missing)

# print(missing.group_by("event_type_index").agg(
#     pl.count("event_type_index").alias("count")
# ))

# print("\nevent_type별 count")
# print(df.group_by("event_type_index").agg(
#     pl.count("event_type_index").alias("count")
# ))

# print("\nevent가 purchase일 때 가격이 0인 행")
# print(
#     df.filter(
#         (df["event_type_index"] == 3) & (df["price"] == 0)
#         )
# )



"""
중복 행
"""
# print("\n중복 행 unique 출력")
# # 중복 확인할 열 목록
# columns_to_check = ["user_id_index", "user_session_index", "event_time_index", "product_id_index"]
# # 각 조합의 개수를 세고 1개 초과하는 경우 중복으로 판단
# duplicates = (
#     df.group_by(columns_to_check)
#     .agg(pl.len().alias("count"))
#     .filter(pl.col("count") > 1)
# )
# print(duplicates)
# print("\n중복 행 출력")
# duplicated_rows = df.join(duplicates, on=columns_to_check, how="inner")
# print(duplicated_rows)
# print("\n중복 행 이벤트 타입별 개수")
# print(duplicated_rows.group_by("event_type_index").agg(
#     pl.count("event_type_index").alias("count")
# ))



"""
price 0
"""
# print("\n가격 0인 행")
# price_0 = df.filter(df["price"] == 0)
# print(price_0)

# print("\n가격 0 : 이벤트 타입 별")
# print(price_0.group_by("event_type_index").agg(
#     pl.len().alias("count")
# ))

# print("\n가격 0 : 카테고리 별")
# print(price_0.group_by("category_1").agg(
#     pl.len().alias("count")
# ).sort("category_1"))

# print("\n가격 0 : 브랜드 별")
# print(price_0.group_by("brand").agg(
#     pl.len().alias("count")
# ).sort("count", descending=True).head())


"""
카테고리별 개요
"""
# print("\n카테고리 리스트")
# print(df["category_1"].unique())

# print("\n카테고리별 행 개수")
# print(df.group_by("category_1").agg(
#     pl.len().alias("count")
# ))

# print("\n카테고리별 브랜드 종류")
# unique_brands_by_category = df.group_by("category_1").agg(
#     pl.col("brand").unique().alias("unique_brands")
# )
# for row in unique_brands_by_category.iter_rows():
#     print(row)
#     print("-" * 50)

# print("\n카테고리별 브랜드 수")
# print(df.group_by("category_1").agg(
#     pl.col("brand").n_unique().alias("unique_brand_count")
# ).sort("category_1"))

"""
단변량 : 이벤트 타입
"""
# print(df.group_by("event_type_index").agg(
#     pl.len().alias("count"))
#     .with_columns(
#         (pl.col("count") / pl.col("count").sum()*100).alias("ratio")
#     )
# )
# unique_event_time_df = pl.read_parquet(os.path.join(config["data"]["map"], "unique_event_time_index.parquet"))
# df = df.select(["event_time_index", "event_type_index"])
# df = df.join(unique_event_time_df, on="event_time_index", how="left")
# df = df.select(["event_time", "event_type_index"])
# print(df)


"""
단변량 : 가격
"""
# print(df.select("price").describe())

"""
논의 : 가격이 떨어지면 이벤트가 많이 발생하는가? (이벤트 수 상위 5개)
"""

# # 1. 이벤트 수 상위 5개 제품 추출
# top_5_products = (
#     df.group_by("product_id_index")
#     .agg(pl.len().alias("event_count"))
#     .sort("event_count", descending=True)
#     .head(5)  # 상위 5개 제품 추출
#     .select("product_id_index")
# )["product_id_index"].to_list()
# print("\n이벤트 수 상위 5개 제품 index")
# print(top_5_products)
# # 2. 상위 5개 제품의 데이터 필터링
# filtered_df = df.filter(pl.col("product_id_index").is_in(top_5_products))
# print("\n상위 5개 제품 행들")
# print(filtered_df)

# print("\n상위 5개 제품 정보")
# for i in top_5_products:
#     print(f"\nproduct_id_index = {i}")
#     temp = filtered_df.filter(pl.col("product_id_index") == i)
#     print(temp
#         .group_by(["product_id_index", "event_type_index", "brand", "category_1"])
#         .agg([pl.len().alias("count"),
#               pl.col("price").median().alias("median_price"),  # 중위값
#             pl.col("price").mode().alias("mode_price"),  # 최빈값
#             pl.col("price").mean().alias("mean_price")  # 평균값
#               ])
#         .sort(["product_id_index", "event_type_index"])
#         )
# # 3. 이벤트 타입별 가격 분포 추출
# event_type_price_distributions = (
#     filtered_df.group_by(["product_id_index", "event_type_index"])
#     .agg(pl.col("price").alias("price_distribution"))
# )
# print("\n상위 5개 제품 이벤트 타입과 가격 분포")
# print(event_type_price_distributions)