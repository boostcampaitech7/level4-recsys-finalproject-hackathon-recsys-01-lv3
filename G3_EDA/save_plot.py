import os
import yaml
import matplotlib.pyplot as plt
import polars as pl

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

FILE_PATH = os.path.join(config["data"]["group"], "PP_Group_3.parquet")
SAVE_PATH = config["img"]

print(f"FILE_PATH : {FILE_PATH}")
print(f"SAVE_PATH : {SAVE_PATH}")


df = pl.read_parquet(FILE_PATH)
df = df.drop("category_codes")


"""
카테고리 별 count
"""
# category_counts = df.group_by("category_1").agg(pl.len().alias("count")).sort("category_1")
# categories = category_counts["category_1"].to_list()
# counts = category_counts["count"].to_list()
# # 유명한 컬러맵 사용 (e.g., tab20, tab10)
# cmap = plt.get_cmap("tab20")  # 유명한 컬러맵 중 하나 사용
# colors = [cmap(i) for i in range(len(categories))]

# # 시각화
# plt.figure(figsize=(8, 6))
# bars = plt.bar(categories, counts, color=colors)
# # 막대 위에 숫자 추가
# for bar, count in zip(bars, counts):
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,  # x 위치: 막대의 중심
#         bar.get_height(),  # y 위치: 막대의 높이
#         str(count),  # 텍스트: count 값
#         ha="center", va="bottom", fontsize=10  # 가운데 정렬
#     )
# plt.xlabel("Category")
# plt.ylabel("Count")
# plt.title("Counts by Category")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH, "category_counts_plot.png"), dpi=300)




"""
카테고리 별 브랜드 수
"""
# 각 카테고리별 고유 브랜드 수 계산
# brand_counts_by_category = df.group_by("category_1").agg(
#     pl.col("brand").n_unique().alias("unique_brand_count")
# ).sort("category_1")
# categories = brand_counts_by_category["category_1"].to_list()
# brand_counts = brand_counts_by_category["unique_brand_count"].to_list()

# plt.figure(figsize=(8, 6))
# cmap = plt.get_cmap("tab20")
# colors = [cmap(i) for i in range(len(categories))]
# bars = plt.bar(categories, brand_counts, color=colors)
# for bar, count in zip(bars, brand_counts):
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height(),
#         str(count),
#         ha="center", va="bottom", fontsize=10
#     )
# plt.xlabel("Category")
# plt.ylabel("Unique Brand Count")
# plt.title("Unique Brand Counts by Category")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH, "unique_brand_counts_by_category.png"), dpi=300)

"""
이벤트별 시간 추이
"""
# unique_event_time_df = pl.read_parquet(os.path.join(config["data"]["map"], "unique_event_time_index.parquet"))
# df = df.select(["event_time_index", "event_type_index"])
# df = df.join(unique_event_time_df, on="event_time_index", how="left")
# df = df.select(["event_time", "event_type_index"])

# df = df.with_columns(pl.col("event_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC"))

# # 시간 단위로 그룹화하여 이벤트 타입별 개수 계산
# event_counts = (
#     df.group_by([pl.col("event_time").dt.truncate("1d"), "event_type_index"])
#     .agg(pl.len().alias("event_count"))
#     .pivot(values="event_count", index="event_time", on="event_type_index", aggregate_function="sum")
# )

# # 이벤트 타입 이름 매핑
# event_counts = event_counts.rename({
#     "1": "view",
#     "2": "cart",
#     "3": "purchase"
# })
# # 정렬 (event_time 기준)
# event_counts = event_counts.sort("event_time")

# # Plotting
# plt.figure(figsize=(16, 8))

# for event_type in ["view", "cart", "purchase"]:
#     plt.plot(
#         event_counts["event_time"],
#         event_counts[event_type],
#         label=event_type
#     )

# # 그래프 설정
# plt.xlabel("Date")
# plt.ylabel("Event Count")
# plt.title("Event Type Trends Over Time")
# plt.legend(title="Event Type")
# plt.grid()
# plt.tight_layout()

# # 그래프 저장
# plt.savefig(os.path.join(SAVE_PATH, "event_type_trends_over_time.png"), dpi=300)

"""
이벤트 수 상위 5개 제품에 대한 이벤트 타입별 가격 분포
"""
top_5_products = (
    df.group_by("product_id_index")
    .agg(pl.len().alias("event_count"))
    .sort("event_count", descending=True)
    .head(5)  # 상위 5개 제품 추출
    .select("product_id_index")
)["product_id_index"].to_list()
filtered_df = df.filter(pl.col("product_id_index").is_in(top_5_products))
event_type_price_distributions = (
    filtered_df.group_by(["product_id_index", "event_type_index"])
    .agg(pl.col("price").alias("price_distribution"))
)
# 이벤트 타입 이름 정의
event_type_labels = {1: "view", 2: "cart", 3: "purchase"}
# 지정된 색상
colors = {"view": "blue", "cart": "orange", "purchase": "green"}

### bar plot
plt.figure(figsize=(16, 12))
for i, product_id in enumerate(top_5_products):
    product_data = event_type_price_distributions.filter(pl.col("product_id_index") == product_id)
    plt.subplot(3, 2, i + 1)  # 3x2 subplot layout
    # 이벤트 타입 순서를 view, cart, purchase로 고정
    for event_type in [1, 2, 3]:  # 고정된 순서로 반복
        if event_type in product_data["event_type_index"].to_list():
            price_data = product_data.filter(pl.col("event_type_index") == event_type)["price_distribution"][0]
            label = event_type_labels[event_type]  # 이벤트 타입 라벨
            plt.hist(price_data, bins=30, alpha=0.6, label=label, color=colors[label])
    plt.title(f"Price Distribution for Product ID {product_id}")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend(title="Event Type")
    plt.grid()
plt.tight_layout()
# 그래프 저장
plt.savefig(os.path.join(SAVE_PATH, "top_5_product_event_type_price_distribution.png"), dpi=300)

### scatter plot
# plt.figure(figsize=(16, 12))

# for i, product_id in enumerate(top_5_products):
#     product_data = event_type_price_distributions.filter(pl.col("product_id_index") == product_id)
#     plt.subplot(3, 2, i + 1)  # 3x2 subplot layout
#     for event_type in product_data["event_type_index"].to_list():
#         price_data = product_data.filter(pl.col("event_type_index") == event_type)["price_distribution"][0]
#         label = event_type_labels[event_type]  # 이벤트 타입 라벨
#         plt.scatter(
#             range(len(price_data)), price_data, alpha=0.6, label=label, color=colors[label]
#         )
#     plt.title(f"Price Distribution for Product ID {product_id}")
#     plt.xlabel("Index")
#     plt.ylabel("Price")
#     plt.legend(title="Event Type")
#     plt.grid()
# plt.tight_layout()
# # 그래프 저장
# plt.savefig(os.path.join(SAVE_PATH, "top_5_product_event_type_price_scatter.png"), dpi=300)
