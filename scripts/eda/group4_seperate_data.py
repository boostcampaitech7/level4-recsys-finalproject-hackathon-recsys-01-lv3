import polars as pl
pl.Config.set_tbl_cols(100)

# 데이터 경로 지정
data_paths = '../../data/opt_data/Group_4_opt.parquet'
df = pl.read_parquet(data_paths)

# 저장할 category_id_index 목록
category_ids = [14, 26, 31, 33]

# 각 category_id_index별로 필터링하고 저장
for category_id in category_ids:
    filtered_df = df.filter(pl.col('category_1_id') == category_id)
    file_name = f"../../data/opt_data/Group_4_{category_id}.parquet"
    filtered_df.write_parquet(file_name)