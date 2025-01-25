# /src/data/load_data.py

import polars as pl
import src.data.preprocessor as preprocessor

# 데이터 로드 함수
def load_data_lazy(base_path: str, preprocess: bool = False):
    if preprocess:
        log_data = pl.scan_parquet(f"{base_path}/log_data.parquet")
    else: 
        log_data = pl.scan_parquet(f"{base_path}/log_data_processed.parquet")
    user_data = pl.scan_parquet(f"{base_path}/user_data.parquet")
    item_data = pl.scan_parquet(f"{base_path}/item_data.parquet")
    cat_table = pl.scan_parquet(f"{base_path}/category_table.parquet")
    cat_table_kr = pl.scan_parquet(f"{base_path}/category_table_kr.parquet")
    brand_table = pl.scan_parquet(f"{base_path}/brand_table.parquet")
    filter_session = pl.scan_parquet(f"{base_path}/filter_session.parquet")

    if preprocess:
        log_data = preprocessor.preprocess_log_data(log_data, filter_session, user_data)

    return log_data, user_data, item_data, cat_table, cat_table_kr, brand_table, filter_session
