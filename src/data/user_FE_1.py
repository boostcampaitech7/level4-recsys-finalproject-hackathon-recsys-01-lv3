import polars as pl
from src.utils import utility

def calculate_user_event_count(log_data: pl.LazyFrame, user_data: pl.LazyFrame) -> pl.DataFrame:
    """
    사용자별 view, cart, purchase 빈도 집계
    Parameters: 
        log_data: 전체 로그 데이터
        user_data: 전체 유저 데이터
    
    Returns: 
        pl.DataFrame: 사용자별 view, cart, purchase 빈도 집계 결과
    """
    df = (
        log_data.drop_nulls()
        .join(user_data, on='user_session_index', how='left')
    )
    user_event_count = df.group_by(['user_id_index', 'event_type_index']).agg([pl.len().alias('count')]).collect()
    
    return user_event_count.pivot(on='event_type_index', index='user_id_index', values='count').fill_null(0).rename({
        '1': 'view_count', '2': 'cart_count', '3': 'purchase_count'
    })
    
def calculate_user_avg_purchase_interval(log_data: pl.LazyFrame, user_data: pl.LazyFrame) -> pl.DataFrame:
    """
    사용자별 구매당 평균 소요 시간
    Parameters: 
        log_data: 전체 로그 데이터
        user_data: 전체 유저 데이터
    Returns:
        pl.LazyFrame: 사용자별 구매당 평균 소요 시간
    """
    purchase_data = log_data.drop_nulls().filter(pl.col('event_type_index')==3)
    customer_purchase_counts = purchase_data.unique().group_by('user_id_index').agg(
        pl.len().alias('purchase_count')
    )
    ltv_data = (
        purchase_data.join(user_data, on='user_session_index', how='left')
        .group_by('user_id_index').agg([
            pl.col('event_time_index').min().alias('first_purchase'),
            pl.col('event_time_index').max().alias('last_purchase'),
            pl.len().alias('purchase_count')
        ])
    )
    ltv_data = utility.time_to_date(utility.time_to_date(ltv_data, column='first_purchase'), column='last_purchase')
    ltv_data = ltv_data.with_columns([
        (pl.col("last_purchase") - pl.col("first_purchase")).dt.total_days().alias('days_active')
    ]).with_columns((pl.col("days_active") / (pl.col("purchase_count") - 1)).fill_nan(0).alias("avg_purchase_interval"))
    return ltv_data.select(["user_id_index", "avg_purchase_interval"])