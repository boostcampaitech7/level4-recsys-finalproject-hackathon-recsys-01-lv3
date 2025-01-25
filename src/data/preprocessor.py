# /src/data/preprocess.py

import polars as pl

# 1. 중복 제거 함수
def remove_duplicates_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    cart_and_purchase_df = df.filter(pl.col('event_type_index').is_in([2,3]))
    view_df = df.filter(pl.col('event_type_index') == 1).group_by(["event_time_index", "event_type_index", "product_id_index", "user_session_index", "price"]).agg(pl.all().first())
    result = pl.concat([cart_and_purchase_df, view_df], how='vertical')
    return result

# 2. 가격이 0 이하인 데이터 제거 함수
def remove_zero_price_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(pl.col("price") > 0)

# 3. 세션 관리 전처리 함수
def session_preprocess_lazy(df: pl.LazyFrame, session_df: pl.LazyFrame) -> pl.LazyFrame:
    invalid_sessions = session_df.select(pl.col("user_session_index").unique())
    return df.join(invalid_sessions, on="user_session_index", how="anti")

# 4. 봇 트래픽 제거 함수
def filter_bot_traffic(
    df: pl.LazyFrame,
    event_threshold: int = 5,
    session_threshold: int = 1
) -> pl.LazyFrame:
    """
    봇 트래픽 필터링 함수
    1. 특정 초당 이벤트 임계값 초과 시 봇 트래픽 식별.
    2. 다중 세션 활동 확인 및 필터링.
    
    Parameters:
        df (pl.LazyFrame): 이벤트 로그 데이터.
        user_data (pl.LazyFrame): 사용자 데이터.
        event_threshold (int): 초당 이벤트 허용 임계값 (기본값: 5).
        session_threshold (int): 세션 활동 임계값 (기본값: 1).
    
    Returns:
        pl.LazyFrame: 봇 트래픽이 필터링된 데이터.
    """
    # 1. 초당 이벤트 수 계산 및 봇 트래픽 식별
    bot = (
        df.select(["user_id_index", "event_time_index"])
        .group_by(["user_id_index", "event_time_index"])
        .agg([
            pl.count().alias("events_per_second")
        ])
        .filter(pl.col("events_per_second") > event_threshold)
    )

    # 2. 봇 트래픽 데이터 추출
    bot_traffic = df.join(bot, on=["user_id_index", "event_time_index"], how="right")

    # 3. 다중 세션 활동 확인
    bot_check = (
        bot_traffic
        .select(["user_id_index", "event_time_index", "user_session_index"])
        .unique()
        .group_by(["user_id_index", "event_time_index"])
        .agg([
            pl.count().alias("session_cnt")
        ])
        .filter(pl.col("session_cnt") > session_threshold)  # 다중 세션 활동 기준 적용
        .sort(["user_id_index", "event_time_index"])
    )

    # 4. 봇 트래픽 필터링
    filter_bot = bot_check.join(
        bot_traffic,
        on=["user_id_index", "event_time_index"],
        how="left"
    )

    # 5. 원본 데이터에서 봇 트래픽 제거
    filtered_df = df.join(
        filter_bot.select([
            "user_id_index", "event_time_index", "event_type_index",
            "user_session_index", "product_id_index", "price"
        ]),
        on=["user_id_index", "event_time_index", "event_type_index",
            "user_session_index", "product_id_index", "price"],
        how="anti"
    )

    return filtered_df

# 5. 전처리 통합 함수
def preprocess_log_data(df: pl.LazyFrame, session_df: pl.LazyFrame, user_data: pl.LazyFrame) -> pl.LazyFrame:
    df = remove_duplicates_lazy(df)
    df = remove_zero_price_lazy(df)
    df = session_preprocess_lazy(df, session_df)
    df = df.join(user_data, on='user_session_index', how='left')
    df = filter_bot_traffic(df, event_threshold=5, session_threshold=1).drop('user_id_index')
    return df