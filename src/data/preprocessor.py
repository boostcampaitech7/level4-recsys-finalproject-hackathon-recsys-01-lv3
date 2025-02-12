import polars as pl

def remove_duplicates_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove duplicate events from the log data.

    For view events, group by selected columns and take the first occurrence.
    Then, concatenate these with cart and purchase events.

    Args:
        df (pl.LazyFrame): Input log data.

    Returns:
        pl.LazyFrame: Log data with duplicates removed.
    """
    cart_and_purchase_df = df.filter(pl.col('event_type_index').is_in([2,3]))
    view_df = (
        df.filter(pl.col("event_type_index") == 1)
        .group_by(
            [
                "event_time_index",
                "event_type_index",
                "product_id_index",
                "user_session_index",
                "price",
            ]
        )
        .agg(pl.all().first())
    )
    result = pl.concat([cart_and_purchase_df, view_df], how='vertical')
    return result

def remove_zero_price_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove rows with a price of 0 or less.

    Args:
        df (pl.LazyFrame): Input log data.

    Returns:
        pl.LazyFrame: Log data with rows where price > 0.
    """
    return df.filter(pl.col("price") > 0)

def session_preprocess_lazy(df: pl.LazyFrame, session_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove log records corresponding to invalid sessions.

    Args:
        df (pl.LazyFrame): Log data.
        session_df (pl.LazyFrame): Session data containing invalid session indices.

    Returns:
        pl.LazyFrame: Log data with invalid sessions removed.
    """
    invalid_sessions = session_df.select(pl.col("user_session_index").unique())
    return df.join(invalid_sessions, on="user_session_index", how="anti")

def filter_bot_traffic(
    df: pl.LazyFrame,
    event_threshold: int = 5,
    session_threshold: int = 1
) -> pl.LazyFrame:
    """
    Filter out bot traffic from log data.

    Bot traffic is identified based on excessive events per second and multiple session activities.

    Args:
        df (pl.LazyFrame): Log data.
        event_threshold (int, optional): Threshold for events per second. Defaults to 5.
        session_threshold (int, optional): Threshold for number of sessions in a second. Defaults to 1.

    Returns:
        pl.LazyFrame: Log data with bot traffic removed.
    """
    bot = (
        df.select(["user_id_index", "event_time_index"])
        .group_by(["user_id_index", "event_time_index"])
        .agg([
            pl.count().alias("events_per_second")
        ])
        .filter(pl.col("events_per_second") > event_threshold)
    )
    bot_traffic = df.join(bot, on=["user_id_index", "event_time_index"], how="right")
    bot_check = (
        bot_traffic
        .select(["user_id_index", "event_time_index", "user_session_index"])
        .unique()
        .group_by(["user_id_index", "event_time_index"])
        .agg([
            pl.count().alias("session_cnt")
        ])
        .filter(pl.col("session_cnt") > session_threshold)  
        .sort(["user_id_index", "event_time_index"])
    )
    filter_bot = bot_check.join(
        bot_traffic,
        on=["user_id_index", "event_time_index"],
        how="left"
    )
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

def preprocess_log_data(
    df: pl.LazyFrame, session_df: pl.LazyFrame, user_data: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Preprocess log data by applying several cleaning steps.

    The steps include:
      1. Removing duplicate events.
      2. Removing records with non-positive price.
      3. Filtering out invalid sessions.
      4. Joining with user data.
      5. Filtering out bot traffic and dropping the 'user_id_index' column.

    Args:
        df (pl.LazyFrame): Input log data.
        session_df (pl.LazyFrame): Session data.
        user_data (pl.LazyFrame): User data.

    Returns:
        pl.LazyFrame: Preprocessed log data.
    """
    df = remove_duplicates_lazy(df)
    df = remove_zero_price_lazy(df)
    df = session_preprocess_lazy(df, session_df)
    df = df.join(user_data, on='user_session_index', how='left')
    df = filter_bot_traffic(df, event_threshold=5, session_threshold=1).drop('user_id_index')
    return df