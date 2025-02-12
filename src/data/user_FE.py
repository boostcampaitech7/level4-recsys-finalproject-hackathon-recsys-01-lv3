import polars as pl
from tqdm import tqdm
from src.utils import utility  # import specific utility functions if needed

def calculate_user_event_count(
    log_data: pl.LazyFrame, user_data: pl.LazyFrame
) -> pl.DataFrame:
    """
    Calculate per-user counts for view, cart, and purchase events.

    Args:
        log_data (pl.LazyFrame): LazyFrame containing the full log data.
        user_data (pl.LazyFrame): LazyFrame containing user data.

    Returns:
        pl.DataFrame: A DataFrame pivoted by event_type_index with columns
            'view_count', 'cart_count', and 'purchase_count'.
    """
    df = log_data.drop_nulls().join(user_data, on="user_session_index", how="left")
    user_event_count = (
        df.group_by(["user_id_index", "event_type_index"])
        .agg([pl.len().alias("count")])
        .collect()
    )
    return (
        user_event_count.pivot(on="event_type_index", index="user_id_index", values="count")
        .fill_null(0)
        .rename({"1": "view_count", "2": "cart_count", "3": "purchase_count"})
    )
    
def calculate_user_avg_purchase_interval(
    log_data: pl.LazyFrame, user_data: pl.LazyFrame
) -> pl.DataFrame:
    """
    Calculate the average purchase interval (in days) for each user.

    Args:
        log_data (pl.LazyFrame): LazyFrame containing the full log data.
        user_data (pl.LazyFrame): LazyFrame containing user data.

    Returns:
        pl.DataFrame: A DataFrame with columns 'user_id_index' and 'avg_purchase_interval'.
    """
    purchase_data = log_data.drop_nulls().filter(pl.col('event_type_index')==3)
    ltv_data = (
        purchase_data.join(user_data, on="user_session_index", how="left")
        .group_by("user_id_index")
        .agg(
            [
                pl.col("event_time_index").min().alias("first_purchase"),
                pl.col("event_time_index").max().alias("last_purchase"),
                pl.len().alias("purchase_count"),
            ]
        )
    )
    ltv_data = utility.time_to_date(
        utility.time_to_date(ltv_data, column="first_purchase"), column="last_purchase"
    )
    ltv_data = ltv_data.with_columns(
        [
            (pl.col("last_purchase") - pl.col("first_purchase"))
            .dt.total_days()
            .alias("days_active")
        ]
    ).with_columns(
        (pl.col("days_active") / (pl.col("purchase_count") - 1))
        .fill_nan(0)
        .alias("avg_purchase_interval")
    )
    return ltv_data.select(["user_id_index", "avg_purchase_interval"])

def make_purchase_data(
    log_data: pl.DataFrame,
    user_data: pl.DataFrame,
    item_data: pl.DataFrame,
    cat_table: pl.DataFrame,
    reference_date: pl.Expr,
) -> pl.DataFrame:
    """
    Generate purchase data by filtering sessions with purchase events before a reference date.

    Args:
        log_data (pl.DataFrame): DataFrame output from load_data_lazy.
        user_data (pl.DataFrame): DataFrame output from load_data_lazy.
        item_data (pl.DataFrame): DataFrame output from load_data_lazy.
        cat_table (pl.DataFrame): DataFrame output from load_data_lazy.
        reference_date (pl.Expr): Reference date (e.g., output of pl.datetime function).

    Returns:
        pl.DataFrame: Processed purchase data.
    """
    cat_table = cat_table.select(["category_id", "category_1"])
    item_data = item_data.drop("brand_id")
    item_data = (item_data
        ).join(
            cat_table, on = "category_id", how = "left"
        )
    item_data = item_data.drop("category_id")

    purchase_data = log_data.filter(pl.col("event_type_index") == 3)
    purchase_data = utility.event_time_id_to_date2(purchase_data)
    purchase_data = (
        purchase_data.drop(["event_time_index", "event_type_index"])
        .filter(pl.col("event_time") <= reference_date)
        .join(user_data, on="user_session_index", how="left")
        .join(item_data, on="product_id_index", how="left")
        .drop(["product_id_index", "user_session_index"])
    )
    return purchase_data

def make_L(purchase_data: pl.DataFrame, cat: str = None) -> pl.DataFrame:
    """
    Calculate the recency and longevity (L) metric for each user.

    Args:
        purchase_data (pl.DataFrame): Purchase data (output of make_purchase_data).
        cat (str, optional): Category filter; if provided, only consider that category. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with user_id_index and a computed length in days.
    """
    if cat != None:
        purchase_data = purchase_data.filter(pl.col("category_1") == cat)
    L_data = (
        purchase_data.group_by("user_id_index")
        .agg(
            [
                pl.col("event_time").max().alias("recent_event_time"),
                pl.col("event_time").min().alias("earliest_event_time"),
            ]
        )
        .with_columns(
            (
                (pl.col("recent_event_time").cast(pl.Int64) - pl.col("earliest_event_time").cast(pl.Int64))
                // 86_400_000_000
            ).alias("length_days")
        )
    )
    return L_data

def make_R(
    purchase_data: pl.DataFrame, reference_date: pl.Expr, cat: str = None
) -> pl.DataFrame:
    """
    Calculate the recency (R) metric: the number of days passed since the most recent event.

    Args:
        purchase_data (pl.DataFrame): Purchase data (output of make_purchase_data).
        reference_date (pl.Expr): Reference date (output of pl.datetime function).
        cat (str, optional): Category filter; if provided, only consider that category. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with user_id_index and passed_time in days.
    """
    if cat is not None:
        purchase_data = purchase_data.filter(pl.col("category_1") == cat)
    R_data = (
        purchase_data.group_by("user_id_index")
        .agg(pl.col("event_time").max().alias("recent_event_time"))
        .with_columns(
            (
                (reference_date - pl.col("recent_event_time").dt.truncate("1d")).cast(pl.Int64)
                // 86_400_000_000
            ).alias("passed_time")
        )
    )
    return R_data

def make_F(purchase_data: pl.DataFrame, cat: str = None) -> pl.DataFrame:
    """
    Calculate the frequency (F) metric: total number of purchases per user.

    Args:
        purchase_data (pl.DataFrame): Purchase data (output of make_purchase_data).
        cat (str, optional): Category filter; if provided, only consider that category. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with user_id_index and purchase_count.
    """
    if cat is not None:
        purchase_data = purchase_data.filter(pl.col("category_1") == cat)
    F_data = purchase_data.group_by("user_id_index").agg(pl.len().alias("purchase_count"))
    return F_data

def make_M(purchase_data: pl.DataFrame, cat: str = None) -> pl.DataFrame:
    """
    Calculate the monetary (M) metric: total spending per user.

    Args:
        purchase_data (pl.DataFrame): Purchase data (output of make_purchase_data).
        cat (str, optional): Category filter; if provided, only consider that category. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with user_id_index and total_spending.
    """
    if cat is not None:
        purchase_data = purchase_data.filter(pl.col("category_1") == cat)
    M_data = purchase_data.group_by("user_id_index").agg(
        pl.col("price").sum().alias("total_spending")
    )
    return M_data

def make_V(purchase_data: pl.DataFrame, cat: str = None) -> pl.DataFrame:
    """
    Calculate the volume (V) metric: average purchase volume per user.

    Args:
        purchase_data (pl.DataFrame): Purchase data (output of make_purchase_data).
        cat (str, optional): Category filter; if provided, only consider that category. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with user_id_index and volume.
    """
    if cat is not None:
        purchase_data = purchase_data.filter(pl.col("category_1") == cat)
    transaction_data = (
        purchase_data.group_by(["user_id_index", "event_time"])
        .agg(pl.len().alias("transaction_quantity"))
    )
    daily_data = (
        transaction_data.with_columns(
            pl.col("event_time").dt.truncate("1d").alias("event_date")
        )
        .group_by(["user_id_index", "event_date"])
        .agg(
            [
                pl.col("transaction_quantity").sum().alias("daily_quantity"),
                pl.col("transaction_quantity").count().alias("transaction_count"),
            ]
        )
        .with_columns(
            (pl.col("daily_quantity") / pl.col("transaction_count")).alias("avg_quantity_per_transaction")
        )
    )
    V_data = daily_data.group_by("user_id_index").agg(
        pl.col("avg_quantity_per_transaction").mean().alias("volume")
    )
    return V_data

def label_by_quantile(df: pl.DataFrame, col_name: str, higher_is_better: bool = True) -> pl.Expr:
    """
    Label a column based on its quantile distribution.

    Args:
        df (pl.DataFrame): Input DataFrame.
        col_name (str): Name of the column to label.
        higher_is_better (bool, optional): Whether higher values are considered better.
            Defaults to True.

    Returns:
        pl.Expr: Expression to create a new label column.
    """
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)

    if higher_is_better:
        return (
            pl.when(pl.col(col_name) <= q1).then(1)
            .when(pl.col(col_name) <= q3).then(2)
            .otherwise(3)
            .alias(f"{col_name}_label")
        )
    else:
        return (
            pl.when(pl.col(col_name) <= q1).then(3)
            .when(pl.col(col_name) <= q3).then(2)
            .otherwise(1)
            .alias(f"{col_name}_label")
        )
        
def make_LRFMV(
    purchase_data: pl.DataFrame, reference_date: pl.Expr, cat: str = None
) -> pl.DataFrame:
    """
    Generate LRFMV metrics (L, R, F, M, V) for each user.

    Args:
        purchase_data (pl.DataFrame): Purchase data (output of make_purchase_data).
        reference_date (pl.Expr): Reference date (from pl.datetime).
        cat (str, optional): Category filter; if not provided, the overall data is used.
            Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame with user_id_index and LRFMV labels.
    """
    if cat is not None:
        purchase_data = purchase_data.filter(pl.col("category_1") == cat)
    L_data = make_L(purchase_data, cat).drop(["recent_event_time", "earliest_event_time"])
    R_data = make_R(purchase_data, reference_date, cat).drop("recent_event_time")
    F_data = make_F(purchase_data, cat)
    M_data = make_M(purchase_data, cat)
    V_data = make_V(purchase_data, cat)
    LRFMV_data = (
        L_data.join(R_data, on="user_id_index", how="inner")
        .join(F_data, on="user_id_index", how="inner")
        .join(M_data, on="user_id_index", how="inner")
        .join(V_data, on="user_id_index", how="inner")
    )

    if cat == None:
        cat = "Total"
    LRFMV_data = LRFMV_data.with_columns(
        label_by_quantile(LRFMV_data, "length_days", higher_is_better=True).alias(f"L_{cat}"),
        label_by_quantile(LRFMV_data, "passed_time", higher_is_better=False).alias(f"R_{cat}"),
        label_by_quantile(LRFMV_data, "purchase_count", higher_is_better=True).alias(f"F_{cat}"),  
        label_by_quantile(LRFMV_data, "total_spending", higher_is_better=True).alias(f"M_{cat}"), 
        label_by_quantile(LRFMV_data, "volume", higher_is_better=True).alias(f"V_{cat}") 
    )
    LRFMV_data = LRFMV_data.select(["user_id_index", f"L_{cat}", f"R_{cat}", f"F_{cat}", f"M_{cat}", f"V_{cat}"])

    return LRFMV_data


def calculate_avg_purchase_price(
    log_data: pl.LazyFrame, user_data: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Calculate the average purchase price for each user.

    Args:
        log_data (pl.LazyFrame): LazyFrame containing log data with purchase events.
        user_data (pl.LazyFrame): LazyFrame containing user data.

    Returns:
        pl.LazyFrame: LazyFrame with columns 'user_id_index' and 'avg_price'.
    """
    avg_purchase_price = (
        log_data.filter(pl.col("event_type_index") == 3)
        .join(user_data, how="left", on="user_session_index")
        .group_by("user_id_index")
        .agg([pl.col("price").mean().alias("avg_price")])
    )
    return avg_purchase_price


def view2purchase_avg_time_by_user(
    log_data: pl.LazyFrame, user_data: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Calculate the average time (in days) from the first view to purchase for each user.

    Args:
        log_data (pl.LazyFrame): LazyFrame containing log data with event times.
        user_data (pl.LazyFrame): LazyFrame containing user data.

    Returns:
        pl.LazyFrame: LazyFrame with columns 'user_id_index' and 'view_to_purchase_time'.
    """
    log_time_data = utility.event_time_id_to_date(log_data)
    valid_product = log_time_data.filter(pl.col("event_type_index") == 3)["product_id_index"].unique()
    purchase_product = (
        log_time_data.join(user_data, how="left", on="user_session_index")
        .filter(pl.col("product_id_index").is_in(valid_product))
        .select(["user_id_index", "product_id_index", "event_time_index", "event_type_index"])
        .sort("event_time_index")
    )
    results = []
    for user_group in tqdm(purchase_product.partition_by('user_id_index')):
        user_id = user_group['user_id_index'][0]
        time_diff_days_user = []
        for product_group in user_group.partition_by('product_id_index'):
            first_view = None
            time_diff_days = []
            for row in product_group.iter_rows(named=True):
                event_type = row['event_type_index']
                event_time = row['event_time_index']
                if event_type == 1 and first_view is None:
                    first_view = event_time
                elif event_type == 3 and first_view is not None:
                    time_diff_seconds = (event_time - first_view).total_seconds()
                    time_diff_days.append(time_diff_seconds / 86400)
                    first_view = None
            time_diff_days_user.extend(time_diff_days)
        if time_diff_days_user:
            results.append({
                "user_id_index": user_id,
                'view_to_purchase_time': sum(time_diff_days_user) / len(time_diff_days_user)
            })
    
    view_to_purchase_time_by_user = pl.DataFrame(results)
    return view_to_purchase_time_by_user
