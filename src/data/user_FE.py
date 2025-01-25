import polars as pl
from src.utils import utility
from tqdm import tqdm
from src.utils import *

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

def make_purchase_data(log_data: pl.DataFrame, user_data: pl.DataFrame,
                        item_data: pl.DataFrame, cat_table: pl.DataFrame, reference_date: pl.Expr) -> pl.DataFrame:
    """
    구매데이터 생성

    Args:
        log_data (pl.DataFrame): load_data_lazy 함수의 출력
        time_data (pl.DataFrame): load_data_lazy 함수의 출력
        user_data (pl.DataFrame): load_data_lazy 함수의 출력
        item_data (pl.DataFrame): load_data_lazy 함수의 출력
        cat_table (pl.DataFrame): load_data_lazy 함수의 출력
        reference_date (pl.Expr): 기준일(pl.datetime 함수의 출력)

    Returns:
        pl.DataFrame: purchase_data
    """
    # 아이템에 해당하는 카테고리1 붙여주기
    cat_table = cat_table.select(["category_id", "category_1"])
    item_data = item_data.drop("brand_id")
    item_data = (item_data
        ).join(
            cat_table, on = "category_id", how = "left"
        )
    item_data = item_data.drop("category_id")

    # 기준일 이전 세션 + 구매(type이 3)인 세션 filter
    purchase_data = (log_data
        ).filter(
            pl.col("event_type_index") == 3
        )
    purchase_data = utility.event_time_id_to_date2(purchase_data)
    purchase_data = purchase_data.drop(
            ["event_time_index", "event_type_index"]
        ).filter(
            pl.col("event_time") <= reference_date
        ).join(
            user_data, on = "user_session_index", how = "left"
        ).join(
            item_data, on = "product_id_index", how = "left"
        ).drop(
            ["product_id_index", "user_session_index"]
        )
    
    return purchase_data

def make_L(purchase_data: pl.DataFrame, cat: str = None) -> pl.DataFrame:
    """
    유저별 최근 접속일, 첫 접속일, 총 이용일 계산

    Args:
        purchase_data (pl.DataFrame): 구매 데이터(make_purchase_data 함수의 출력)
        cat (str): 카테고리, 없으면 전체

    Returns:
        pl.DataFrame: L
    """
    if cat != None:
        purchase_data = (purchase_data
            ).filter(
                pl.col("category_1") == cat
            )
    L_data = (purchase_data
        .group_by("user_id_index")
        .agg([
            pl.col("event_time").max().alias("recent_event_time"),  # 가장 최근 날짜
            pl.col("event_time").min().alias("earliest_event_time"),  # 가장 오래된 날짜
            ])
        .with_columns(((pl.col("recent_event_time").cast(pl.Int64) - pl.col("earliest_event_time").cast(pl.Int64)) // 86_400_000_000)
                      .alias("length_days"))
        )

    return L_data

def make_R(purchase_data: pl.DataFrame, reference_date: pl.Expr, cat: str = None) -> pl.DataFrame:
    """
    유저별 최근 접속일로부터 기준일까지의 일 계산

    Args:
        purchase_data (pl.DataFrame): 구매 데이터(make_purchase_data 함수의 출력)
        reference_date: (pl.datetime): 기준일(pl.datetime 함수의 출력)
        cat (str): 카테고리, 없으면 전체
        
    Returns:
        pl.DataFrame: R
    """
    if cat != None:
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
            ((reference_date - pl.col("recent_event_time").dt.truncate("1d")).cast(pl.Int64) // 86_400_000_000).alias("passed_time")
        )

    return R_data

def make_F(purchase_data: pl.DataFrame, cat: str = None) -> pl.DataFrame:
    """
    유저별 총 구매 제품 수 계산

    Args:
        purchase_data (pl.DataFrame): 구매 데이터(make_purchase_data 함수의 출력)
        cat (str): 카테고리, 없으면 전체

    Returns:
        pl.DataFrame: F
    """
    if cat != None:
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

    return F_data

def make_M(purchase_data: pl.DataFrame, cat: str = None) -> pl.DataFrame:
    """
    유저별 총 구매 금액 계산

    Args:
        purchase_data (pl.DataFrame): 구매 데이터(make_purchase_data 함수의 출력)
        cat (str): 카테고리, 없으면 전체

    Returns:
        pl.DataFrame: M
    """
    if cat != None:
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

    return M_data

def make_V(purchase_data: pl.DataFrame, cat: str = None) -> pl.DataFrame:
    """
    유저별 평균 구매량 계산

    Args:
        purchase_data (pl.DataFrame): 구매 데이터(make_purchase_data 함수의 출력)
        cat (str): 카테고리, 없으면 전체

    Returns:
        pl.DataFrame: V
    """
    if cat != None:
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

    return V_data

def label_by_quantile(df: pl.DataFrame, col_name: str, higher_is_better: bool = True) -> pl.Expr:
    """
    데이터프레임의 해당 열에 대해 분위수 계산 후, 라벨링

    Args:
        df (pl.DataFrame): 데이터프레임
        col_name (str): 해당 열
        higher_is_better (bool): 높은 숫자가 더 좋은가? Defaults to True.

    Returns:
        pl.Expr: with_columns 내 조건식
    """
    q1 = df[col_name].quantile(0.25) # 25%
    q3 = df[col_name].quantile(0.75) # 75%

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
        
def make_LRFMV(purchase_data: pl.DataFrame, reference_date: pl.Expr, cat: str = None) -> pl.DataFrame:
    """
    LRFMV 생성

    Args:
        purchase_data (pl.DataFrame): 구매 데이터(make_purchase_data 함수의 출력)
        cat (str): 카테고리, 없으면 전체
        reference_date (pl.Expr): 기준일(pl.datetime 함수의 출력)

    Returns:
        pl.DataFrame: LRFMV
    """
    if cat != None:
        purchase_data = (purchase_data
            ).filter(
                pl.col("category_1") == cat
            )

    L_data = make_L(purchase_data, cat).drop(["recent_event_time", "earliest_event_time"])
    R_data = make_R(purchase_data, reference_date, cat).drop("recent_event_time")
    F_data = make_F(purchase_data, cat)
    M_data = make_M(purchase_data, cat)
    V_data = make_V(purchase_data, cat)

    # LRFMV 통합
    LRFMV_data = (
        L_data
        .join(R_data, on="user_id_index", how="inner")
        .join(F_data, on="user_id_index", how="inner")
        .join(M_data, on="user_id_index", how="inner")
        .join(V_data, on="user_id_index", how="inner")
    )

    if cat == None:
        cat = "Total"
    LRFMV_data = LRFMV_data.with_columns(
        label_by_quantile(LRFMV_data, "length_days", higher_is_better=True).alias(f"L_{cat}"),  # L: 길수록 좋음
        label_by_quantile(LRFMV_data, "passed_time", higher_is_better=False).alias(f"R_{cat}"),  # R: 짧을수록 좋음
        label_by_quantile(LRFMV_data, "purchase_count", higher_is_better=True).alias(f"F_{cat}"),  # F: 클수록 좋음
        label_by_quantile(LRFMV_data, "total_spending", higher_is_better=True).alias(f"M_{cat}"),  # M: 클수록 좋음
        label_by_quantile(LRFMV_data, "volume", higher_is_better=True).alias(f"V_{cat}")  # V: 클수록 좋음
    )
    LRFMV_data = LRFMV_data.select(["user_id_index", f"L_{cat}", f"R_{cat}", f"F_{cat}", f"M_{cat}", f"V_{cat}"])

    return LRFMV_data


def calculate_avg_purchase_price(log_data: pl.LazyFrame, user_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    유저별 평균 구매가를 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        user_data (pl.LazyFrame): 전체 유저 데이터.
        
    Returns:
        pl.LazyFrame: 유저별 평균 구매가 데이터.
    """
    avg_purchase_price = (log_data
        ).filter(pl.col('event_type_index') == 3
        ).join(user_data, how='left', on='user_session_index'
        ).group_by('user_id_index'
        ).agg([
            pl.col('price').mean().alias('avg_price')
        ])
    return avg_purchase_price


def view2purchase_avg_time_by_user(log_data: pl.LazyFrame, user_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    유저별 모든 아이템에 대한 첫 view2purhcase 시간 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        user_data (pl.LazyFrame): 전체 유저 데이터.
        
    Returns:
        pl.LazyFrame: 유저별 모든 아이템에 대한 첫 view2purchase 데이터.
    """
    log_time_data = event_time_id_to_date(log_data) 
    
    valid_product = (log_time_data
        ).filter(pl.col('event_type_index') == 3)['product_id_index'].unique()
    
    purchase_product = (log_time_data
        ).join(
            user_data, how='left', on='user_session_index'
        ).filter(
            pl.col('product_id_index').is_in(valid_product)
        ).select(
            ['user_id_index', 'product_id_index', 'event_time_index', 'event_type_index']
        ).sort(
            'event_time_index'
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
