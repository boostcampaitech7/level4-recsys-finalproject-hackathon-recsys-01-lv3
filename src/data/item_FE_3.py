import  math
import polars as pl
from tqdm import tqdm
from src.utils import *

def calculate_product_avg_price(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    제품별 평균 가격을 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        
    Returns:
        pl.LazyFrame: 제품별 평균 가격 데이터.
    """
    product_avg_price = (
        log_data
        .group_by("product_id_index")
        .agg(pl.col("price").mean().alias("avg_price"))
    )
    return product_avg_price


def calculate_product_avg_purchase_price(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    제품별 평균 구매가를 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        
    Returns:
        pl.LazyFrame: 제품별 평균 구매가 데이터.
    """
    product_avg_purchase_price = (
        log_data
        .group_by('product_id_index')
        .agg((pl.col('event_type_index') == 3).sum().alias('avg_purchase_price'))
    )
    return product_avg_purchase_price


def calculate_alter_product_and_market_share(cat_data: pl.LazyFrame, log_data:pl.LazyFrame, item_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    제품별 대체상품수 및 시장점유율을 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        
    Returns:
        pl.LazyFrame: 제품별 대체상품수 및 시장점유율 데이터.
    """
    cat_last_data = category_codes_to_last_category(cat_data)
    product_avg_price = calculate_product_avg_price(log_data)
    product_avg_purchase_price = calculate_product_avg_purchase_price(log_data)
    
    join_data = (cat_last_data
        ).join(
            item_data, how='left', on='category_id'
        ).join(
            product_avg_price, how='left', on='product_id_index'
        ).join(
            product_avg_purchase_price, how='left', on='product_id_index'
        ).select(
            ['category_codes', 'category_id', 'product_id_index', 'avg_price', 'avg_purchase_price', 'brand_id']
        )
    
    results = []
    for group in tqdm(join_data.partition_by('category_codes')):
        unique_group = group.unique(subset='product_id_index')
        
        for row in unique_group.iter_rows(named=True):
            if row['avg_price']:
                base_price = row['avg_price']
                similar_items = unique_group.filter(
                    (pl.col('avg_price') >= 0.9 * base_price) & 
                    (pl.col('avg_price') <= 1.1 * base_price) & 
                    (pl.col('brand_id') != row['brand_id'])
                )
                base_purchase_price = row['avg_purchase_price']
                alternative_price = similar_items['avg_purchase_price'].sum()
                
                results.append({
                    'product_id_index': row['product_id_index'],
                    'alter_product_count': len(similar_items),
                    'market_share': base_purchase_price / alternative_price if alternative_price != 0 else math.inf
                })
                
    alter_prod_and_market_share = pl.DataFrame(results)
    return alter_prod_and_market_share


def view2purchase_avg_time_by_item(log_data: pl.LazyFrame, user_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    아이템별 모든 유저에 대한 첫 view2purhcase 시간 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        user_data (pl.LazyFrame): 전체 유저 데이터.
        
    Returns:
        pl.LazyFrame: 아이템별 모든 유저에 대한 첫 view2purchase 데이터.
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
    for product_group in tqdm(purchase_product.partition_by('product_id_index')):
        product_id = product_group['product_id_index'][0]
        time_diff_days_product = []
        
        for user_group in product_group.partition_by('user_id_index'):
            first_view = None
            time_diff_days = []
            
            for row in user_group.iter_rows(named=True):
                event_type = row['event_type_index']
                event_time = row['event_time_index']
                
                if event_type == 1 and first_view is None:
                    first_view = event_time
                elif event_type == 3 and first_view is not None:
                    time_diff_seconds = (event_time - first_view).total_seconds()
                    time_diff_days.append(time_diff_seconds / 86400)
                    first_view = None
            
            time_diff_days_product.extend(time_diff_days)
        
        if time_diff_days_product:
            results.append({
                "product_id_index": product_id,
                'view_to_purchase_time': sum(time_diff_days_product) / len(time_diff_days_product)
            })
    
    view_to_purchase_time_by_item = pl.DataFrame(results)
    return view_to_purchase_time_by_item


def calculate_price_volatility(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    아이템별 가격 변동성(표준편차) 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        
    Returns:
        pl.LazyFrame: 아이템별 가격 변동성 데이터.
    """
    price_volatility = (log_data
        ).group_by(
            'product_id_index'
        ).agg(
            pl.col('price').std().alias('price_std')
        ).select(
            ['product_id_index', 'price_std']
        )
    return price_volatility


def calculate_price_change_rate_ver1(log_data: pl.LazyFrame, base_date: pl.datetime) -> pl.LazyFrame:
    """
    아이템별 기준 날짜일과 전날의 가격 변화율 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        
    Returns:
        pl.LazyFrame: 아이템별 가격 변화율 데이터.
    """
    log_time_data = event_time_id_to_date(log_data)
    
    filtered_df = (log_time_data
        ).filter(
            (pl.col('event_time_index') >= base_date - pl.duration(days=1)) &
            (pl.col('event_time_index') < base_date + pl.duration(days=1))
        ).sort(
            'event_time_index'
        )

    price_change_rate = (filtered_df
        ).group_by('product_id_index'
        ).agg([
            pl.col('price').first().alias('price_before'),
            pl.col('price').last().alias('price_base')
        ]).with_columns([
            ((pl.col('price_base') - pl.col('price_before')) / pl.col('price_before') * 100).alias('price_change_rate')
        ]).drop(
            ['price_before', 'price_base']
        )
    
    return price_change_rate


def calculate_price_change_rate_ver2(log_data: pl.LazyFrame, base_date: pl.datetime) -> pl.LazyFrame:
    """
    아이템별 기준 날짜일과 전날의 가격 변화율 계산.
    
    Parameters:
        log_data (pl.LazyFrame): 전체 로그 데이터.
        
    Returns:
        pl.LazyFrame: 아이템별 가격 변화율 데이터.
    """
    log_time_data = event_time_id_to_date(log_data)
    
    filtered_df = log_time_data.filter(
        (pl.col('event_time_index') >= base_date - pl.duration(days=1)) &
        (pl.col('event_time_index') < base_date + pl.duration(days=1))
    )

    price_change_rate = (filtered_df
        ).group_by('product_id_index'
        ).agg([
            pl.when(pl.col('event_time_index') < base_date).then(pl.col('price')).mean().alias('price_before'),
            pl.when(pl.col('event_time_index') >= base_date).then(pl.col('price')).mean().alias('price_base')
        ]).with_columns([
            ((pl.col('price_base') - pl.col('price_before')) / pl.col('price_before') * 100).alias('price_change_rate')
        ]).drop(
            ['price_before', 'price_base']
        )
        
    return price_change_rate