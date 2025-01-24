import polars as pl
from tqdm import tqdm
from src.utils import *

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
