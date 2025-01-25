import polars as pl

def cat_brand_to_item(item_data: pl.LazyFrame, cat_table_kr: pl.LazyFrame, brand_table: pl.LazyFrame) -> pl.LazyFrame:
    """
    item_data에 대분류 & 소분류 카테고리 이름과 브랜드 이름 붙이기
    """
    results = item_data.join(cat_table_kr.select(['category_id', 'first_category', 'last_category']), on='category_id', how='left').join(brand_table, on='brand_id', how='left')
    return results.select(['product_id_index', 'first_category', 'last_category', 'brand'])

# view/cart/purchase count for product_id
def calculate_event_count_for_item(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    product_id별 view/cart/purchase 빈도 집계
    """
    df = log_data.drop_nulls().group_by(['product_id_index', 'event_type_index']).agg([pl.len().alias('count')]).collect()
    return df.pivot(on='event_type_index', index='product_id_index', values='count').fill_null(0).rename({
        '1': 'view_count', '2': 'cart_count', '3': 'purchase_count'
    })
    
def calculate_item_avg_price(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    제품별 평균 가격 집계
    """
    return log_data.drop_nulls().group_by('product_id_index').agg(pl.col('price').mean().alias('avg_price'))

def make_total_purcases_and_unique_buyers(log_data: pl.LazyFrame, user_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    제품별 총 구매 횟수 및 고유한 구매 유저 수 집계
    """
    return log_data.drop_nulls().filter(pl.col('event_type_index')==3).join(user_data, on='user_session_index', how='left').group_by('product_id_index').agg(
        pl.len().alias('total_purchases'),
        pl.col('user_id_index').n_unique().alias('unique_buyers')
    )