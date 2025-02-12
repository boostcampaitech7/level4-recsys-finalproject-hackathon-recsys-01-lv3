import polars as pl
from tqdm import tqdm
from src.utils import event_time_id_to_date, category_codes_to_last_category

def cat_brand_to_item(
    item_data: pl.LazyFrame,
    cat_table_kr: pl.LazyFrame,
    brand_table: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Join Korean category and brand information to the item data.

    Args:
        item_data (pl.LazyFrame): LazyFrame containing item data.
        cat_table_kr (pl.LazyFrame): LazyFrame with Korean category data (must include columns 'category_id',
            'first_category', and 'last_category').
        brand_table (pl.LazyFrame): LazyFrame with brand information.

    Returns:
        pl.LazyFrame: LazyFrame with selected columns: 'product_id_index', 'first_category',
        'last_category', and 'brand'.
    """
    results = (
        item_data.join(
            cat_table_kr.select(["category_id", "first_category", "last_category"]),
            on="category_id",
            how="left",
        )
        .join(brand_table, on="brand_id", how="left")
    )
    return results.select(['product_id_index', 'first_category', 'last_category', 'brand'])

def calculate_event_count_for_item(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate event counts (view, cart, purchase) for each product.

    Args:
        log_data (pl.LazyFrame): Log data containing events.

    Returns:
        pl.LazyFrame: Pivoted LazyFrame with counts per product_id_index and columns renamed to
        'view_count', 'cart_count', and 'purchase_count'.
    """
    df = (
        log_data.drop_nulls()
        .group_by(["product_id_index", "event_type_index"])
        .agg([pl.len().alias("count")])
        .collect()
    )
    return (
        df.pivot(on="event_type_index", index="product_id_index", values="count")
        .fill_null(0)
        .rename({"1": "view_count", "2": "cart_count", "3": "purchase_count"})
    )
    
def calculate_item_avg_price(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate the average price for each product.

    Args:
        log_data (pl.LazyFrame): Log data containing price information.

    Returns:
        pl.LazyFrame: LazyFrame with the average price per product (column 'avg_price').
    """
    return (
        log_data.drop_nulls()
        .group_by("product_id_index")
        .agg(pl.col("price").mean().alias("avg_price"))
    )

def make_total_purcases_and_unique_buyers(
    log_data: pl.LazyFrame, user_data: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Calculate total purchases and the number of unique buyers per product.

    Args:
        log_data (pl.LazyFrame): Log data containing event information.
        user_data (pl.LazyFrame): User data containing session information.

    Returns:
        pl.LazyFrame: LazyFrame with columns 'total_purchases' and 'unique_buyers' aggregated by product_id_index.
    """
    return (
        log_data.drop_nulls()
        .filter(pl.col("event_type_index") == 3)
        .join(user_data, on="user_session_index", how="left")
        .group_by("product_id_index")
        .agg(
            pl.len().alias("total_purchases"),
            pl.col("user_id_index").n_unique().alias("unique_buyers"),
        )
    )

def calculate_product_avg_price(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate the average price for each product.

    Args:
        log_data (pl.LazyFrame): Log data containing price information.

    Returns:
        pl.LazyFrame: LazyFrame with average price data per product.
    """
    product_avg_price = (
        log_data
        .group_by("product_id_index")
        .agg(pl.col("price").mean().alias("avg_price"))
    )
    return product_avg_price


def calculate_product_avg_purchase_price(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate the average purchase price for each product.

    Args:
        log_data (pl.LazyFrame): Log data containing event and price information.

    Returns:
        pl.LazyFrame: LazyFrame with average purchase price data per product.
    """
    product_avg_purchase_price = (
        log_data
        .filter(pl.col('event_type_index') == 3)
        .group_by('product_id_index')
        .agg(pl.col('price').mean().alias('avg_purchase_price'))
    )
    return product_avg_purchase_price


def calculate_alter_product_and_market_share(
    cat_data: pl.LazyFrame, log_data: pl.LazyFrame, item_data: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Calculate the number of alternative products and the market share for each product.

    Args:
        cat_data (pl.LazyFrame): Category data.
        log_data (pl.LazyFrame): Log data containing sales and price information.
        item_data (pl.LazyFrame): Item data with additional attributes.

    Returns:
        pl.LazyFrame: LazyFrame with columns 'product_id_index', 'alter_product_count', and 'market_share'.
    """
    cat_last_data = category_codes_to_last_category(cat_data)
    product_avg_price = calculate_product_avg_price(log_data)
    
    sales_data = (
        log_data.filter(pl.col("event_type_index") == 3)
        .group_by("product_id_index")
        .agg(pl.len().alias("sales_count"))
    )
    
    join_data = (
        cat_last_data.join(item_data, how="left", on="category_id")
        .join(product_avg_price, how="left", on="product_id_index")
        .join(sales_data, how="left", on="product_id_index")
        .select(
            [
                "category_codes",
                "category_id",
                "product_id_index",
                "avg_price",
                "brand_id",
                "sales_count",
            ]
        )
        .collect()
    )
    
    results = []
    for group in tqdm(join_data.partition_by('category_codes')):
        unique_group = group.unique(subset='product_id_index')
        
        for row in unique_group.iter_rows(named=True):
            if row['avg_price'] and row['sales_count']:
                base_price = row['avg_price']
                base_sales_count = row['sales_count']
                base_sales_amount = base_price * base_sales_count
                
                similar_items = unique_group.filter(
                    (pl.col('avg_price') >= 0.9 * base_price) & 
                    (pl.col('avg_price') <= 1.1 * base_price) & 
                    (pl.col('brand_id') != row['brand_id'])
                )
                alternative_sales_amount = (similar_items['avg_price'] * similar_items['sales_count']).sum()
                
                results.append({
                    'product_id_index': row['product_id_index'],
                    'alter_product_count': len(similar_items),
                    'market_share': (base_sales_amount / (alternative_sales_amount + base_sales_amount)) * 100
                })
                
    alter_prod_and_market_share = pl.DataFrame(results)
    return alter_prod_and_market_share


def view2purchase_avg_time_by_item(
    log_data: pl.LazyFrame, user_data: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Calculate the average time (in days) from the first view to purchase for each product.

    Args:
        log_data (pl.LazyFrame): Log data containing event times and types.
        user_data (pl.LazyFrame): User data with session information.

    Returns:
        pl.LazyFrame: LazyFrame with the average view-to-purchase time per product.
    """
    log_time_data = event_time_id_to_date(log_data) 
    
    valid_product = log_time_data.filter(pl.col("event_type_index") == 3)["product_id_index"].unique()
    
    purchase_product = (
        log_time_data.join(user_data, how="left", on="user_session_index")
        .filter(pl.col("product_id_index").is_in(valid_product))
        .select(["user_id_index", "product_id_index", "event_time_index", "event_type_index"])
        .sort("event_time_index")
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
            results.append(
                {
                    "product_id_index": product_id,
                    "view_to_purchase_time": sum(time_diff_days_product) / len(time_diff_days_product),
                }
            )
    
    view_to_purchase_time_by_item = pl.DataFrame(results)
    return view_to_purchase_time_by_item


def calculate_price_volatility(log_data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate the price volatility (standard deviation) for each product.

    Args:
        log_data (pl.LazyFrame): Log data containing price information.

    Returns:
        pl.LazyFrame: LazyFrame with columns 'product_id_index' and 'price_std'.
    """
    price_volatility = (
        log_data.group_by("product_id_index")
        .agg(pl.col("price").std().alias("price_std"))
        .select(["product_id_index", "price_std"])
    )
    return price_volatility


def calculate_price_change_rate(
    log_data: pl.LazyFrame, base_date: pl.Expr
) -> pl.LazyFrame:
    """
    Calculate the price change rate for each product based on a base date.

    Args:
        log_data (pl.LazyFrame): Log data containing price and event time information.
        base_date (pl.Expr): Expression representing the base date.

    Returns:
        pl.LazyFrame: LazyFrame with price change rate data per product.
    """
    log_time_data = event_time_id_to_date(log_data)
    
    filtered_df = log_time_data.filter(
        (pl.col("event_time_index") >= base_date - pl.duration(days=1))
        & (pl.col("event_time_index") < base_date + pl.duration(days=1))
    )

    price_change_rate = (
        filtered_df.group_by("product_id_index")
        .agg(
            [
                pl.when(pl.col("event_time_index") < base_date)
                .then(pl.col("price"))
                .mean()
                .alias("price_before"),
                pl.when(pl.col("event_time_index") >= base_date)
                .then(pl.col("price"))
                .mean()
                .alias("price_base"),
            ]
        )
        .with_columns(
            [((pl.col("price_base") - pl.col("price_before")) / pl.col("price_before") * 100).alias("price_change_rate")]
        )
        .drop(["price_before", "price_base"])
    )
        
    return price_change_rate