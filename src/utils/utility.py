import os
from pprint import pprint
from time import time

import numpy as np
import pandas as pd
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def format_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format numeric values in a DataFrame for better readability.

    Integer columns are formatted with commas, and float columns are formatted with
    two decimal places (or multiplied by 100 and formatted if the column is 'ratio').

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: A copy of the DataFrame with formatted numeric values.
    """
    formatted_df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            formatted_df[col] = df[col].apply(lambda x: f"{x:,}" if pd.notnull(x) else x)
        elif (pd.api.types.is_float_dtype(df[col])):
            if col == 'ratio':
                formatted_df[col] = df[col].apply(
                    lambda x: f"{x*100:,.2f}" if pd.notnull(x) else x
                )
            else:
                formatted_df[col] = df[col].apply(
                    lambda x: f"{x:,.2f}" if pd.notnull(x) else x
                )
    return formatted_df

def to_markdown_with_ellipsis(df: pd.DataFrame, n: int = 5) -> str:
    """
    Convert a DataFrame to a markdown string with ellipsis inserted if the DataFrame
    is too long.

    If the number of rows is greater than 2*n, the top n rows and bottom n rows are
    kept, with an ellipsis row in between.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n (int, optional): Number of rows to display at the top and bottom. Defaults to 5.

    Returns:
        str: Markdown formatted string representing the DataFrame.
    """
    if len(df) <= n * 2:
        return df.to_markdown(index=False)
    else:
        head = df.head(n)
        tail = df.tail(n)
        ellipsis = pd.DataFrame([["..."] * len(df.columns)], columns=df.columns)
        combined = pd.concat([head, ellipsis, tail], ignore_index=True)
        return combined.to_markdown(index=False)
 
def event_time_id_to_date(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert the 'event_time_index' column in a Polars DataFrame to a datetime column.

    Args:
        df (pl.DataFrame): Input DataFrame with an 'event_time_index' column.

    Returns:
        pl.DataFrame: DataFrame with 'event_time_index' converted to datetime.
    """
    return df.with_columns(
        (pl.col('event_time_index').cast(pl.UInt64)  * 1_000_000).cast(pl.Datetime())
    )
def event_time_id_to_date2(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert the 'event_time_index' column in a Polars DataFrame to a datetime column
    and rename it to 'event_time'.

    Args:
        df (pl.DataFrame): Input DataFrame with an 'event_time_index' column.

    Returns:
        pl.DataFrame: DataFrame with a new column 'event_time' as datetime.
    """
    return df.with_columns(
        (pl.col('event_time_index').cast(pl.UInt64)  * 1_000_000).cast(pl.Datetime()).alias('event_time')
    )
    
def time_to_date(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Convert a specified column in a Polars DataFrame from a Unix timestamp to a datetime column.

    Args:
        df (pl.DataFrame): Input DataFrame.
        column (str): Name of the column to convert.

    Returns:
        pl.DataFrame: DataFrame with the specified column converted to datetime.
    """
    return df.with_columns(
        (pl.col(column).cast(pl.UInt64)  * 1_000_000).cast(pl.Datetime())
    )
    
def join_cat_table(
    df: pl.DataFrame, cat_table: pl.DataFrame, columns: str | list = ["category_1"]
) -> pl.DataFrame:
    """
    Join a category table to the input DataFrame on the 'category_id' column.

    Args:
        df (pl.DataFrame): Input DataFrame.
        cat_table (pl.DataFrame): Category table containing 'category_id' and additional columns.
        columns (str or list, optional): Column or list of columns to include from the category table.
            Defaults to ["category_1"].

    Returns:
        pl.DataFrame: DataFrame after performing a left join on 'category_id'.
    """
    return df.join(
        cat_table.select(['category_id'] + columns), on='category_id', how='left'
    ) 
    
def join_cat_table_kr(
    df: pl.DataFrame,
    cat_table: pl.DataFrame,
    cat_kr_table: pl.DataFrame,
    columns: str | list = ["category_1"],
) -> pl.DataFrame:
    """
    Join category and Korean category tables to the input DataFrame based on 'category_id'.

    Args:
        df (pl.DataFrame): Input DataFrame.
        cat_table (pl.DataFrame): Category table containing 'category_codes' and 'category_id'.
        cat_kr_table (pl.DataFrame): Korean category table to be joined on 'category_codes'.
        columns (str or list, optional): Column or list of columns to include from the joined table.
            Defaults to ["category_1"].

    Returns:
        pl.DataFrame: DataFrame after performing the join on 'category_id'.
    """
    tmp = cat_table.select(['category_codes', 'category_id']).join(
        cat_kr_table, on='category_codes', how='left'
    )
    return df.join(
        tmp.select(['category_id'] + columns), on='category_id', how='left'
    )

def category_codes_to_last_category(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract the last category code from the 'category_codes' column in a Polars DataFrame.

    Args:
        df (pl.DataFrame): Input DataFrame containing a 'category_codes' column.

    Returns:
        pl.DataFrame: DataFrame with the last category code extracted.
    """
    return df.with_columns(
        (pl.col('category_codes').str.extract(r'([^\.]+)$'))
    )