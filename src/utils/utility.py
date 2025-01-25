import polars as pl
import os
from pprint import pprint
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 표 데이터 형식 맞추기: 천의 자리 구분자 등
def format_values(df):
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

# 마크 다운 copy 기능
def to_markdown_with_ellipsis(df, n=5):
    if len(df) <= n * 2:
        return df.to_markdown(index=False)
    else:
        head = df.head(n)
        tail = df.tail(n)
        ellipsis = pd.DataFrame([["..."] * len(df.columns)], columns=df.columns)
        combined = pd.concat([head, ellipsis, tail], ignore_index=True)
        return combined.to_markdown(index=False)
 
# event_time_index를 실제 시간으로 변경
def event_time_id_to_date(df):
    return df.with_columns(
        (pl.col('event_time_index').cast(pl.UInt64)  * 1_000_000).cast(pl.Datetime())
    )
    
def time_to_date(df, column):
    return df.with_columns(
        (pl.col(column).cast(pl.UInt64)  * 1_000_000).cast(pl.Datetime())
    )
    
def join_cat_table(df: pl.DataFrame, cat_table, columns: str|list = ['category_1']) -> pl.DataFrame:
        return df.join(cat_table.select(['category_id'] + columns), on='category_id', how='left') 
    
def join_cat_table_kr(df: pl.DataFrame, cat_table, cat_kr_table, columns: str|list = ['category_1']) -> pl.DataFrame:
    tmp = cat_table.select(['category_codes', 'category_id']).join(cat_kr_table, on='category_codes', how='left')
    return df.join(tmp.select(['category_id'] + columns), on='category_id', how='left')
