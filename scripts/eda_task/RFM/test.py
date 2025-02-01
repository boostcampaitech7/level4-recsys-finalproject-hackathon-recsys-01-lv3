import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import polars as pl
from time import time
import pandas as pd
import numpy as np
from src.utils import load_file

def main():
    base_path = "~/Hackathon/src/data/raw"
    log_data, user_data, item_data, cat_table, cat_table_kr, brand_table, filter_session = load_file.load_data_lazy(base_path, preprocess=False)

    log_data = log_data.collect()
    user_data = user_data.collect()
    item_data = item_data.collect()
    
    print(log_data)
    print(user_data)
    print(item_data)
    

if __name__ == '__main__':
    start = time()
    main()
    print(f"총 걸린 시간: {time() - start:.2f}s") 