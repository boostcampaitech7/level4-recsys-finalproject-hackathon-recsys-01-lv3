import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

################################################
# 1) Re-mapping user/item ID to index
################################################
def remap_ids(df: pl.DataFrame, col_name: str):
    """
    Map unique values in the specified column to consecutive integers (0-based)
    and return the new DataFrame with a mapped column and the mapping dictionary.

    Args:
        df (pl.DataFrame): Input DataFrame.
        col_name (str): Column name to remap.

    Returns:
        Tuple[pl.DataFrame, Dict[Any, int]]: A tuple containing:
            - The updated DataFrame with a new column '{col_name}_mapped'.
            - A dictionary mapping original IDs to new 0-based indices.
    """
    unique_vals = df[col_name].unique() 
    val_map = {old_id: new_id for new_id, old_id in enumerate(unique_vals.to_list())}
    df_new = df.with_columns(
        pl.col(col_name).replace(val_map).alias(f"{col_name}_mapped")
    )
    return df_new, val_map

################################################
# 2) User Filtering (Remove users with < 3 interactions)
################################################
def filter_users(
    df: pl.DataFrame, user_col: str = "user_id", min_interactions: int = 3
) -> Tuple[pl.DataFrame, List[int]]:
    """
    Filter out users with fewer than min_interactions.

    Args:
        df (pl.DataFrame): User-item interaction DataFrame.
        user_col (str, optional): Name of the user ID column. Defaults to "user_id".
        min_interactions (int, optional): Minimum required interactions per user. Defaults to 3.

    Returns:
        Tuple[pl.DataFrame, List[int]]: A tuple containing:
            - The filtered DataFrame.
            - A list of valid user IDs.
    """
    user_counts = df.group_by(user_col).agg(pl.len().alias("count"))
    valid_users = user_counts.filter(pl.col("count") >= min_interactions)["user_id"].to_list()
    filtered_df = df.filter(pl.col(user_col).is_in(valid_users))
    print(f"[filter_users] 전체 유저 수: {df.select(pl.col(user_col).n_unique()).item(0,0):,}")
    print(f"[filter_users] 필터링된 유저 수 (≥ {min_interactions} interactions): {len(valid_users):,}")
    print(f"[filter_users] 제거된 유저 수: {df.select(pl.col(user_col).n_unique()).item(0,0) - len(valid_users):,}")
    return filtered_df, valid_users

################################################
# 3) Data Splitting (Leave-One-Out)
################################################
def split_train_test(
    df: pl.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    test_ratio: float = 0.2,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split data into training and testing sets using a Leave-One-Out approach on a per-user basis.

    Args:
        df (pl.DataFrame): Interaction DataFrame.
        user_col (str, optional): Column name for user IDs. Defaults to "user_id".
        item_col (str, optional): Column name for item IDs. Defaults to "item_id".
        time_col (str, optional): Column name for timestamp. Defaults to "timestamp".
        test_ratio (float, optional): Proportion of interactions to use as test data. Defaults to 0.2.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
            - The training DataFrame.
            - The testing DataFrame.
    """
    df_sorted = df.sort([user_col, time_col])
    user_count = df_sorted.group_by(user_col).agg(pl.len().alias('count'))
    user_count = user_count.with_columns(
        (pl.col('count') * test_ratio).ceil().cast(pl.Int64).alias('count_test')
    )

    df_with_test_size = df_sorted.join(user_count, on=user_col)
    df_with_test_size = df_with_test_size.with_columns(
        pl.col(time_col).rank(method='dense').over(user_col).alias('rank')
    )

    df_with_test_size = df_with_test_size.with_columns(
        (pl.col('rank') > (pl.col('count') - pl.col('count_test'))).alias('is_test')
    )
    train_df = df_with_test_size.filter(~pl.col("is_test")).select([user_col, item_col, time_col, 'behavior'])
    test_df = df_with_test_size.filter(pl.col("is_test")).select([user_col, item_col, time_col, 'behavior'])
    print(f"[split_train_test] Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

################################################
# 4) Parquet Reader
################################################
def load_parquet_data(file_path: str, sample_size: int = 100_000) -> pl.DataFrame:
    """
    Load a Parquet file and sample rows if necessary.

    Args:
        file_path (str): Path to the Parquet file.
        sample_size (int, optional): Maximum number of rows to sample. Defaults to 100_000.

    Returns:
        pl.DataFrame: Loaded (and possibly sampled) DataFrame.
    """
    df = pl.read_parquet(file_path)
    print(f"[load_parquet_data] Full shape = {df.shape}")
    if sample_size and not None and sample_size < df.shape[0]:
        df = df.sample(n=sample_size, seed=42)
    return df

def get_basic_stats(df: pl.DataFrame):
    """
    Print basic statistics (max and unique counts) for user_id, item_id, and behavior.

    Args:
        df (pl.DataFrame): DataFrame containing the data.
    """
    user_max = df.select(pl.col("user_id").max()).item(0,0)
    item_max = df.select(pl.col("item_id").max()).item(0,0)
    unique_users = df.select(pl.col("user_id").n_unique()).item(0,0)
    unique_items = df.select(pl.col("item_id").n_unique()).item(0,0)
    unique_behaviors = df.select(pl.col("behavior").n_unique()).item(0,0)
    print("== Basic Stats ==")
    print(f"UserID max: {user_max},  Unique Users: {unique_users:,}")
    print(f"ItemID max: {item_max},  Unique Items: {unique_items:,}")
    print(f"Unique Behaviors: {unique_behaviors:,}")
    

################################################
# 5) Build user-item adjacency by behavior
################################################
def build_user_item_matrices(
    df: pl.DataFrame,
    behaviors_map: Dict[int, str] = None,
    user_map: Dict[int, int] = None,
    item_map: Dict[int, int] = None,
) -> Tuple[Dict[str, sp.csr_matrix], int, int, Dict[int, int], Dict[int, int]]:
    """
    Build user-item CSR matrices for each behavior after remapping user and item IDs.

    Args:
        df (pl.DataFrame): DataFrame with columns 'user_id', 'item_id', and 'behavior'.
        behaviors_map (dict, optional): Mapping from behavior code to behavior name.
            Defaults to {1: "view", 2: "cart", 3: "purchase"}.
        user_map (dict, optional): Existing user mapping dictionary.
        item_map (dict, optional): Existing item mapping dictionary.

    Returns:
        Tuple containing:
            - Dictionary mapping behavior names to user-item CSR matrices.
            - Number of users.
            - Number of items.
            - User mapping dictionary.
            - Item mapping dictionary.
    """
    if behaviors_map is None:
        behaviors_map = { 1: "view", 2: "cart", 3: "purchase"}
    if user_map is None:
        df_u, user_map = remap_ids(df, "user_id")
    else:
        df_u = df.with_columns([
            pl.col("user_id").replace(user_map).alias("user_id_mapped")
        ])
    if item_map is None:
        df_ui, item_map = remap_ids(df_u, "item_id")
    else:
        df_ui = df_u.with_columns([
            pl.col("item_id").replace(item_map).alias("item_id_mapped")
        ])
        
    del df, df_u
    
    num_users = df_ui["user_id_mapped"].max() + 1
    num_items = df_ui["item_id_mapped"].max() + 1
    
    result = {}
    for bcode, bname in behaviors_map.items():
        sub = df_ui.filter(pl.col('behavior') == bcode).select(['user_id_mapped', 'item_id_mapped'])
        if sub.is_empty():
            print(f"[build_user_item_matrices] No rows for {bname}")
            mat_empty = sp.csr_matrix((num_users, num_items), dtype=np.float32)
            result[bname] = mat_empty
            continue
        rows = sub["user_id_mapped"].to_numpy()
        cols = sub["item_id_mapped"].to_numpy()
        data = np.ones(len(rows), dtype=np.float32)
        mat_coo = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(num_users, num_items)
        )
        mat_csr = mat_coo.tocsr()
        result[bname] = mat_csr
        print(f"[build_user_item_matrices] {bname}: shape={mat_csr.shape}, nnz={mat_csr.nnz:,}")
    return result, num_users, num_items, user_map, item_map

################################################
# 6) Build item-item adjacency
################################################
def build_item_item_matrices_transpose(
    ui_mats: Dict[str, sp.csr_matrix], threshold: int = 1
) -> Dict[str, sp.csr_matrix]:
    """
    Build item-item CSR matrices using the transpose-based method (mat.T @ mat).

    Args:
        ui_mats (Dict[str, sp.csr_matrix]): Dictionary of user-item CSR matrices keyed by behavior.
        threshold (int, optional): Minimum co-occurrence count to retain an edge. Defaults to 1.

    Returns:
        Dict[str, sp.csr_matrix]: Dictionary of item-item CSR matrices.
    """
    item_item_dict = {}
    for bname, mat in ui_mats.items():
        if mat.nnz == 0:
            item_item_dict[bname] = sp.csr_matrix((mat.shape[1], mat.shape[1]))
            continue
        coo = (mat.T @ mat).tocoo()
        # 필터링을 먼저 적용
        mask = (coo.row != coo.col) & (coo.data >= threshold)
        rows, cols, vals = coo.row[mask], coo.col[mask], coo.data[mask]
        item_item_dict[bname] = sp.csr_matrix((vals, (rows, cols)), shape=coo.shape)
    return item_item_dict

################################################
# 7) Build user behavior Dictionary
################################################
def build_user_behavior_dict(
    df: pl.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
    behaviors: List[str],
    event_table: Dict[str, str] = None,
) -> Dict[str, Dict[int, List[int]]]:
    """
    Build a dictionary mapping each behavior to a dictionary mapping user IDs to lists of item IDs.

    Args:
        df (pl.DataFrame): DataFrame with columns 'user_id', 'item_id', and 'behavior'.
        user_map (Dict[int, int]): Mapping from original user IDs to new indices.
        item_map (Dict[int, int]): Mapping from original item IDs to new indices.
        behaviors (List[str]): List of behavior names.
        event_table (Dict[str, str], optional): Mapping from behavior codes to names.
            Defaults to {"1": "view", "2": "cart", "3": "purchase"}.

    Returns:
        Dict[str, Dict[int, List[int]]]: Dictionary of the form {behavior: {user_id: [item_ids]}}.
    """
    if event_table is None:
        event_table = {"1": "view", "2": "cart", "3": "purchase"}
    df = df.with_columns(
        pl.col('behavior').cast(pl.String).replace(event_table)
    )
    user_bhv_dict = defaultdict(lambda: defaultdict(list))
    for row in df.to_dicts():
        orig_user = row['user_id']
        orig_item = row['item_id']
        behavior = row['behavior']
        mapped_user = user_map.get(orig_user, None)
        mapped_item = item_map.get(orig_item, None)
        
        if mapped_user is None or mapped_item is None:
            continue
        user_bhv_dict[behavior][mapped_user].append(mapped_item)
    return user_bhv_dict

################################################
# 8) BPR Dataset for negative sampling
################################################
class BPRDataset(Dataset):
    """
    BPR Dataset for negative sampling.

    Scenario A:
        - Positive: (user, item) pairs where the user purchased the item.
        - Negative: (user, item) pairs where the user did NOT purchase the item.
        (Items with only view/cart interactions are considered negatives.)
    """
    def __init__(self, user_item_mat: sp.csr_matrix, num_items: int, neg_size: int=1):
        """
        Args:
            user_item_mat (sp.csr_matrix): CSR matrix of shape (num_users, num_items) for positive interactions.
            num_items (int): Total number of items.
            neg_size (int, optional): Number of negative samples per positive example. Defaults to 1.
        """
        super().__init__()
        self.mat = user_item_mat
        self.num_users = self.mat.shape[0]
        self.num_items = num_items
        self.neg_size = neg_size
        
        coo = self.mat.tocoo()
        self.pos_u = coo.row
        self.pos_i = coo.col
        self.length = len(self.pos_u)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Get a sample consisting of a user, a positive item, and negative items.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys 'user', 'pos_item', and 'neg_item'.
        """
        u = self.pos_u[idx]
        i = self.pos_i[idx]
        neg_items = []
        user_interactions = set(self.mat[u].indices)
        while len(neg_items) < self.neg_size:
            neg_candidates = np.random.randint(0, self.num_items, size=self.neg_size * 2)
            neg_candidates = np.clip(neg_candidates, 0, self.num_items - 1)
            valid_neg = [item for item in neg_candidates if item not in user_interactions]
            neg_items.extend(valid_neg)
            neg_items = list(dict.fromkeys(neg_items))[:self.neg_size]
        
        if len(neg_items) < self.neg_size:
            neg_items = neg_items + [0]*(self.neg_size - len(neg_items))
        
        return {
            "user": torch.LongTensor([u]),
            "pos_item": torch.LongTensor([i]),
            "neg_item": torch.LongTensor(neg_items)
        }
    
################################################
# 9) Load Tmall Data
################################################
import pandas as pd

def load_tmall_data(
    root_dir: str = "src/data/MBGCN/Tmall"
) -> Tuple[
    int,
    int,
    Dict[str, sp.csr_matrix],
    Dict[str, sp.csr_matrix],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Dict[int, List[int]]]
]:
    """
    Load Tmall dataset and build the necessary data structures.

    Expected Tmall data structure:
        - data_size.txt: Contains num_users and num_items.
        - buy.txt, cart.txt, click.txt, collect.txt: Interaction files.
        - item/ directory: Contains item_{behavior}.pth files.
        - train.txt, validation.txt, test.txt: Data splits.

    Returns:
        Tuple containing:
            - num_users (int)
            - num_items (int)
            - ui_mats_train (dict): User-item CSR matrices for training.
            - ii_mats (dict): Item-item CSR matrices from item graph files.
            - train_df (pd.DataFrame): Training pairs (user_id, item_id).
            - valid_df (pd.DataFrame): Validation pairs.
            - test_df (pd.DataFrame): Test pairs.
            - user_bhv_dict (dict): Mapping {behavior: {user_id: [item_ids]}}.
    """
    size_path = os.path.join(root_dir, "data_size.txt")
    with open(size_path, "r") as f:
        line = f.readline().strip()
        num_users, num_items = line.split()
        num_users, num_items = int(num_users), int(num_items)
    print(f"[load_tmall_data_split] num_users={num_users}, num_items={num_items}")
    
    behavior_list = ["buy", "cart", "click", "collect"]
    ui_mats = {}

    for bhv in behavior_list:
        txt_path = os.path.join(root_dir, f"{bhv}.txt")
        user_arr = []
        item_arr = []
        with open(txt_path, "r") as bf:
            for row in bf:
                u_str, i_str = row.strip().split()
                u = int(u_str)
                i = int(i_str)
                user_arr.append(u)
                item_arr.append(i)
        data = np.ones(len(user_arr), dtype=np.float32)
        coo = sp.coo_matrix((data, (user_arr, item_arr)), shape=(num_users, num_items))
        ui_mats[bhv] = coo.tocsr()

    ii_mats = {}
    for bhv in behavior_list:
        pth_path = os.path.join(root_dir, "item", f"item_{bhv}.pth")
        if os.path.exists(pth_path):
            item_graph_tensor = torch.load(pth_path)  
            item_graph_np = item_graph_tensor.numpy().astype(np.float32)  
            ii_coo = sp.coo_matrix(item_graph_np)
            ii_csr = ii_coo.tocsr()
            ii_mats[bhv] = ii_csr
        else:
            ii_mats[bhv] = sp.csr_matrix((num_items, num_items), dtype=np.float32)

    def load_pairs_to_df(txt_file):
        pairs = []
        with open(txt_file, "r") as f:
            for row in f:
                u_str, i_str = row.strip().split()
                u = int(u_str)
                i = int(i_str)
                pairs.append((u, i))
        df = pd.DataFrame(pairs, columns=["user_id", "item_id"])
        return df

    train_path = os.path.join(root_dir, "train.txt")
    val_path   = os.path.join(root_dir, "validation.txt")
    test_path  = os.path.join(root_dir, "test.txt")

    train_df = load_pairs_to_df(train_path)
    valid_df = load_pairs_to_df(val_path)
    test_df  = load_pairs_to_df(test_path)

    user_bhv_dict = {}
    for bhv in behavior_list:
        mat = ui_mats[bhv]
        bhv_dict = {}
        indptr = mat.indptr
        indices = mat.indices
        for u in range(num_users):
            start = indptr[u]
            end = indptr[u+1]
            item_list = indices[start:end]
            bhv_dict[u] = list(item_list)
        user_bhv_dict[bhv] = bhv_dict

    return (
        num_users, 
        num_items,
        ui_mats,
        ii_mats,
        train_df,
        valid_df,
        test_df,
        user_bhv_dict
    )
    
def load_tmall_data_split(root_dir: str = "src/data/MBGCN/Tmall") -> Tuple[
    int,
    int,
    Dict[str, sp.csr_matrix],
    Dict[str, sp.csr_matrix],
    Dict[str, sp.csr_matrix],
    Dict[str, sp.csr_matrix],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Dict[int, List[int]]],
    Dict[str, Dict[int, List[int]]],
    Dict[str, Dict[int, List[int]]],
]:
    """
    Load and split Tmall data into training, validation, and test sets, and build
    the corresponding user-item and item-item matrices along with user behavior dictionaries.

    Tmall data structure:
        - data_size.txt: Contains num_users and num_items.
        - train.txt, validation.txt, test.txt: Interaction pairs.
        - buy.txt, cart.txt, click.txt, collect.txt: Interaction files for all behaviors.
        - item/ directory: Contains item_{behavior}.pth files.

    Returns:
        Tuple containing:
            - num_users (int)
            - num_items (int)
            - ui_mats_train (dict): User-item matrices for training.
            - ui_mats_valid (dict): User-item matrices for validation.
            - ui_mats_test (dict): User-item matrices for testing.
            - ii_mats (dict): Item-item matrices from item graphs.
            - train_df (pd.DataFrame): Training interactions.
            - valid_df (pd.DataFrame): Validation interactions.
            - test_df (pd.DataFrame): Test interactions.
            - user_bhv_dict_train (dict): User behavior dictionary for training.
            - user_bhv_dict_valid (dict): User behavior dictionary for validation.
            - user_bhv_dict_test (dict): User behavior dictionary for testing.
    """

    size_path = os.path.join(root_dir, "data_size.txt")
    with open(size_path, "r") as f:
        line = f.readline().strip()
        num_users_str, num_items_str = line.split()
        num_users = int(num_users_str)
        num_items = int(num_items_str)
    print(f"[load_tmall_data_split] num_users={num_users}, num_items={num_items}")

    def load_pairs(txt_file):
        pairs = []
        with open(txt_file, "r") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts=line.split()
                if len(parts)<2: continue
                u_str,i_str=parts
                u=int(u_str); i=int(i_str)
                pairs.append((u,i))
        return pairs

    train_txt = os.path.join(root_dir, "train.txt")
    valid_txt = os.path.join(root_dir, "validation.txt")
    test_txt  = os.path.join(root_dir, "test.txt")

    train_pairs = load_pairs(train_txt)
    valid_pairs = load_pairs(valid_txt)
    test_pairs  = load_pairs(test_txt)

    train_set = set(train_pairs)
    valid_set = set(valid_pairs)
    test_set  = set(test_pairs)

    train_df = pd.DataFrame(train_pairs, columns=["user_id","item_id"])
    valid_df = pd.DataFrame(valid_pairs, columns=["user_id","item_id"])
    test_df  = pd.DataFrame(test_pairs,  columns=["user_id","item_id"])

    print(f"[load_tmall_data_split] #train={len(train_df)}, #valid={len(valid_df)}, #test={len(test_df)}")

    behavior_list = ["buy","cart","click","collect"]
    ui_mats_train = {}
    ui_mats_valid = {}
    ui_mats_test  = {}

    for bhv in behavior_list:
        txt_file = os.path.join(root_dir, f"{bhv}.txt")
        if not os.path.exists(txt_file):
            ui_mats_train[bhv] = sp.csr_matrix((num_users,num_items), dtype=np.float32)
            ui_mats_valid[bhv] = sp.csr_matrix((num_users,num_items), dtype=np.float32)
            ui_mats_test[bhv]  = sp.csr_matrix((num_users,num_items), dtype=np.float32)
            continue
        row_train=[]; col_train=[]
        row_valid=[]; col_valid=[]
        row_test =[]; col_test=[]

        with open(txt_file,"r") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts=line.split()
                if len(parts)<2: continue
                u_str,i_str=parts
                u=int(u_str); i=int(i_str)
                if (u,i) in train_set:
                    row_train.append(u); col_train.append(i)
                elif (u,i) in valid_set:
                    row_valid.append(u); col_valid.append(i)
                elif (u,i) in test_set:
                    row_test.append(u); col_test.append(i)
                else:
                    pass

        data_train = np.ones(len(row_train), dtype=np.float32)
        data_valid = np.ones(len(row_valid), dtype=np.float32)
        data_test  = np.ones(len(row_test),  dtype=np.float32)
        mat_train = sp.coo_matrix((data_train,(row_train,col_train)), shape=(num_users,num_items)).tocsr()
        mat_valid = sp.coo_matrix((data_valid,(row_valid,col_valid)), shape=(num_users,num_items)).tocsr()
        mat_test  = sp.coo_matrix((data_test, (row_test, col_test)),  shape=(num_users,num_items)).tocsr()
        ui_mats_train[bhv] = mat_train
        ui_mats_valid[bhv] = mat_valid
        ui_mats_test[bhv]  = mat_test
        print(f"behavior={bhv}, train nnz={mat_train.nnz}, valid nnz={mat_valid.nnz}, test nnz={mat_test.nnz}")

    def build_user_bhv_dict(mats_dict):
        ret={}
        for bhv,mat in mats_dict.items():
            ret_b = {}
            indptr=mat.indptr
            indices=mat.indices
            for u in range(num_users):
                start=indptr[u]
                end=indptr[u+1]
                items=indices[start:end]
                ret_b[u]=list(items)
            ret[bhv]=ret_b
        return ret

    user_bhv_dict_train = build_user_bhv_dict(ui_mats_train)
    user_bhv_dict_valid = build_user_bhv_dict(ui_mats_valid)
    user_bhv_dict_test  = build_user_bhv_dict(ui_mats_test)

    ii_mats = {}
    for bhv in behavior_list:
        pth_path = os.path.join(root_dir, "item", f"item_{bhv}.pth")
        if os.path.exists(pth_path):
            item_graph_tensor = torch.load(pth_path)  # shape=[num_items, num_items]
            item_graph_np = item_graph_tensor.numpy().astype(np.float32)
            coo = sp.coo_matrix(item_graph_np)
            csr_ = coo.tocsr()
            ii_mats[bhv] = csr_
        else:
            ii_mats[bhv] = sp.csr_matrix((num_items,num_items), dtype=np.float32)

    return (
        num_users,
        num_items,
        ui_mats_train,
        ui_mats_valid,
        ui_mats_test,
        ii_mats,
        train_df,
        valid_df,
        test_df,
        user_bhv_dict_train,
        user_bhv_dict_valid,
        user_bhv_dict_test
    )