# src/data/preparation.py

"""
src/data/preparation.py

Step 2:
    - Load Parquet
    - Build user-item adjacency matrices (behavior별)
    - Build item-item adjacency matrices (co-behavior) with 3 methods:
        (1) Transpose-based: mat.T @ mat
        (2) User-driven construction
        (3) Top-K approximate
    - Provide a BPR Dataset class for negative sampling
"""

import os
import random
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Dict, List

################################################
# 1) Re-mapping user/item ID to index
################################################
def remap_ids(df: pl.DataFrame, col_name: str):
    """
    df[col_name]의 고유값을 0,1,...,(n_unique-1)로 매핑하여, 새 열을 반환하는 함수
    
    예: user_id가 10, 100, 31 이라면 -> {10:0, 31:1, 100:2} 등.
    """
    # 고유값 추출 (정렬 or 순서 상관없이, dict만 consistent 하면 됨)
    unique_vals = df[col_name].unique() # polars Series -> python list
    # 딕셔너리 만들기: old_id -> new_id
    val_map = {old_id: new_id for new_id, old_id in enumerate(unique_vals.to_list())}
    
    # Polars의 replace를 사용해 매핑 적용
    df_new = df.with_columns(
        pl.col(col_name).replace(val_map).alias(f"{col_name}_mapped")
    )
    return df_new, val_map

################################################
# 2) User Filtering (Remove users with < 3 interactions)
################################################

def filter_users(df: pl.DataFrame, user_col: str='user_id', min_interactions: int=3):
    """
    유저별 상호작용 수가 min_interactions 미만인 유저를 필터링합니다.
    
    Args:
        df (pl.DataFrame): 전체 사용자-아이템-행동 데이터.
        user_col (str): 사용자 ID 컬럼 이름.
        min_interactions (int): 최소 상호작용 수.
    
    Returns:
        filtered_df (pl.DataFrame): 필터링된 데이터프레임.
        filtered_users (List[int]): 필터링된 유저 ID 리스트.
    """
    # 사용자별 상호작용 수 계산
    user_counts = df.group_by(user_col).agg(pl.len().alias("count"))
    
    # 상호작용 수가 min_interactions 이상인 유저 필터링
    valid_users = user_counts.filter(pl.col("count") >= min_interactions)["user_id"].to_list()
    
    # 필터링된 데이터
    filtered_df = df.filter(pl.col(user_col).is_in(valid_users))
    
    print(f"[filter_users] 전체 유저 수: {df.select(pl.col(user_col).n_unique()).item(0,0):,}")
    print(f"[filter_users] 필터링된 유저 수 (≥ {min_interactions} interactions): {len(valid_users):,}")
    print(f"[filter_users] 제거된 유저 수: {df.select(pl.col(user_col).n_unique()).item(0,0) - len(valid_users):,}")
    
    return filtered_df, valid_users

################################################
# 3) Data Splitting (Leave-One-Out)
################################################

def split_train_test(df: pl.DataFrame, user_col: str="user_id", item_col: str="item_id", time_col: str="timestamp", test_ratio: float=0.2):
    """
    사용자별로 데이터를 훈련 세트와 테스트 세트로 분할합니다 (Leave-One-Out).
    
    Args:
        df (pl.DataFrame): 사용자-아이템-시간 상호작용 데이터.
        user_col (str): 사용자 ID 컬럼 이름.
        item_col (str): 아이템 ID 컬럼 이름.
        time_col (str): 시간 정보 컬럼 이름.
        test_ratio (float): 테스트 세트 비율 (기본값: 0.2).
    
    Returns:
        train_df (pl.DataFrame): 훈련 세트 데이터프레임.
        test_df (pl.DataFrame): 테스트 세트 데이터프레임.
    """
    # 사용자별로 정렬하여 시간 기준으로 분할
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

def load_parquet_data(file_path: str, sample_size: int=100_000):
    """
    Parquet 파일을 로드한 뒤,
    sample_size보다 데이터가 많다면 랜덤 샘플링하여 DataFrame을 반환
    
    Args:
        - file_path: Parquet 경로
        - sample_size: 대규모 데이터 중 일부만 샘플링해서 로드 (테스트용), None이면 전체 로드
    """
    
    df = pl.read_parquet(file_path)
    print(f"[load_parquet_data] Full shape = {df.shape}")
    
    if sample_size and not None and sample_size < df.shape[0]:
        df = df.sample(n=sample_size, seed=42)
    
    return df

def get_basic_stats(df: pl.DataFrame):
    """
    user_id, item_id, behavior의 기본 통계치 출력
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
    behaviors_map: dict = None,
    user_map: dict = None,
    item_map: dict = None
):
    """
    행동(behavior)별로 user-item CSR 행렬을 생성하되,
    df 내 user_id, item_id를 0-based로 재매핑한 뒤 행렬 생성.
        {"view": CSR, "cart": CSR, "purchase": CSR}
    
    Args:
        - df: (user_id, item_id, behavior) DataFrame
        - behaviors_map (dict, optional): 행동 코드에서 행동 이름으로의 매핑
        - user_map (dict, optional): 기존 사용자 매핑
        - item_map (dict, optional): 기존 아이템 매핑
    Returns:
        - dict 형태
            예: {
                    'view': sp.csr_matrix(...),
                    'cart': sp.csr_matrix(...),
                    'purchase': sp.csr_matrix(...)
                }
        - num_users, 
        - num_items
        - user_map: dict
        - item_map: dict
    """
    if behaviors_map is None:
        behaviors_map = { 1: "view", 2: "cart", 3: "purchase"}
    # user_id, item_id 재매핑
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
    
    # 재매핑된 user/item 컬럼 이름 (user_id_mapped, item_id_mapped)
    # 최대 index + 1 = 크기
    num_users = df_ui["user_id_mapped"].max() + 1
    num_items = df_ui["item_id_mapped"].max() + 1
    
    # 각 behavior 별로 user-item 쌍 추출
    # user_id, item_id가 int 형이라고 가정
    result = {}
    for bcode, bname in behaviors_map.items():
        sub = df_ui.filter(pl.col('behavior') == bcode).select(['user_id_mapped', 'item_id_mapped'])
        if sub.is_empty():
            print(f"[build_user_item_matrices] No rows for {bname}")
            mat_empty = sp.csr_matrix((num_users, num_items), dtype=np.float32)
            result[bname] = mat_empty
            continue
        
        # sub를 numpy로 변환
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
# 6) Build item-item adjacency (3 methods)
################################################

## (1) Transpose-based method (mat.T @ mat)
def build_item_item_matrices_transpose(ui_mats: dict, threshold: int=1):
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

### (2) User-driven construction
def build_item_item_matrices_userdriven(
    ui_mats: dict,
    threshold: int=1
):
    """
    User-driven 방식으로 item-item co-occurrence 계산.
    - user-item mat에서 user별로 item 목록을 추출
    - 같은 user가 가진 (i,j) 쌍에 대해 co-occurrence 카운팅
    - i<j 등의 조건을 사용해서 중복 제거
    - co-occurrence >= threshold인 쌍만 남김
    
    Args: 
        ui_mats:  ui_mats: {"view": CSR, "cart": CSR, "purchase": CSR}
        threshold: co-occurrence가 threshold 이상일 때만 남김
    Returns:
        {"view": item-item sparse matrix, "cart": ..., "purchase": ...}
    """
    item_item_dict = {}
    
    for bname, mat in ui_mats.items():
        if mat.nnz == 0:
            item_item_dict[bname] = sp.csr_matrix((mat.shape[1], mat.shape[1]))
            continue
    
        num_items = mat.shape[1]
        cooccur_dict = defaultdict(int)
        
        # CSR에서 row별로 non-zero cols
        # mat.indptr[u], mat.indptr[u+1] 사이의 mat.indices[...]가 user u가 가진 아이템
            # sp.csr_matrix().indices: 행렬에서 값이 0이 아닌 요소의 열 인덱스
            # sp.csr_matrix().indptr: 각 행의 시작점과 끝점을 나타내는 인덱스
        indptr = mat.indptr
        indices = mat.indices
        
        for u in range(mat.shape[0]):
            start = indptr[u]
            end = indptr[u+1]
            items_u = indices[start:end]
            if len(items_u) <= 1:
                continue
            # (i,j) 조합 생성
            # i < j 조건
            items_u_sorted = np.sort(items_u)
            for idx_i in range(len(items_u_sorted)):
                i_id = items_u_sorted[idx_i]
                for idx_j in range(idx_i+1, len(items_u_sorted)):
                    j_id = items_u_sorted[idx_j]
                    cooccur_dict[(i_id, j_id)] += 1
        
        # cooccur_dict -> coo_matrix
        rows_list = []
        cols_list = []
        vals_list = []
        for (i_id, j_id), c in cooccur_dict.items():
            if c >= threshold:
                rows_list.append(i_id)
                cols_list.append(j_id)
                vals_list.append(c)
                # 대칭일 경우 (j_id, i_id)도 추가?
                # -> 보통 item-item 행렬은 대칭이므로 하삼각행렬도도 추가
                rows_list.append(j_id)
                cols_list.append(i_id)
                vals_list.append(c)
        
        rows = np.array(rows_list, dtype=np.int32)
        cols = np.array(cols_list, dtype=np.int32)
        vals = np.array(vals_list, dtype=np.float32)
        
        coo = sp.coo_matrix((vals, (rows, cols)), shape=(num_items, num_items))
        item_item_csr = coo.tocsr()
        print(f"[userdriven] {bname}: shape={item_item_csr.shape}, nnz={item_item_csr.nnz:,}")
        item_item_dict[bname] = item_item_csr
        
    return item_item_dict

### (3) Top-K approximate (user-driven)
def build_item_item_matrices_topk_approx(
    ui_mats: dict,
    k: int=20
):
    """
    User-driven 방식 + 아이템별 top-K 근사
    - 유저별 아이템 목록 -> (i,j) co-occurrence 계산
    - 아이템별 (cooccurrence, j) 리스트 중 상위 K만 남긴다. (co-occ 값 기준 내림차순)
    Note) co-occ 점수가 같다면 j_id 순서로
    """
    
    item_item_dict = {}
    for bname, mat in ui_mats.items():
        if mat.nnz == 0:
            item_item_dict[bname] = sp.csr_matrix((mat.shape[1], mat.shape[1]))
            continue
        
        num_items = mat.shape[1]
        # coocc[item_i] = {item_j: count}
        coocc = [defaultdict(int) for _ in range(num_items)]
        
        indptr = mat.indptr
        indices = mat.indices
        
        # 1) (i,j) 직접 카운팅
        for u in range(mat.shape[0]):
            start = indptr[u]
            end = indptr[u+1]
            items_u = indices[start:end]
            if len(items_u) <= 1:
                continue
            items_u_sorted = np.sort(items_u)
            for idx_i in range(len(items_u_sorted)):
                i_id = items_u_sorted[idx_i]
                for idx_j in range(idx_i+1, len(items_u_sorted)):
                    j_id = items_u_sorted[idx_j]
                    coocc[i_id][j_id] += 1
                    coocc[j_id][i_id] += 1
        
        # 2) 아이템별 Top-K
        rows_list = []
        cols_list = []
        vals_list = []
        for i_id in range(num_items):
            # coocc[i_id]는 dict: {j_id: co-occ_count, ...}
            if not coocc[i_id]:
                continue
            # 내림차순 정렬
            sorted_pairs = sorted(coocc[i_id].items(), key=lambda x: x[1], reverse=True)
            top_k_pairs = sorted_pairs[:k]
            for (j_id, c) in top_k_pairs:
                rows_list.append(i_id)
                cols_list.append(j_id)
                vals_list.append(float(c))
        
        rows = np.array(rows_list, dtype=np.int32)
        cols = np.array(cols_list, dtype=np.int32)
        vals = np.array(vals_list, dtype=np.float32)
        
        coo = sp.coo_matrix((vals, (rows, cols)), shape=(num_items, num_items))
        item_item_csr = coo.tocsr()
        print(f"[topk_approx] {bname}: shape={item_item_csr.shape}, nnz={item_item_csr.nnz:,}")
        item_item_dict[bname] = item_item_csr

    return item_item_dict

################################################
# 7) Build user behavior Dictionary
################################################

def build_user_behavior_dict(df: pl.DataFrame, user_map: Dict[int, int], item_map: Dict[int, int], behaviors: List[str], 
                             event_table: Dict[int, str] = {
        "1": "view", 
        "2": "cart",
        "3": "purchase"
    }) -> Dict[str, Dict[int, List[int]]]:
    """
    사용자별 행동별 아이템 리스트 생성
    
    Returns:
        - user_bhv_dict: {behavior: {user_id: [item_ids]}}
    """
    
    event_table = {
        "1": "view", 
        "2": "cart",
        "3": "purchase"
    }
    df = df.with_columns(
        pl.col('behavior').cast(pl.String).replace(event_table)
    )
    user_bhv_dict = defaultdict(lambda: defaultdict(list))
    for row in df.to_dicts():
        orig_user = row['user_id']
        orig_item = row['item_id']
        behavior = row['behavior']
        
        # 매핑된 인덱스로 변환
        mapped_user = user_map.get(orig_user, None)
        mapped_item = item_map.get(orig_item, None)
        
        if mapped_user is None or mapped_item is None:
            # 매핑되지 않은 사용자 또는 아이템은 무시하거나 로그를 남길 수 있습니다.
            continue
        
        user_bhv_dict[behavior][mapped_user].append(mapped_item)
    
    return user_bhv_dict

################################################
# 8) BPR Dataset for negative sampling
################################################

class BPRDataset(Dataset):
    """
    BPR Dataset (Scenario A)
        - Positive = (user, item) where user purchased item
        - Negative = (user, item) where user did NOT purchase item
        - view/cart only? -> negative (since no purchase)
    """
    def __init__(self, user_item_mat: sp.csr_matrix, num_items: int, neg_size: int=1):
        """
        Args:
            user_item_mat (sp.csr_matrix): (num_users, num_items)
            num_items (int): total item count
            neg_size (int): how many negative samples per positive
        """
        super().__init__()
        self.mat = user_item_mat
        self.num_users = self.mat.shape[0]
        self.num_items = num_items
        self.neg_size = neg_size
        
        # positive pairs: user-item where mat[u,i]==1
        coo = self.mat.tocoo()
        self.pos_u = coo.row
        self.pos_i = coo.col
        self.length = len(self.pos_u) # total # of pos pairs
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Return a dict:
            user = [u]
            pos_item = [i]
            neg_item = [neg_j1, ..., neg_jN]
        """
        u = self.pos_u[idx]
        i = self.pos_i[idx]

        # Negative Sampling 벡터화 (수정)
        neg_items = []
        user_interactions = set(self.mat[u].indices)
        while len(neg_items) < self.neg_size:
            neg_candidates = np.random.randint(0, self.num_items, size=self.neg_size * 2)
            # Ensure neg_candidates are within bounds
            neg_candidates = np.clip(neg_candidates, 0, self.num_items - 1)
            # Check if user has not interacted with neg_candidates
            valid_neg = [item for item in neg_candidates if item not in user_interactions]
            neg_items.extend(valid_neg)
            # Remove duplicates and limit to neg_size
            neg_items = list(dict.fromkeys(neg_items))[:self.neg_size]
        
        # 만약 충분한 부정 샘플이 없으면 0으로 채움
        if len(neg_items) < self.neg_size:
            neg_items = neg_items + [0]*(self.neg_size - len(neg_items))
        
        return {
            "user": torch.LongTensor([u]),
            "pos_item": torch.LongTensor([i]),
            "neg_item": torch.LongTensor(neg_items)
        }
    
################################################################################################
################################################################################################
################################################################################################

import pandas as pd

def load_tmall_data(
    root_dir: str = "src/data/MBGCN/Tmall"
):
    """
    Tmall 데이터(멀티 행동 buy, cart, click, collect + train/test/val)를 로드하여,
    우리의 기존 코드에서 쓰는 자료구조를 생성한다.

    Args:
        root_dir (str): Tmall 데이터가 위치한 디렉토리 (e.g. "src/data/MBGCN/Tmall")
    
    Returns:
        num_users (int)
        num_items (int)
        ui_mats (dict): 예: {"buy": sp.csr_matrix, "cart": sp.csr_matrix, "click":..., "collect":...}
        ii_mats (dict): 예: {"buy": sp.csr_matrix, "cart":..., ...}
        train_df (pd.DataFrame): (user, item) 열
        valid_df (pd.DataFrame)
        test_df  (pd.DataFrame)
        user_bhv_dict (dict): { behavior: { user_id: [item_ids...] } }
    
    Note:
        - txt파일 내 user/item ID가 이미 0-based라고 가정.
        - 만약 1-based라면, 별도 -1 처리 필요.
    """
    # 1) data_size.txt
    size_path = os.path.join(root_dir, "data_size.txt")
    with open(size_path, "r") as f:
        line = f.readline().strip()
        num_users, num_items = line.split()
        num_users, num_items = int(num_users), int(num_items)

    # 2) 행동별 user-item txt => CSR matrix
    behavior_list = ["buy", "cart", "click", "collect"]
    ui_mats = {}

    for bhv in behavior_list:
        txt_path = os.path.join(root_dir, f"{bhv}.txt")
        # 로드
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
        # COO => CSR
        coo = sp.coo_matrix((data, (user_arr, item_arr)), shape=(num_users, num_items))
        ui_mats[bhv] = coo.tocsr()

    # 3) 아이템-아이템 그래프 => ii_mats
    #   item_buy.pth 등 (11953 x 11953) torch int32 텐서
    #   => sparse로 변환 or coo?
    ii_mats = {}
    for bhv in behavior_list:
        pth_path = os.path.join(root_dir, "item", f"item_{bhv}.pth")
        if os.path.exists(pth_path):
            item_graph_tensor = torch.load(pth_path)  # shape=[num_items,num_items], dtype=torch.int32
            # Tensor -> coo -> csr
            item_graph_np = item_graph_tensor.numpy().astype(np.float32)  # (11953,11953)
            # 만약 0/1 (or count) 형식이라면 coo_matrix 변환
            ii_coo = sp.coo_matrix(item_graph_np)
            ii_csr = ii_coo.tocsr()
            ii_mats[bhv] = ii_csr
        else:
            # 혹은 empty
            ii_mats[bhv] = sp.csr_matrix((num_items, num_items), dtype=np.float32)

    # 4) train.txt / validation.txt / test.txt => DataFrame
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

    # 5) user_bhv_dict: {behavior: {u: [i1, i2,...]}, ...}
    user_bhv_dict = {}
    for bhv in behavior_list:
        mat = ui_mats[bhv]  # csr_matrix
        # row-based 접근
        bhv_dict = {}
        # mat.indptr[u], mat.indptr[u+1] 사이가 user u의 col indices
        indptr = mat.indptr
        indices = mat.indices
        for u in range(num_users):
            start = indptr[u]
            end = indptr[u+1]
            item_list = indices[start:end]
            bhv_dict[u] = list(item_list)  # or np.ndarray
        user_bhv_dict[bhv] = bhv_dict

    # 반환
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
    
def load_tmall_data_split(
    root_dir: str = "src/data/MBGCN/Tmall"
):
    """
    Tmall 데이터 구조:
        Tmall/
        ├── item/
        │   ├── item_buy.pth
        │   ├── item_cart.pth
        │   ├── item_click.pth
        │   └── item_collect.pth
        ├── buy.txt
        ├── cart.txt
        ├── click.txt
        ├── collect.txt
        ├── data_size.txt
        ├── test.txt
        ├── train.txt
        └── validation.txt

    1) data_size.txt => num_users, num_items
    2) train.txt, validation.txt, test.txt => (u,i) 쌍
       => 각각 "학습/검증/테스트" 세트에 속한 (u,i) 목록
    3) buy.txt / cart.txt / click.txt / collect.txt => 모든 (u,i)
       => (u,i)가 train에 속하면 ui_mats_train["buy"][u,i] = 1
          valid면 ui_mats_valid["buy"][u,i] = 1, 등등
    4) item_buy.pth / item_cart.pth ... => 아이템-아이템 그래프
       => ii_mats["buy"], etc.
    5) user_bhv_dict_{train,valid,test}: { behavior: { user_id: [item_ids...] } }

    Returns:
        num_users (int)
        num_items (int)

        ui_mats_train (dict): {"buy":csr, "cart":csr, ...}
        ui_mats_valid (dict)
        ui_mats_test  (dict)

        ii_mats (dict): {"buy":csr, "cart":..., ...} from item_buy.pth, etc.

        train_df (pd.DataFrame): (user_id, item_id)
        valid_df (pd.DataFrame)
        test_df  (pd.DataFrame)

        user_bhv_dict_train (dict): {behavior:{u:[i1,i2,...]}}
        user_bhv_dict_valid (dict)
        user_bhv_dict_test  (dict)
    """

    # 1) data_size.txt => num_users, num_items
    size_path = os.path.join(root_dir, "data_size.txt")
    with open(size_path, "r") as f:
        line = f.readline().strip()
        num_users_str, num_items_str = line.split()
        num_users = int(num_users_str)
        num_items = int(num_items_str)
    print(f"[load_tmall_data_split] num_users={num_users}, num_items={num_items}")

    # 2) 읽어서 (u,i) 목록을 세트로 보관 => train_set, valid_set, test_set
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

    # dataframe
    train_df = pd.DataFrame(train_pairs, columns=["user_id","item_id"])
    valid_df = pd.DataFrame(valid_pairs, columns=["user_id","item_id"])
    test_df  = pd.DataFrame(test_pairs,  columns=["user_id","item_id"])

    print(f"[load_tmall_data_split] #train={len(train_df)}, #valid={len(valid_df)}, #test={len(test_df)}")

    # 3) buy.txt / cart.txt / click.txt / collect.txt => behavior별 (user,item)
    #    => (u,i)가 train_set에 있으면 ui_mats_train[behavior][u,i]=1
    #    => valid_set, test_set 마찬가지
    behavior_list = ["buy","cart","click","collect"]
    ui_mats_train = {}
    ui_mats_valid = {}
    ui_mats_test  = {}

    for bhv in behavior_list:
        txt_file = os.path.join(root_dir, f"{bhv}.txt")
        if not os.path.exists(txt_file):
            # empty
            ui_mats_train[bhv] = sp.csr_matrix((num_users,num_items), dtype=np.float32)
            ui_mats_valid[bhv] = sp.csr_matrix((num_users,num_items), dtype=np.float32)
            ui_mats_test[bhv]  = sp.csr_matrix((num_users,num_items), dtype=np.float32)
            continue
        # 읽기
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
                    # (u,i)가 어느 세트에도 없는 => 무시
                    pass

        data_train = np.ones(len(row_train), dtype=np.float32)
        data_valid = np.ones(len(row_valid), dtype=np.float32)
        data_test  = np.ones(len(row_test),  dtype=np.float32)

        # COO => CSR
        mat_train = sp.coo_matrix((data_train,(row_train,col_train)), shape=(num_users,num_items)).tocsr()
        mat_valid = sp.coo_matrix((data_valid,(row_valid,col_valid)), shape=(num_users,num_items)).tocsr()
        mat_test  = sp.coo_matrix((data_test, (row_test, col_test)),  shape=(num_users,num_items)).tocsr()

        ui_mats_train[bhv] = mat_train
        ui_mats_valid[bhv] = mat_valid
        ui_mats_test[bhv]  = mat_test

        print(f"behavior={bhv}, train nnz={mat_train.nnz}, valid nnz={mat_valid.nnz}, test nnz={mat_test.nnz}")

    # 4) user_bhv_dict_{train,valid,test}
    def build_user_bhv_dict(mats_dict):
        """
        mats_dict = {behavior: CSR}
        => {behavior:{u:[items...]}}
        """
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

    # 5) 아이템-아이템 그래프 => ii_mats
    #    item_buy.pth, item_cart.pth, ...
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
            # empty
            ii_mats[bhv] = sp.csr_matrix((num_items,num_items), dtype=np.float32)

    # 반환
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