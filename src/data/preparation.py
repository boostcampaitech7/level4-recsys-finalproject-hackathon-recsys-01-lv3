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
    df_sorted = filtered_df.sort([user_col, time_col])
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
    behaviors_map: dict = None
):
    """
    행동(behavior)별로 user-item CSR 행렬을 생성하되,
    df 내 user_id, item_id를 0-based로 재매핑한 뒤 행렬 생성.
        {"view": CSR, "cart": CSR, "purchase": CSR}
    
    Args:
        - df: (user_id, item_id, behavior) DataFrame
        - num_users, num_items: 행렬 크기를 결정하기 위한 값 (max_id + 1)
        - behavior_map: {1:'view',2:'cart',3:'purchase'}와 같이 behavior를 구분하는 딕셔너리
                   -> 실제 구현에서는 {1:'view', ...}처럼 써도 되고, 그냥 숫자로만 구분해도 됨
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
    
    # 먼저 user_id, item_id 재매핑
    df_u, user_map = remap_ids(df, "user_id")
    df_ui, item_map = remap_ids(df_u, "item_id")
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
# 7) BPR Dataset for negative sampling
################################################

class BPRTrainDataset(Dataset):
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
        while len(neg_items) < self.neg_size:
            neg_candidates = np.random.randint(0, self.num_items, size=self.neg_size * 2)
            neg_mask = (self.mat[u].toarray().flatten()[neg_candidates] == 0)  # numpy 변환 후 인덱싱
            valid_neg = neg_candidates[neg_mask]
            neg_items.extend(valid_neg.tolist())
            neg_items = neg_items[:self.neg_size]
        
        # 만약 충분한 부정 샘플이 없으면 0으로 채움
        if len(neg_items) < self.neg_size:
            neg_items = neg_items + [0]*(self.neg_size - len(neg_items))
        
        return {
            "user": torch.LongTensor([u]),
            "pos_item": torch.LongTensor([i]),
            "neg_item": torch.LongTensor(neg_items)
        }
    
