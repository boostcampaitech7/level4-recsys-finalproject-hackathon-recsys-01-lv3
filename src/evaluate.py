import torch
import torch.profiler
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.sparse as sp
from src.models.mbgcn import MBGCN
from src.utils.metrics import Recall, Precision, MRR, NDCG
import numpy as np
from typing import List, Dict
import polars as pl

# src/evaluate.py 수정 예시

def evaluate(
    model: MBGCN,
    data_loader: DataLoader,
    device: torch.device,
    K: List[int] = [10, 20, 40, 80],
    ui_mats_train: Dict[str, sp.csr_matrix] = None,  # 학습 데이터의 사용자-아이템 매트릭스
    skip_encode: bool = True
):
    """
    모델을 평가하는 함수.
    
    Args:
        - model (MBGCN): 학습된 MBGCN 모델
        - data_loader (DataLoader): 평가 데이터 로더
        - device (torch.device): 디바이스
        - K (List[int]): 평가할 K 값들의 리스트
        - ui_mats_train (dict): 학습 데이터의 사용자-아이템 매트릭스 (이미 본 아이템 제외에 필요)
        
    Returns:
        metrics (dict): 각 K 값에 대한 메트릭 결과
    """
    """
    0131 변경 사항:
    - skip_encode=True => model.encode() 미호출 (이미 epoch-level로 최신화된 상태라고 가정)
    - already-seen items => row-based .indices => -inf
    - single-pass metric calc (hit_matrix)
    - user sampling => data_loader를 미리 샘플링된 데이터로 구성
    """
    model.eval()
    
    # 1) (옵션) encode 스킵 or 호출
    if not skip_encode:
        model.embedding_cached.fill_(0)
        model.encode()
    
    recall_metrics = {k: Recall(k) for k in K}
    precision_metrics = {k: Precision(k) for k in K}
    mrr_metrics = {k: MRR(k) for k in K}
    ndcg_metrics = {k: NDCG(k) for k in K}
    max_k = max(K)
    
    purchase_csr = None
    if ui_mats_train is not None and 'purchase' in ui_mats_train:
        # CSR
        purchase_csr = ui_mats_train['purchase']
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            user_ids = batch['user'].squeeze().to(device)          # shape=(batch_size,)
            pos_item_ids = batch['pos_item'].squeeze().to(device)  # shape=(batch_size,)
            
            batch_size = user_ids.size(0)
            
            # 2) get_scores => shape=(batch_size, num_items)
            scores = model.get_scores(user_ids)  # already epoch-level cached
            
            # 3) 이미 본 아이템 제외 => row-based .indices
            if purchase_csr is not None:
                # in-place
                for i in range(batch_size):
                    u_id = user_ids[i].item()
                    items_u = purchase_csr[u_id].indices  # np.array
                    scores[i, items_u] = float('-inf')
            
            # 4) single-pass metric calc
            # topK => (batch_size, max_k)
            _, topk = torch.topk(scores, max_k, dim=1, largest=True, sorted=True)  # shape=(batch_size, max_k)
            hits_matrix = (topk == pos_item_ids.unsqueeze(1))  # bool (batch_size, max_k)
            
            # DCG matrix, MRR matrix
            rank_array = torch.arange(1, max_k+1, device=device).float()  # [1..max_k]
            log2_ranks = torch.log2(rank_array + 1.0)
            hits_float = hits_matrix.float()
            dcg_matrix = hits_float / log2_ranks  # shape=(batch_size, max_k)
            
            for k_ in K:
                # slice
                hits_k = hits_matrix[:, :k_].any(dim=1).float()  # shape=(batch_size,)
                recall_metrics[k_].update(hits_k)
                precision_val = hits_k / k_
                precision_metrics[k_].update(precision_val)

                # MRR => sum of hits[i,j]/(j+1) for j=0..k_-1
                mrr_val = torch.zeros(batch_size, device=device)
                for j in range(k_):
                    # topk[:, j], check == pos_item_ids => if True => 1/(j+1)
                    hit_j = (topk[:, j] == pos_item_ids)
                    mrr_val += hit_j.float()/(j+1)
                mrr_metrics[k_].update(mrr_val)

                # NDCG => DCG / IDCG
                dcg_k = dcg_matrix[:, :k_].sum(dim=1)
                idcg_k = torch.ones_like(dcg_k)  # single relevant doc => DCG=1, real formula ~ 1/log2(2)=1
                ndcg_val = dcg_k/idcg_k
                ndcg_metrics[k_].update(ndcg_val)
                    
    # Compute final metrics
    metrics = {}
    for k in K:
        metrics[k] = {
            'Recall': recall_metrics[k].compute(),
            'Precision': precision_metrics[k].compute(),
            'MRR': mrr_metrics[k].compute(),
            'NDCG': ndcg_metrics[k].compute()
        }
    
    model.train()
    return metrics


def evaluate_with_profiler(
    model: MBGCN,
    data_loader: DataLoader,
    device: torch.device,
    K: List[int] = [10, 20, 40, 80],
    ui_mats_train: Dict[str, sp.csr_matrix] = None
):
    model.eval()
    recall_metrics = {k: Recall(k) for k in K}
    precision_metrics = {k: Precision(k) for k in K}
    mrr_metrics = {k: MRR(k) for k in K}
    ndcg_metrics = {k: NDCG(k) for k in K}
    
    with torch.no_grad(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            user_ids = batch['user'].squeeze().to(device)
            pos_item_ids = batch['pos_item'].squeeze().to(device)
            
            batch_size = user_ids.size(0)
            
            # Debugging: 배치 크기 출력
            print(f"Batch {batch_idx+1}: batch_size={batch_size}")
            
            # Get scores for all items
            scores = model.get_scores(user_ids)
            
            # 이미 본 아이템 제외
            if ui_mats_train is not None:
                user_ids_np = user_ids.cpu().numpy()
                purchase_mat = ui_mats_train['purchase']
                
                # Get all purchase item indices for the batch users
                purchase_submat = purchase_mat[user_ids_np]
                purchase_coo = purchase_submat.tocoo()
                
                # Debugging: purchase_coo의 크기 출력
                print(f"  Batch {batch_idx+1}: purchase_coo.row.shape={purchase_coo.row.shape}, purchase_coo.col.shape={purchase_coo.col.shape}")
                
                
                # Convert to torch tensors
                # Fix: Correctly calculate flat indices
                rows = torch.LongTensor(purchase_coo.row * model.num_items + purchase_coo.col).to(device)
                # cols = torch.LongTensor(purchase_coo.col).to(device)
                
                # Set scores of already interacted items to -inf
                scores = scores.view(-1)
                scores[rows] = -float('inf')
                scores = scores.view(batch_size, model.num_items)
            
            # Get top max(K) predictions
            max_k = max(K)
            _, topk = torch.topk(scores, max_k, dim=1, largest=True, sorted=True)  # shape=(batch_size, max_k)
            
            # Calculate metrics
            for k in K:
                topk_k = topk[:, :k]  # shape=(batch_size, k)
                
                # Recall@K: 1 if positive item is in topk, else 0
                hits = (topk_k == pos_item_ids.unsqueeze(1)).any(dim=1)  # Bool tensor
                recall_metrics[k].update(hits.float())
                
                # Precision@K: hits / k
                precision = hits.float() / k
                precision_metrics[k].update(precision)
                
                # MRR@K: 1 / rank if hit, else 0
                reciprocal_ranks = torch.zeros(batch_size, device=device)
                for i in range(k):
                    hit = (topk_k[:, i] == pos_item_ids)
                    reciprocal_ranks += hit.float() / (i + 1)
                mrr_metrics[k].update(reciprocal_ranks)
                
                # NDCG@K: DCG / IDCG
                dcg = torch.zeros(batch_size, device=device)
                for i in range(k):
                    hit = (topk_k[:, i] == pos_item_ids).float()
                    dcg += hit / torch.log2(torch.tensor(i + 2, dtype=torch.float, device=device))
                
                # IDCG: ideal DCG is 1 / log2(2) = 1.0 if there's at least one relevant item
                idcg = torch.ones(batch_size, device=device)
                
                # NDCG
                ndcg = dcg / idcg
                ndcg_metrics[k].update(ndcg)
    
    # 프로파일 결과 출력
    prof.export_chrome_trace("evaluate_trace.json")
    print("Profiler trace saved to evaluate_trace.json")
    
    # Compute final metrics
    metrics = {}
    for k in K:
        metrics[k] = {
            'Recall': recall_metrics[k].compute(),
            'Precision': precision_metrics[k].compute(),
            'MRR': mrr_metrics[k].compute(),
            'NDCG': ndcg_metrics[k].compute()
        }
    
    model.train()
    return metrics