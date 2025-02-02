# src/train.py

"""
학습 및 평가 스크립트

- train_epoch: 단일 에폭 동안 모델을 학습
- evaluate: 모델을 평가하여 Recall@K 및 NDCG@K 계산
"""

import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.mbgcn import MBGCN
from src.data.preparation import (
    load_parquet_data,
    filter_users,
    split_train_test,
    build_user_item_matrices,
    build_item_item_matrices_transpose,
    build_user_behavior_dict
)
from src.loss import bpr_loss

def train_one_epoch(
    model: MBGCN,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float
):
    """
    Epoch 단위 학습
    1) model.encode() => 전체 임베딩 1회 계산 (스파스 연산)
    2) 각 미니배치별로 forward (dot-product 기반), BPR Loss
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    if model.alpha_mode == "per_user":
        model.calc_user_count()
    
    # 1) 메시지 패싱 (Epoch-level)
    with torch.no_grad():
        model.embedding_cached.fill_(0)  # 혹시 이미 캐싱됐다면 초기화
        model.encode()                   # => user_latent, item_latent, s_item_list 계산
    
    progress = tqdm(train_loader, desc="Training 🏋️‍♂️", leave=False)
    for batch_idx, batch in enumerate(progress):
        user_ids = batch["user"].squeeze().to(device)
        pos_item_ids = batch["pos_item"].squeeze().to(device)
        neg_item_ids = batch["neg_item"].to(device)  
        
        batch_size, neg_size = neg_item_ids.size()
        user_ids_expanded = user_ids.unsqueeze(1).repeat(1, neg_size).view(-1)
        neg_item_ids_flat = neg_item_ids.view(-1)
        
        # forward
        pos_scores = model(user_ids, pos_item_ids)  # (batch_size,)
        neg_scores = model(user_ids_expanded, neg_item_ids_flat)  # (batch_size*neg_size,)

        pos_scores_expanded = pos_scores.unsqueeze(1).repeat(1, neg_size).view(-1)
        
        optimizer.zero_grad()
        loss = bpr_loss(pos_scores_expanded, neg_scores, model, beta)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss