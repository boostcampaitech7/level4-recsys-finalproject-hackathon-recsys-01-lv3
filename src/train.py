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

# def train(
#     model: MBGCN,
#     train_loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     beta: float
# ):
#     """
#     모델 학습 함수.
    
#     Args:
#         model (MBGCN): 학습할 모델
#         train_loader (DataLoader): 훈련 데이터 로더
#         optimizer (Optimizer): 옵티마이저
#         device (torch.device): 디바이스
#         beta (float): L2 정규화 계수
#     """
#     model.train() # 모델을 학습 모드로 설정
#     total_loss = 0.0
#     num_batches = len(train_loader)
    
#     # progress bar 설정
#     progress = tqdm(train_loader, desc="Training 🏋️‍♂️", leave=False)
    
#     for batch_idx, batch in enumerate(progress):
#         # 배치 데이터 
#         user_ids = batch["user"].squeeze().to(device)         # shape: (batch_size,)
#         pos_item_ids = batch["pos_item"].squeeze().to(device) # shape: (batch_size,)
#         neg_item_ids = batch["neg_item"].to(device) # shape: (batch_size, neg_size)
        
#         batch_size, neg_size = neg_item_ids.size()
        
#         # 사용자 IDs를 (batch_size * neg_size,) 형태로 확장
#         user_ids_expanded = user_ids.unsqueeze(1).repeat(1, neg_size).view(-1)  # shape=(batch_size * neg_size,)
#         neg_item_ids_flat = neg_item_ids.view(-1)                            # shape=(batch_size * neg_size,)
        
#         # 모델을 통해 부정 샘플 점수 계산
#         neg_scores = model(user_ids_expanded, neg_item_ids_flat)            # shape=(batch_size * neg_size,)
        
#         # 모델을 통해 긍정 샘플 점수 계산
#         pos_scores = model(user_ids, pos_item_ids)                          # shape=(batch_size,)
#         pos_scores_expanded = pos_scores.unsqueeze(1).repeat(1, neg_size).view(-1)  # shape=(batch_size * neg_size,)
        
#         optimizer.zero_grad() # 옵티마이저의 기울기 초기화
        
#         # 디버그: 텐서 크기 출력
#         # print(f"Batch {batch_idx}:")
#         # print(f"  user_ids shape: {user_ids.shape}")               # (batch_size,)
#         # print(f"  pos_item_ids shape: {pos_item_ids.shape}")       # (batch_size,)
#         # print(f"  neg_item_ids shape: {neg_item_ids.shape}")       # (batch_size, neg_size)
#         # print(f"  user_ids_expanded shape: {user_ids_expanded.shape}")  # (batch_size * neg_size,)
#         # print(f"  neg_item_ids_flat shape: {neg_item_ids_flat.shape}")  # (batch_size * neg_size,)
#         # print(f"  pos_scores shape: {pos_scores.shape}")           # (batch_size,)
#         # print(f"  pos_scores_expanded shape: {pos_scores_expanded.shape}")  # (batch_size * neg_size,)
#         # print(f"  neg_scores shape: {neg_scores.shape}")           # (batch_size * neg_size,)
        
#         # BPR Loss 계산
#         try:
#             loss = bpr_loss(pos_scores_expanded, neg_scores, model, beta)
#         except Exception as e:
#             print(f"Error in batch {batch_idx}: {e}")
#             raise
        
#         # 역전파 및 최적화
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         progress.set_postfix(loss=loss.item())  # 현재 배치의 손실 표시

#     avg_loss = total_loss / num_batches
#     return avg_loss

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
    
    # 1) 메시지 패싱 (Epoch-level)
    model.embedding_cached.fill_(0)  # 혹시 이미 1로 되어있다면 초기화
    model.encode()                   # => 캐싱된 임베딩(user_latent, item_latent, s_item_list)
    
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
        progress.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss