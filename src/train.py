# src/train.py

"""
학습 및 평가 스크립트

- train_epoch: 단일 에폭 동안 모델을 학습
- evaluate: 모델을 평가하여 Recall@K 및 NDCG@K 계산
"""

import torch
from tqdm import tqdm
import numpy as np
from src.utils.metrics import Recall, NDCG, MRR, Precision

def train_epoch(model, dataloader, optimizer, device, beta):
    """
    단일 에폭 동안 모델을 학습하는 함수 (BPR Loss + L2 정규화)
    
    Args:
        - model (torch.nn.Module): 학습할 모델
        - dataloader (torch.utils.data.DataLoader): 학습 데이터로더
        - optimizer (torch.optim.Optimizer): 최적화 알고리즘
        - device (str): "cuda" 또는 "cpu"
        - beta (float): L2 정규화 정도
        
    Returns:
        - (float): 에폭 동안의 평균 손실 값
    """
    model.train() # 모델을 학습 모드로 설정
    epoch_loss = 0.0
    num_batches = len(dataloader)
    
    # progress bar 설정
    progress = tqdm(dataloader, desc="Training 🏋️‍♂️", leave=False)
    
    for batch in progress:
        # 배치 데이터 
        user = batch["user"].squeeze().to(device)         # shape: (batch_size,)
        pos_item = batch["pos_item"].squeeze().to(device) # shape: (batch_size,)
        neg_item = batch["neg_item"].squeeze().to(device) # shape: (batch_size, neg_size)
        
        optimizer.zero_grad() # 옵티마이저의 기울기 초기화
        
        # 긍정 샘플 점수 예측
        pos_scores = model(user, pos_item) # shape: (batch_size,)
        # 부정 샘플 점수 예측
        neg_scores = model(user, neg_item) # shape: (batch_size, neg_size)
        neg_scores = neg_scores.mean(dim=1)  # shape: (batch_size,)
        
        # BPR Loss 계산: -log(sigmoid(pos - neg))
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        # L2 정규화 계산: 모든 파라미터의 제곱합
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)
        loss += beta * l2_reg # 정규화 항 추가
        
        # 역전파 및 최적화
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())  # 현재 배치의 손실 표시

    average_loss = epoch_loss / num_batches
    return average_loss
    
def evaluate(model, dataloader, device, topk=10):
    """
    모델을 평가하여 Recall@K, Precision@K, NDCG@K, MRR@K 를 계산하는 함수

    Args:
        model (torch.nn.Module): 평가할 모델
        dataloader (torch.utils.data.DataLoader): 평가 데이터로더
        device (str): "cuda" 또는 "cpu"
        k (int): 상위 k개의 추천을 고려
    
    Returns:
        dict: {'Recall@K': value, 'Precision@K': value, 'NDCG@K': value, 'MRR@K': value}
    """
    model.eval()  # 모델을 평가 모드로 설정
    
    # 메트릭 인스턴스 생성
    recall_metric = Recall(topk)
    precision_metric = Precision(topk)
    ndcg_metric = NDCG(topk)
    mrr_metric = MRR(topk)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating 📊", leave=False):
            user = batch['user'].squeeze().to(device)          # shape: (batch_size,)
            pos_item = batch['pos_item'].squeeze().to(device)  # shape: (batch_size,)
            neg_item = batch['neg_item'].squeeze().to(device)  # shape: (batch_size, neg_size)

            # 긍정 샘플 점수 예측
            pos_scores = model(user, pos_item)  # shape: (batch_size,)

            # 부정 샘플 점수 예측
            neg_scores = model(user, neg_item)  # shape: (batch_size, neg_size)
            neg_scores = neg_scores.mean(dim=1)  # shape: (batch_size,)
            
            # 점수 차이 계산
            score_diff = pos_scores - neg_scores  # shape: (batch_size,)

            # 시그모이드 함수 적용하여 확률로 변환
            probs = torch.sigmoid(score_diff)  # shape: (batch_size,)