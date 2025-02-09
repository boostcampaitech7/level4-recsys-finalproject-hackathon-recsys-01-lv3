import os
import sys
import torch
import polars as pl
import gc
from tqdm import tqdm
from colorama import init, Fore, Style
from argparse import Namespace
from src.models.mbgcn2 import MBGCN
from src.data.dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader

def score_prep(model_path="MBGCN_lr3e-5_L21e-4_dim128", eval_batch_size=16384):
    print(Fore.CYAN + Style.BRIGHT + ">>> Loading Dataset & Model ...")
    project_root = os.path.join(os.path.expanduser("~"), "Hackathon")
    DATA_PATH = os.path.join(project_root, "src", "data", "MBGCN")
    dataset_name = "data/final"
    SAVE_PATH = os.path.join(DATA_PATH, "data/final", "trained_models")
    relations = "view,cart"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = TrainDataset(
        data_path=DATA_PATH,
        dataset_name=dataset_name,
        relations=relations,
        neg_sample_size=2,
        debug_sample=False
    )
    total_dataset = TestDataset(
        data_path=DATA_PATH,
        dataset_name=dataset_name,
        trainset=train_dataset,
        task="total"
    )
    total_loader = DataLoader(
        total_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    args = Namespace(
        model="MBGCN",
        data_path=DATA_PATH,
        dataset_name=dataset_name,
        relations=relations,
        valid_batch_size=eval_batch_size,         # 테스트용 배치 사이즈
        train_batch_size=4096,
        epoch=400,                 # 짧은 epoch 수로 테스트
        patience=10,
        lr=1e-4,
        L2_norm=1e-4,            # MBGCN에서는 L2_norm이라는 이름으로 사용됨
        embedding_size=128,
        create_embeddings=True,  # 새 임베딩을 생성 (사전학습 임베딩 사용 안함)
        pretrain_path="",        # create_embeddings가 True이면 사용하지 않음
        pretrain_frozen=False,   # 사용되지 않음
        mgnn_weight=[1.0, 1.0],    # 각 행동에 대한 가중치 (예시)
        lamb=0.5,
        node_dropout=0.2,
        message_dropout=0.2,
        device=DEVICE,
        save_path=SAVE_PATH
    )
    
    # num_user, num_items 계산
    num_users, num_items = total_dataset.num_users, total_dataset.num_items
    
    save_path = os.path.join(SAVE_PATH, model_path, "total_model.pth")
    state_dict = torch.load(save_path, map_location=torch.device(DEVICE), weights_only=True)
    model = MBGCN(args, train_dataset, device=DEVICE)
    model.load_state_dict(state_dict)
    return model, total_loader, num_items, num_users, DEVICE

def mbgcn_scores_version_A(model_path="MBGCN_lr3e-5_L21e-4_dim128", eval_batch_size=16384, topk=100_000):
    """
    버전 B: topk_scores는 CPU, topk_users는 GPU에서 유지.
    """
    model, total_loader, num_items, num_users, DEVICE = score_prep(model_path, eval_batch_size)
    gc.collect()

    print(Fore.GREEN + Style.BRIGHT + ">>> Initializing Top-K tensors (scores on CPU, users on GPU)...")
    topk_scores = torch.full((num_items, topk), -float('inf'), device="cpu")
    topk_users  = torch.full((num_items, topk), -1, device=DEVICE, dtype=torch.long)
    
    progress = tqdm(total_loader, desc=Fore.MAGENTA + "Top-K Evaluating", ncols=80)
    model.eval()
    with torch.no_grad():
        for batch in progress:
            # 모델 예측은 GPU에서 수행
            user_ids = batch[0].to(DEVICE)  # (batch_size,)
            batch_scores = model.evaluate(user_ids)  # (batch_size, num_items) → GPU
            
            # 예측 결과를 CPU로 옮겨서 처리
            batch_scores_cpu = batch_scores.cpu()
            del batch_scores
            
            # 전치: (num_items, batch_size) → CPU 상에서 처리
            batch_scores_t = batch_scores_cpu.t()
            del batch_scores_cpu
            
            # 사용자 인덱스 정보 생성: 먼저 CPU로 옮긴 후, 병합 후 GPU로 이동
            batch_users_cpu = user_ids.cpu()
            batch_users_t = batch_users_cpu.expand(num_items, -1)
            del user_ids, batch_users_cpu
            
            # CPU에서 topk_scores와 현재 배치 점수를 병합
            combined_scores = torch.cat([topk_scores, batch_scores_t], dim=1)
            del batch_scores_t
            
            # GPU에 있는 topk_users를 CPU로 옮겨 병합
            topk_users_cpu = topk_users.cpu()
            combined_users = torch.cat([topk_users_cpu, batch_users_t], dim=1)
            del batch_users_t, topk_users_cpu
            
            # CPU에서 top-k 선택
            new_topk_scores, indices = torch.topk(combined_scores, k=topk, dim=1, largest=True, sorted=True)
            new_topk_users = torch.gather(combined_users, 1, indices)
            del combined_scores, combined_users, indices
            
            # 갱신: topk_scores는 CPU에 그대로 업데이트, topk_users는 GPU로 다시 올림.
            topk_scores = new_topk_scores
            topk_users = new_topk_users.to(DEVICE)
            del new_topk_scores, new_topk_users
            
            gc.collect()
    print(Fore.GREEN + Style.BRIGHT + ">>> Preparing Top-K Dictionary...")
    # 최종 결과: topk_users를 GPU에서 CPU로 옮겨 딕셔너리 생성
    topk_users_cpu = topk_users.cpu()
    item_topk_dict = {item: topk_users_cpu[item].tolist() for item in range(num_items)}
    
    del topk_scores, topk_users, topk_users_cpu
    gc.collect()
    
    return item_topk_dict

def mbgcn_scores_version_B(model_path="MBGCN_lr3e-5_L21e-4_dim128", eval_batch_size=16384, topk=10_000):
    # 버전 A: topk_scores (부동소수점 텐서)와 topk_users 모두 GPU에 할당
    model, total_loader, num_items, num_users, DEVICE = score_prep(model_path, eval_batch_size)
    gc.collect()

    print(Fore.GREEN + Style.BRIGHT + ">>> Initializing Top-K tensors on GPU...")
    topk_scores = torch.full((num_items, topk), -float('inf'), device=DEVICE)
    topk_users  = torch.full((num_items, topk), -1, device=DEVICE, dtype=torch.long)
    
    progress = tqdm(total_loader, desc=Fore.MAGENTA + "Top-K Evaluating", ncols=80)
    model.eval()
    with torch.no_grad():
        for batch in progress:
            user_ids = batch[0].to(DEVICE)  # (batch_size,)
            batch_scores = model.evaluate(user_ids)  # (batch_size, num_items) → GPU
            
            # 전치: (num_items, batch_size)
            batch_scores_t = batch_scores.t()
            del batch_scores  # GPU 메모리 해제
            
            # 사용자 인덱스 정보 생성 (GPU에서 유지)
            batch_users_t = user_ids.expand(num_items, -1)
            del user_ids
            
            # 기존 top-k와 현재 배치 결과 병합 (GPU 텐서끼리)
            combined_scores = torch.cat([topk_scores, batch_scores_t], dim=1)
            combined_users  = torch.cat([topk_users, batch_users_t], dim=1)
            del batch_scores_t, batch_users_t
            
            # GPU에서 top-k 선택
            new_topk_scores, indices = torch.topk(combined_scores, k=topk, dim=1, largest=True, sorted=True)
            new_topk_users = torch.gather(combined_users, 1, indices)
            del combined_scores, combined_users, indices
            
            topk_scores = new_topk_scores
            topk_users = new_topk_users
            
            gc.collect()
            torch.cuda.empty_cache()
    
    print(Fore.GREEN + Style.BRIGHT + ">>> Preparing Top-K Dictionary...")
    # 최종 결과: topk_users를 CPU로 전송
    topk_users_cpu = topk_users.cpu()
    item_topk_dict = {item: topk_users_cpu[item].tolist() for item in range(num_items)}
    
    del topk_scores, topk_users, topk_users_cpu
    gc.collect()
    
    return item_topk_dict