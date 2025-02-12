import os
import sys
import time
import random
import gc
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import polars as pl
from tqdm import tqdm
from colorama import init, Fore, Style

from src.models.mbgcn import MBGCN
from src.data.dataset import TrainDataset, TestDataset

def score_prep(model_path="MBGCN_lr3e-5_L21e-4_dim128", eval_batch_size=16384):
    """
    Load dataset and model state, and prepare model, total loader, and dataset info.

    Args:
        model_path (str): Path string specifying the model directory.
        eval_batch_size (int): Batch size for evaluation.

    Returns:
        tuple: (model, total_loader, num_items, num_users, DEVICE)
            - model (torch.nn.Module): The loaded MBGCN model.
            - total_loader (DataLoader): DataLoader for the total dataset.
            - num_items (int): Number of items in the dataset.
            - num_users (int): Number of users in the dataset.
            - DEVICE (str): Device used for computation.
    """
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
        valid_batch_size=eval_batch_size,       
        train_batch_size=4096,
        epoch=400,                
        patience=10,
        lr=1e-4,
        L2_norm=1e-4,            
        embedding_size=128,
        create_embeddings=True,  
        pretrain_path="",        
        pretrain_frozen=False,   
        mgnn_weight=[1.0, 1.0],   
        lamb=0.5,
        node_dropout=0.2,
        message_dropout=0.2,
        device=DEVICE,
        save_path=SAVE_PATH
    )
    
    num_users, num_items = total_dataset.num_users, total_dataset.num_items
    
    save_path = os.path.join(SAVE_PATH, model_path, "total_model.pth")
    state_dict = torch.load(
        save_path_full,
        map_location=torch.device(DEVICE),
        weights_only=True
    )
    model = MBGCN(args, train_dataset, device=DEVICE)
    model.load_state_dict(state_dict)
    return model, total_loader, num_items, num_users, DEVICE

def mbgcn_scores_version_A(model_path="MBGCN_lr3e-5_L21e-4_dim128", 
                           eval_batch_size=16384, topk=100_000):
    """
    Compute Top-K scores on CPU (scores) and GPU (users) using version A strategy.

    Args:
        model_path (str): Path string specifying the model directory.
        eval_batch_size (int): Batch size for evaluation.
        topk (int): Number of top scores to keep.

    Returns:
        dict: A dictionary mapping each item index to a list of top-K user indices.
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
            user_ids = batch[0].to(DEVICE)  
            batch_scores = model.evaluate(user_ids)  
            
            batch_scores_cpu = batch_scores.cpu()
            del batch_scores
            
            batch_scores_t = batch_scores_cpu.t()
            del batch_scores_cpu
            
            batch_users_cpu = user_ids.cpu()
            batch_users_t = batch_users_cpu.expand(num_items, -1)
            del user_ids, batch_users_cpu
            
            combined_scores = torch.cat([topk_scores, batch_scores_t], dim=1)
            del batch_scores_t
            
            topk_users_cpu = topk_users.cpu()
            combined_users = torch.cat([topk_users_cpu, batch_users_t], dim=1)
            del batch_users_t, topk_users_cpu
            
            new_topk_scores, indices = torch.topk(combined_scores, k=topk, dim=1, largest=True, sorted=True)
            new_topk_users = torch.gather(combined_users, 1, indices)
            del combined_scores, combined_users, indices
            
            topk_scores = new_topk_scores
            topk_users = new_topk_users.to(DEVICE)
            del new_topk_scores, new_topk_users
            
            gc.collect()
    print(Fore.GREEN + Style.BRIGHT + ">>> Preparing Top-K Dictionary...")
    topk_users_cpu = topk_users.cpu()
    item_topk_dict = {item: topk_users_cpu[item].tolist() for item in range(num_items)}
    
    del topk_scores, topk_users, topk_users_cpu
    gc.collect()
    
    return item_topk_dict

def mbgcn_scores_version_B(model_path="MBGCN_lr3e-5_L21e-4_dim128", 
                           eval_batch_size=16384, topk=10_000):
    """
    Compute Top-K scores on GPU using version B strategy.

    Args:
        model_path (str): Path string specifying the model directory.
        eval_batch_size (int): Batch size for evaluation.
        topk (int): Number of top scores to keep.

    Returns:
        dict: A dictionary mapping each item index to a list of top-K user indices.
    """
    model, total_loader, num_items, num_users, DEVICE = score_prep(model_path, eval_batch_size)
    gc.collect()

    print(Fore.GREEN + Style.BRIGHT + ">>> Initializing Top-K tensors on GPU...")
    topk_scores = torch.full((num_items, topk), -float('inf'), device=DEVICE)
    topk_users  = torch.full((num_items, topk), -1, device=DEVICE, dtype=torch.long)
    
    progress = tqdm(total_loader, desc=Fore.MAGENTA + "Top-K Evaluating", ncols=80)
    model.eval()
    with torch.no_grad():
        for batch in progress:
            user_ids = batch[0].to(DEVICE) 
            batch_scores = model.evaluate(user_ids)  
            
            batch_scores_t = batch_scores.t()
            del batch_scores
            
            batch_users_t = user_ids.expand(num_items, -1)
            del user_ids
        
            combined_scores = torch.cat([topk_scores, batch_scores_t], dim=1)
            combined_users  = torch.cat([topk_users, batch_users_t], dim=1)
            del batch_scores_t, batch_users_t
            
            new_topk_scores, indices = torch.topk(combined_scores, k=topk, dim=1, largest=True, sorted=True)
            new_topk_users = torch.gather(combined_users, 1, indices)
            del combined_scores, combined_users, indices
            
            topk_scores = new_topk_scores
            topk_users = new_topk_users
            
            gc.collect()
            torch.cuda.empty_cache()
    
    print(Fore.GREEN + Style.BRIGHT + ">>> Preparing Top-K Dictionary...")
    topk_users_cpu = topk_users.cpu()
    item_topk_dict = {item: topk_users_cpu[item].tolist() for item in range(num_items)}
    
    del topk_scores, topk_users, topk_users_cpu
    gc.collect()
    
    return item_topk_dict