# src/train_manager.py

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from colorama import init, Fore, Style
import random

from src.data.dataset import TrainDataset, TestDataset
from src.models.mbgcn2 import MF, MBGCN
from src.utils.metrics import Recall, Precision, NDCG, MRR
from src.loss import bpr_loss

# colorama 초기화 
init(autoreset=True)

class TrainManager:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        # 데이터셋 로드 (TrainDataset, TestDataset)
        print(Fore.CYAN + Style.BRIGHT + ">>> Loading dataset ...")
        self.train_dataset = TrainDataset(
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            relations=args.relations,
            neg_sample_size=2,
            debug_sample=False
        )
        self.train_eval_dataset = TestDataset(
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            trainset=self.train_dataset,
            task="train"   # train.txt 파일을 읽어들임
        )
        self.valid_dataset = TestDataset(
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            trainset=self.train_dataset,
            task="validation"
        )
        self.test_dataset = TestDataset(
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            trainset=self.train_dataset,
            task="test"
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=4
        )
        self.train_eval_loader = DataLoader(
            self.train_eval_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=4
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=4
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=4
        )
        self.num_users = self.train_dataset.num_users
        self.num_items = self.train_dataset.num_items

        # 모델 선택
        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        print(Fore.GREEN + Style.BRIGHT + f">>> Model [{args.model}] loaded on {self.device}")
        
    def _build_model(self):
        if self.args.model == "MF":
            model = MF(
                self.args, self.train_dataset, self.device
            )
        elif self.args.model == 'MBGCN': 
            model = MBGCN(self.args, self.train_dataset, self.device)
        else:
            raise ValueError(Fore.RED + f"Unknown model type: {self.args.model}")
        return model
    
    def _evaluate_loader(self, loader):
        self.model.eval()
        # if self.args.model == "MBGCN":
        #     self.model.encode()
        k_values = [10, 20, 40]
        recall_metrics = {k: Recall(topk=k) for k in k_values}
        ndcg_metrics = {k: NDCG(topk=k) for k in k_values}
        
        progress = tqdm(loader, desc=Fore.MAGENTA + "Evaluating", ncols=80)
        with torch.no_grad():
            for batch in progress:
                # batch: (user_idx, ground_truth, train_mask)
                user_ids = batch[0].to(self.device)                     # (batch_size,)
                ground_truth = batch[1].to(self.device)                 # (batch_size, num_items)
                train_mask = batch[2].to(self.device)                   # (batch_size, num_items)
                
                scores = self.model.evaluate(user_ids)                # (batch_size, num_items)
                scores[train_mask.bool()] = float('-inf')
                
                for k in k_values:
                    # topk_items: (batch_size, k) 예측 인덱스
                    _, topk_items = torch.topk(scores, k, dim=1)        # (batch_size, k)
                    # 디버깅: topk_items의 최대 인덱스와 ground_truth shape 출력
                    max_index = topk_items.max().item()
                    num_items = ground_truth.size(1)
                    if max_index >= num_items:
                        print(f"DEBUG: max_index in topk_items: {max_index}, num_items: {num_items}")
                    # print(f"DEBUG: topk_items: {topk_items}")
                    # print(f"DEBUG: topk_items.min(): {topk_items.min().item()}, topk_items.max(): {topk_items.max().item()}")
                    # ground_truth.gather(1, topk_items) => (batch_size, k)
                    # 각 사용자에 대해, topk_items 위치의 값이 1이면 hit
                    hits = (ground_truth.gather(1, topk_items) > 0).any(dim=1).float()
                    recall_metrics[k].update(hits)
                    
                    batch_size = user_ids.size(0)
                    ndcg_vals = torch.zeros(batch_size, device=self.device)
                    # 각 사용자별로 첫 hit의 rank를 계산
                    for i in range(batch_size):
                        # ground_truth.gather(1, topk_items[i].unsqueeze(1)) => (k,1), squeeze => (k,)
                        rel = ground_truth[i].gather(0, topk_items[i])
                        nonzero = (rel > 0).nonzero(as_tuple=False)
                        if nonzero.numel() > 0:
                            rank = nonzero[0].item() + 1  # 1-based rank
                            ndcg_vals[i] = 1.0 / np.log2(rank + 1)
                    ndcg_metrics[k].update(ndcg_vals)
        metrics = {}
        for k in k_values:
            metrics[f"Recall@{k}"] = recall_metrics[k].compute()
            metrics[f"NDCG@{k}"] = ndcg_metrics[k].compute()
        return metrics
    
    def _evaluate_loader_train(self, loader):
        self.model.eval()
        k_values = [10, 20, 40]
        recall_metrics = {k: Recall(topk=k) for k in k_values}
        ndcg_metrics = {k: NDCG(topk=k) for k in k_values}
        
        progress = tqdm(loader, desc=Fore.MAGENTA + "Evaluating (Train)", ncols=80)
        with torch.no_grad():
            for batch in progress:
                user_ids = batch[0].to(self.device)                     # (batch_size,)
                ground_truth = batch[1].to(self.device)                 # (batch_size, num_items)
                # **여기서는 train_mask를 사용하지 않고 평가**
                scores = self.model.evaluate(user_ids)                # (batch_size, num_items)
                
                for k in k_values:
                    _, topk_items = torch.topk(scores, k, dim=1)        # (batch_size, k)
                    hits = (ground_truth.gather(1, topk_items) > 0).any(dim=1).float()
                    recall_metrics[k].update(hits)
                    
                    batch_size = user_ids.size(0)
                    ndcg_vals = torch.zeros(batch_size, device=self.device)
                    for i in range(batch_size):
                        rel = ground_truth[i].gather(0, topk_items[i])
                        nonzero = (rel > 0).nonzero(as_tuple=False)
                        if nonzero.numel() > 0:
                            rank = nonzero[0].item() + 1
                            ndcg_vals[i] = 1.0 / np.log2(rank + 1)
                    ndcg_metrics[k].update(ndcg_vals)
        metrics = {}
        for k in k_values:
            metrics[f"Recall@{k}"] = recall_metrics[k].compute()
            metrics[f"NDCG@{k}"] = ndcg_metrics[k].compute()
        return metrics
    
    def train_epoch(self):
        # MBGCN 모델의 경우, 학습 epoch 시작 전에 encode()를 호출해서 임베딩을 캐시합니다.
        # if self.args.model == "MBGCN":
        #     self.model.encode()
        self.model.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=Fore.YELLOW + "Training", ncols=80)
        for batch in progress:
            self.optimizer.zero_grad()
            user_ids = batch[0].squeeze().to(self.device).view(-1) # shape: [batch_size]
            pos_neg = batch[1].to(self.device)  # shape: [batch_size, 2]
            pos_item_ids = pos_neg[:, 0].view(-1, 1)
            neg_item_ids = pos_neg[:, 1].view(-1, 1)
            # 디버깅: 배치 내 negative 샘플들의 통계 출력
            neg_values = neg_item_ids.cpu().numpy()
            # print(f"DEBUG: Neg sample stats: min={neg_values.min()}, max={neg_values.max()}, mean={neg_values.mean()}")
            
            pos_scores, pos_L2 = self.model(user_ids, pos_item_ids)
            neg_scores, neg_L2 = self.model(user_ids, neg_item_ids)
            
            loss = bpr_loss(pos_scores, neg_scores, pos_L2, neg_L2, self.args.train_batch_size, self.args.L2_norm)
            loss.backward()
            # self.model 내의 behavior_alpha gradient 확인
            # print("[DEBUG] behavior_alpha grad:", self.model.behavior_alpha.grad)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            total_loss += loss.item()
            
            # # TrainManager.train_epoch()의 루프 안에 추가해 보세요.
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.norm().item())
            
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def train(self):
        best_recall = -float('inf')
        patience_count = 0
        best_state = None

        print(Fore.CYAN + Style.BRIGHT + ">>> Start training ...")
        for epoch in range(1, self.args.epoch + 1):
            start_time = time.time()
            train_loss = self.train_epoch() # 학습 데이터셋에서 계산한 train loss
            # train_metrics = self._evaluate_loader_train(self.train_eval_loader)
            valid_metrics = self._evaluate_loader(self.valid_loader) # validation 데이터셋 평가
            epoch_time = time.time() - start_time
            log = (f"{Fore.WHITE}{Style.BRIGHT}Epoch {epoch:03d}/{self.args.epoch:03d} | "
                   f"{Fore.YELLOW}Train Loss: {train_loss:.6f}\n"
                #    + " | ".join([f"{Fore.GREEN}Train Recall@{k}: {train_metrics[f'Recall@{k}']:.4f}" for k in [10, 20, 40]])
                   + "\n" + " | ".join([f"{Fore.BLUE}Valid Recall@{k}: {valid_metrics[f'Recall@{k}']:.4f}" for k in [10, 20, 40]])
                   + "\n" + " | ".join([f"{Fore.CYAN}Valid NDCG@{k}: {valid_metrics[f'NDCG@{k}']:.4f}" for k in [10, 20, 40]])
                   + f" | {Fore.MAGENTA}Time: {epoch_time:.2f}s")
            print(log)
            # Early stopping 기준: Recall@40 on validation
            if valid_metrics["Recall@40"] > best_recall:
                best_recall = valid_metrics["Recall@40"]
                best_state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.args.patience:
                    print(Fore.RED + f">>> Early stopping at epoch {epoch}, best Validation Recall@40: {best_recall:.4f}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state, strict=False)
        
        # 최종 테스트 평가
        final_test_metrics = self._evaluate_loader(self.test_loader)
        print(Fore.GREEN + Style.BRIGHT + ">>> Final Test Metrics:")
        for k in sorted(final_test_metrics.keys()):
            print(f"{k}: {final_test_metrics[k]:.4f}")
        
        # 모델 저장: 최종 테스트 평가 이후에 모델을 저장합니다.
        # 예를 들어, self.args.save_dir 경로에 best_model.pth 파일로 저장
        save_path = self.args.save_path
        os.makedirs(save_path, exist_ok=True)
        save_dir = os.path.join(save_path, "best_model.pth")
        torch.save(self.model.state_dict(), save_dir)
        print(Fore.GREEN + Style.BRIGHT + f">>> Model saved at {save_dir}")

        # 최종 평가 지표(예: Recall@40)를 반환하도록 수정
        return final_test_metrics["Recall@40"]

        
        