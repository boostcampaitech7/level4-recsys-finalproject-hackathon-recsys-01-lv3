# src/train_manager.py

import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from colorama import init, Fore, Style

from src.data.dataset import TrainDataset, TestDataset, TotalTrainDataset
from src.models.mbgcn import MF, MBGCN
from src.utils.metrics import Recall, Precision, NDCG, MRR
from src.loss import bpr_loss

init(autoreset=True)

class TrainManager:
    """
    Manager class responsible for handling the training, evaluation, and retraining processes.
    
    Args:
        args (Namespace): Configuration parameters for training.
    """
    def __init__(self, args):
        """
        Initialize the training manager by setting up datasets, dataloaders, model, and optimizer.
        """
        self.args = args
        self.device = torch.device(args.device)
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
            task="train"
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

        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        print(Fore.GREEN + Style.BRIGHT + f">>> Model [{args.model}] loaded on {self.device}")
        
    def _build_model(self):
        """
        Build and return the model based on the model type specified in the arguments.

        Returns:
            torch.nn.Module: The model instance (MF or MBGCN) built according to the configuration.
        """
        if self.args.model == "MF":
            model = MF(self.args, self.train_dataset, self.device)
        elif self.args.model == 'MBGCN': 
            model = MBGCN(self.args, self.train_dataset, self.device)
        else:
            raise ValueError(Fore.RED + f"Unknown model type: {self.args.model}")
        return model
    
    def _evaluate_loader(self, loader):
        """
        Evaluate the model on the provided dataloader by computing recall (and/or ndcg) metrics,
        while applying a train mask.

        Args:
            loader (DataLoader): DataLoader instance for evaluation.

        Returns:
            dict: A dictionary containing computed recall metrics keyed by metric name.
        """
        self.model.eval()
        k_values = [20, 40, 80]
        recall_metrics = {k: Recall(topk=k) for k in k_values}
        ndcg_metrics = {k: NDCG(topk=k) for k in k_values}
        
        progress = tqdm(loader, desc=Fore.MAGENTA + "Evaluating", ncols=80)
        with torch.no_grad():
            for batch in progress:
                user_ids = batch[0].to(self.device)                    
                ground_truth = batch[1].to(self.device)                 
                train_mask = batch[2].to(self.device)                  
                
                scores = self.model.evaluate(user_ids)               
                scores[train_mask.bool()] = float('-inf')
                
                for k in k_values:
                    _, topk_items = torch.topk(scores, k, dim=1)       
                    max_index = topk_items.max().item()
                    num_items = ground_truth.size(1)
                    if max_index >= num_items:
                        print(
                            f"DEBUG: max_index in topk_items: {max_index}, "
                            f"num_items: {num_items}"
                        )
                    
                    hits = (ground_truth.gather(1, topk_items) > 0).any(dim=1).float()
                    recall_metrics[k].update(hits)
        metrics = {}
        for k in k_values:
            metrics[f"Recall@{k}"] = recall_metrics[k].compute()
        return metrics
    
    def _evaluate_loader_train(self, loader):
        """
        Evaluate the model on the training evaluation dataloader without using a train mask.

        Args:
            loader (DataLoader): DataLoader instance for training evaluation.

        Returns:
            dict: A dictionary containing computed recall metrics keyed by metric name.
        """
        self.model.eval()
        k_values = [20, 40, 80]
        recall_metrics = {k: Recall(topk=k) for k in k_values}
        
        progress = tqdm(loader, desc=Fore.MAGENTA + "Evaluating (Train)", ncols=80)
        with torch.no_grad():
            for batch in progress:
                user_ids = batch[0].to(self.device)                     
                ground_truth = batch[1].to(self.device)                
                scores = self.model.evaluate(user_ids)                  
                
                for k in k_values:
                    _, topk_items = torch.topk(scores, k, dim=1)        
                    hits = (ground_truth.gather(1, topk_items) > 0).any(dim=1).float()
                    recall_metrics[k].update(hits)
                    
        metrics = {}
        for k in k_values:
            metrics[f"Recall@{k}"] = recall_metrics[k].compute()
        return metrics
    
    def train_epoch(self):
        """
        Run one epoch of training on the training dataset and update the model parameters.

        Returns:
            float: The average loss computed for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=Fore.YELLOW + "Training", ncols=80)
        for batch in progress:
            self.optimizer.zero_grad()
            user_ids = batch[0].squeeze().to(self.device).view(-1) 
            pos_neg = batch[1].to(self.device)  
            pos_item_ids = pos_neg[:, 0].view(-1, 1)
            neg_item_ids = pos_neg[:, 1].view(-1, 1)
            neg_values = neg_item_ids.cpu().numpy()
            
            pos_scores, pos_L2 = self.model(user_ids, pos_item_ids)
            neg_scores, neg_L2 = self.model(user_ids, neg_item_ids)
            
            loss = bpr_loss(
                pos_scores,
                neg_scores,
                pos_L2,
                neg_L2,
                self.args.train_batch_size,
                self.args.L2_norm,
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def train_total_epoch(self, total_loader):
        """
        Execute one epoch of retraining on the total dataset and update the model parameters.

        Args:
            total_loader (DataLoader): DataLoader instance for the total dataset.

        Returns:
            float: The average loss computed for the retraining epoch.
        """
        self.model.train()
        total_loss = 0.0
        progress = tqdm(total_loader, desc=Fore.YELLOW + "Retraining on Total", ncols=80)
        for batch in progress:
            self.optimizer.zero_grad()
            user_ids = batch[0].squeeze().to(self.device).view(-1)  
            pos_neg = batch[1].to(self.device)  
            pos_item_ids = pos_neg[:, 0].view(-1, 1)
            neg_item_ids = pos_neg[:, 1].view(-1, 1)
            
            pos_scores, pos_L2 = self.model(user_ids, pos_item_ids)
            neg_scores, neg_L2 = self.model(user_ids, neg_item_ids)
            
            loss = bpr_loss(
                pos_scores,
                neg_scores,
                pos_L2,
                neg_L2,
                self.args.train_batch_size,
                self.args.L2_norm,
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(total_loader)
        return avg_loss
    
    def train(self):
        """
        Conduct the full training process including early stopping, retraining, and final evaluation.

        Returns:
            float: The final test Recall@40 metric after training.
        """
        best_recall = -float('inf')
        patience_count = 0
        best_state = None
        best_epoch = 0  

        print(Fore.CYAN + Style.BRIGHT + ">>> Start training ...")
        for epoch in range(1, self.args.epoch + 1):
            start_time = time.time()
            train_loss = self.train_epoch() 
            valid_metrics = self._evaluate_loader(self.valid_loader)
            epoch_time = time.time() - start_time
            log = (
                f"{Fore.WHITE}{Style.BRIGHT}Epoch {epoch:03d}/{self.args.epoch:03d} | "
                f"{Fore.YELLOW}Train Loss: {train_loss:.6f}\n\n"
                + " | ".join(
                    [
                        f"{Fore.BLUE}Valid Recall@{k}: {valid_metrics[f'Recall@{k}']:.4f}"
                        for k in [20, 40, 80]
                    ]
                )
                + f" | {Fore.MAGENTA}Time: {epoch_time:.2f}s"
            )
            print(log)
            if valid_metrics["Recall@40"] > best_recall:
                best_recall = valid_metrics["Recall@40"]
                best_state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.args.patience:
                    print(Fore.RED + f">>> Early stopping at epoch {epoch}, best Validation Recall@40: {best_recall:.4f}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state, strict=False)
        
        final_test_metrics = self._evaluate_loader(self.test_loader)
        print(Fore.GREEN + Style.BRIGHT + ">>> Final Test Metrics:")
        for k in sorted(final_test_metrics.keys()):
            print(f"{k}: {final_test_metrics[k]:.4f}")
        
        print(
            Fore.CYAN
            + Style.BRIGHT
            + f">>> Retraining on total dataset for {best_epoch} epochs ..."
        )
        total_train_dataset = TotalTrainDataset(
            data_path=self.args.data_path,
            dataset_name=self.args.dataset_name,
            relations=self.args.relations,
            neg_sample_size=2,
            debug_sample=False
        )
        total_train_loader = DataLoader(
            total_train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=4
        )
        for epoch in range(1, best_epoch + 1):
            start_time = time.time()
            train_loss = self.train_total_epoch(total_train_loader)
            epoch_time = time.time() - start_time
            log = (
                f"{Fore.WHITE}{Style.BRIGHT}[Retrain] Epoch {epoch:03d}/{best_epoch:03d} | "
                f"{Fore.YELLOW}Train Loss: {train_loss:.6f} | {Fore.MAGENTA}Time: {epoch_time:.2f}s"
            )
            print(log)
       
        save_path = self.args.save_path
        os.makedirs(save_path, exist_ok=True)
        total_save_dir = os.path.join(save_path, "total_model.pth")
        torch.save(self.model.state_dict(), total_save_dir)
        print(
            Fore.GREEN + Style.BRIGHT + f">>> Retrained model saved at {total_save_dir}"
        )
        return final_test_metrics["Recall@40"]
