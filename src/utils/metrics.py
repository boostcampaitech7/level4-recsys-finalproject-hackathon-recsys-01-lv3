# src/utils/metrics.py

import torch
import numpy as np

class Recall:
    """
    Recall@K Metric
    """
    def __init__(self, topk):
        self.topk = topk
        self.reset()
        
    def reset(self):
        self.recall_sum = 0.0
        self.count = 0
    
    def __call__(self, scores, ground_truth):
        """
        Args:
            - scores: Tensor of shape (batch_size, num_items)
            - ground_truth: Tensor of shape (batch_size, num_items), one-hot encoding
        """
        with torch.no_grad():
            # Get top K indices
            _, topk_indices = torch.topk(scores, self.topk, dim=1) # shape: (batch_size, topk)
            
            # Expand ground_truth to match topk_indices shape
            ground_truth = ground_truth.bool()
            
            # Gather the ground truth for the top K indices
            topk_ground_truth = ground_truth.gather(1, topk_indices) # shape: (batch_size, topk)
            
            # Compute number of hits per user
            hits = topk_ground_truth.sum(dim=1).float() # shape: (batch_size,)
            
            # Compute number of ground_truth positives per user
            num_pos = ground_truth.sum(dim=1).float() # shape: (batch_size,)
            
            # Avoid division by zero
            num_pos = torch.clamp(num_pos, min=1.0)
            
            # Calculate Recall@K per user
            recall = hits/num_pos # shape: (batch_size,)
            
            # Sum recalls
            self.recall_sum += recall.sum().item()
            self.count += scores.size(0)
    
    def compute(self):
        if self.count == 0:
            return 0.0
        return self.recall_sum / self.count
    
class Precision:
    """
    Precision@K Metric
    """

    def __init__(self, topk):
        self.topk = topk
        self.reset()

    def reset(self):
        self.precision_sum = 0.0
        self.count = 0

    def __call__(self, scores, ground_truth):
        """
        Args:
            scores: Tensor of shape (batch_size, num_items)
            ground_truth: Tensor of shape (batch_size, num_items), one-hot encoding
        """
        with torch.no_grad():
            # Get top K indices
            _, topk_indices = torch.topk(scores, self.topk, dim=1)  # shape: (batch_size, topk)

            # Expand ground_truth to match topk_indices shape
            ground_truth = ground_truth.bool()

            # Gather the ground truth for the top K indices
            topk_ground_truth = ground_truth.gather(1, topk_indices)  # shape: (batch_size, topk)

            # Compute number of hits per user
            hits = topk_ground_truth.sum(dim=1).float()  # shape: (batch_size,)

            # Calculate Precision@K per user
            precision = hits / self.topk  # shape: (batch_size,)

            # Sum precisions
            self.precision_sum += precision.sum().item()
            self.count += scores.size(0)

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.precision_sum / self.count
    
class NDCG:
    """
    NDCG@K Metric
    """
    def __init__(self, topk):
        self.topk = topk
        self.reset()
        # Precompute the logarithm denominator
        self.log2_positions = torch.log2(torch.arange(2, self.topk + 2, dtype=torch.float))
    
    def reset(self):
        self.ndcg_sum = 0.0
        self.count = 0
        
    def __call__(self, scores, ground_truth):
        """
        Args:
            scores: Tensor of shape (batch_size, num_items)
            ground_truth: Tensor of shape (batch_size, num_items), one-hot encoding
        """
        with torch.no_grad():
            # Get top K indices
            _, topk_indices = torch.topk(scores, self.topk, dim=1)  # shape: (batch_size, topk)

            # Expand ground_truth to match topk_indices shape
            ground_truth = ground_truth.bool()

            # Gather the ground truth for the top K indices
            topk_ground_truth = ground_truth.gather(1, topk_indices)  # shape: (batch_size, topk)

            # Compute DCG@K
            gains = topk_ground_truth.float()
            discounts = self.log2_positions.to(scores.device)  # shape: (topk,)
            dcg = (gains / discounts).sum(dim=1)  # shape: (batch_size,)

            # Compute IDCG@K
            # For each user, the maximum possible DCG is when all ground truths are in the top K
            num_pos = ground_truth.sum(dim=1).clamp(max=self.topk).float()  # shape: (batch_size,)
            # Compute ideal gains
            ideal_gains = torch.ones_like(gains)
            ideal_gains = torch.cumsum(ideal_gains, dim=1)
            ideal_gains = torch.clamp(ideal_gains, max=num_pos.view(-1, 1))
            idcg = (ideal_gains / discounts).sum(dim=1)  # shape: (batch_size,)

            # Compute NDCG@K
            ndcg = dcg / (idcg + 1e-8)

            # Sum NDCG
            self.ndcg_sum += ndcg.sum().item()
            self.count += scores.size(0)

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.ndcg_sum / self.count

class MRR:
    """
    Mean Reciprocal Rank@K Metric
    """

    def __init__(self, topk):
        self.topk = topk
        self.reset()

    def reset(self):
        self.mrr_sum = 0.0
        self.count = 0

    def __call__(self, scores, ground_truth):
        """
        Args:
            scores: Tensor of shape (batch_size, num_items)
            ground_truth: Tensor of shape (batch_size, num_items), one-hot encoding
        """
        with torch.no_grad():
            # Get top K indices
            _, topk_indices = torch.topk(scores, self.topk, dim=1)  # shape: (batch_size, topk)

            # Expand ground_truth to match topk_indices shape
            ground_truth = ground_truth.bool()

            # Gather the ground truth for the top K indices
            topk_ground_truth = ground_truth.gather(1, topk_indices)  # shape: (batch_size, topk)

            # Compute reciprocal ranks
            # For each user, find the first hit in top K
            hits = topk_ground_truth.float()
            # To avoid division by zero, set non-hits to 0
            reciprocal_ranks = hits / (torch.arange(1, self.topk + 1, device=scores.device).float().view(1, -1))
            # For each user, get the first non-zero reciprocal rank
            mrr = reciprocal_ranks.max(dim=1)[0]  # shape: (batch_size,)

            # Sum MRR
            self.mrr_sum += mrr.sum().item()
            self.count += scores.size(0)

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.mrr_sum / self.count