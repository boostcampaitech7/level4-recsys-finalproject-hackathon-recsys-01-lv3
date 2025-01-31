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
    
    def update(self, hits):
        """
        Args:
            hits: Tensor of shape (batch_size,), 1.0 if hit, else 0.0
        """
        self.recall_sum += hits.sum().item()
        self.count += hits.size(0)
    
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

    def update(self, precision):
        """
        Args:
            precision: Tensor of shape (batch_size,), precision values per user
        """
        self.precision_sum += precision.sum().item()
        self.count += precision.size(0)

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
        self.register_log2_discounts()
    
    def register_log2_discounts(self):
        # Precompute log2(positions + 1) for ranks 1 to topk
        self.log2_discounts = torch.log2(torch.arange(2, self.topk + 2, dtype=torch.float)).to('cpu')  # shape=(topk,)
    
    def reset(self):
        self.ndcg_sum = 0.0
        self.count = 0
        
    def update(self, ndcg):
        """
        Args:
            ndcg: Tensor of shape (batch_size,), NDCG values per user
        """
        self.ndcg_sum += ndcg.sum().item()
        self.count += ndcg.size(0)
    

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

    def update(self, reciprocal_ranks):
        """
        Args:
            reciprocal_ranks: Tensor of shape (batch_size,), reciprocal ranks per user
        """
        self.mrr_sum += reciprocal_ranks.sum().item()
        self.count += reciprocal_ranks.size(0)

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.mrr_sum / self.count