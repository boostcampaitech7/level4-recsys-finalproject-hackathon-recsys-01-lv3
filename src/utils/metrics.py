# src/utils/metrics.py

import torch
import numpy as np

class Recall:
    """
    Recall@K Metric.

    Args:
        topk (int): The value of K for Recall@K.
    """

    def __init__(self, topk):
        self.topk = topk
        self.reset()
        
    def reset(self):
        """
        Reset the accumulated recall sum and count.
        """
        self.recall_sum = 0.0
        self.count = 0
    
    def update(self, hits):
        """
        Update the recall metric with a new batch of hit indicators.
        
        Args:
            hits: Tensor of shape (batch_size,), 1.0 if hit, else 0.0
        """
        self.recall_sum += hits.sum().item()
        self.count += hits.size(0)
    
    def compute(self):
        """
        Compute the final Recall metric.

        Returns:
            float: The computed Recall value.
        """
        if self.count == 0:
            return 0.0
        return self.recall_sum / self.count
    
class Precision:
    """
    Precision@K Metric
    
    Args:
        topk (int): The value of K for Precision@K.
    """

    def __init__(self, topk):
        self.topk = topk
        self.reset()

    def reset(self):
        """
        Reset the accumulated precision sum and count.
        """
        self.precision_sum = 0.0
        self.count = 0

    def update(self, precision):
        """
        Update the precision metric with new precision values.
        
        Args:
            precision: Tensor of shape (batch_size,), precision values per user
        """
        self.precision_sum += precision.sum().item()
        self.count += precision.size(0)

    def compute(self):
        """
        Compute the final Precision metric.
        """
        if self.count == 0:
            return 0.0
        return self.precision_sum / self.count
    
class NDCG:
    """
    NDCG@K Metric
    
    Args:
        topk (int): The value of K for NDCG@K.
    """
    def __init__(self, topk):
        self.topk = topk
        self.reset()
        self.register_log2_discounts()
    
    def register_log2_discounts(self):
        """
        Precompute logarithm denominator values for ranks 1 to topk.
        """
        self.log2_discounts = torch.log2(
            torch.arange(2, self.topk + 2, dtype=torch.float)
        ).to('cpu')  
    
    def reset(self):
        """
        Reset the accumulated NDCG sum and count.
        """
        
        self.ndcg_sum = 0.0
        self.count = 0
        
    def update(self, ndcg):
        """
        Update the NDCG metric with new NDCG values.
        
        Args:
            ndcg: Tensor of shape (batch_size,), NDCG values per user
        """
        self.ndcg_sum += ndcg.sum().item()
        self.count += ndcg.size(0)
    

    def compute(self):
        """
        Compute the final NDCG metric.
        
        Returns:
            float: The computed NDCG value.
        """
        if self.count == 0:
            return 0.0
        return self.ndcg_sum / self.count

class MRR:
    """
    Mean Reciprocal Rank@K Metric
    
    Args:
        topk (int): The value of K for MRR@K.
    """

    def __init__(self, topk):
        self.topk = topk
        self.reset()

    def reset(self):
        """
        Reset the accumulated MRR sum and count.
        """
        self.mrr_sum = 0.0
        self.count = 0

    def update(self, reciprocal_ranks):
        """
        Update the MRR metric with new reciprocal rank values.
        
        Args:
            reciprocal_ranks: Tensor of shape (batch_size,), reciprocal ranks per user
        """
        self.mrr_sum += reciprocal_ranks.sum().item()
        self.count += reciprocal_ranks.size(0)

    def compute(self):
        """
        Compute the final Mean Reciprocal Rank (MRR) metric.
        """
        if self.count == 0:
            return 0.0
        return self.mrr_sum / self.count