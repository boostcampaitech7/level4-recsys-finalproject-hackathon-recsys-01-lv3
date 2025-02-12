import torch
import torch.nn.functional as F

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
             pos_L2: torch.Tensor, neg_L2: torch.Tensor,
             batch_size: int, beta: float) -> torch.Tensor:
    """
    Calculate the BPR loss with L2 regularization.

    Args:
        pos_scores (torch.Tensor): Predicted scores for positive samples, shape=(batch_size, 1).
        neg_scores (torch.Tensor): Predicted scores for negative samples, shape=(batch_size, 1).
        pos_L2 (torch.Tensor): L2 regularization term for positive samples (scalar or per-batch value).
        neg_L2 (torch.Tensor): L2 regularization term for negative samples.
        batch_size (int): The batch size used during training.
        beta (float): Coefficient for L2 regularization.

    Returns:
        torch.Tensor: The final BPR loss (scalar).
    """
    loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    L2_loss = (pos_L2 + neg_L2) / 2.0
    loss += beta * L2_loss
    return loss