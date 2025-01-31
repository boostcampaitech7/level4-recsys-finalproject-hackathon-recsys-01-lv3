import torch
import torch.nn.functional as F

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, model: torch.nn.Module, beta: float) -> torch.Tensor:
    """
    L2 정규화가 적용된 BPR Loss 계산 함수 
    논문에서는 학습 가능한 모든 파라미터에 대해 L2 정규화를 적용한다고 했지만,
    과적합이 발생할 수 있으므로, 주 학습 대상인 유저 및 아이템 임베딩에만 적용한다.
    
    Args:
        - pos_scores (Tensor): 긍정 샘플의 예측 점수, shape=(batch_size,)
        - neg_scores (Tensor): 부정 샘플의 예측 점수, shape=(batch_size,)
        - model (nn.Module): 학습 중인 모델
        - beta (float): L2 정규화 계수
    
    Returns:
        - loss (Tensor): 스칼라 손실 값값
    """
    # 크기 검증
    assert pos_scores.shape == neg_scores.shape, f"pos_scores shape {pos_scores.shape} does not match neg_scores shape {neg_scores.shape}"
    
    # BPR Loss 계산
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    
    # 사용자 및 아이템 임베딩에 대한 L2 정규화
    l2_reg = torch.norm(model.user_emb, p=2) + torch.norm(model.item_emb, p=2)
    
    # 최종 손실: BPR Loss + beta * L2 정규화
    loss += beta * l2_reg
    return loss