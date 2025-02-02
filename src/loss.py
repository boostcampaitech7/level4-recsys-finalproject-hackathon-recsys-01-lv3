import torch
import torch.nn.functional as F

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
             pos_L2: torch.Tensor, neg_L2: torch.Tensor,
             batch_size: int, beta: float) -> torch.Tensor:
    """
    L2 정규화가 적용된 BPR Loss 계산 함수.
    
    Args:
        pos_scores (Tensor): 긍정 샘플의 예측 점수, shape=(batch_size, 1)
        neg_scores (Tensor): 부정 샘플의 예측 점수, shape=(batch_size, 1)
        pos_L2 (Tensor): 긍정 샘플에 대한 L2 정규화 항 (스칼라 혹은 배치별 값)
        neg_L2 (Tensor): 부정 샘플에 대한 L2 정규화 항
        beta (float): L2 정규화 계수
    
    Returns:
        loss (Tensor): 최종 BPR Loss (스칼라)
    """
    # BPR Loss 계산: 긍정-음수 차이에 대해 로그 시그모이드 취함
    loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    # 두 forward 호출에서 나온 L2_loss를 평균 내어 정규화 항에 반영
    L2_loss = (pos_L2 + neg_L2) / 2.0
    # loss += 0.5 * beta * L2_loss
    loss += L2_loss / batch_size
    return loss