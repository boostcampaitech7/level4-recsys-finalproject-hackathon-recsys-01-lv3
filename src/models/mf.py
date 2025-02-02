# src/models/mf.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRMF(nn.Module):
    """
    BPR 기반의 Matrix Factorization 모델.
    - user_emb: (num_users, emb_dim)
    - item_emb: (num_items, emb_dim)

    용도:
     - 학습 후 user_emb, item_emb를 npy 등으로 저장,
       이후 MBGCN 초기값으로 사용 가능.
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_size: int=64,
        device: str="cpu"
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.device = device

        # 파라미터 (user/item 임베딩)
        self.user_emb = nn.Parameter(torch.empty(num_users, embedding_size))
        self.item_emb = nn.Parameter(torch.empty(num_items, embedding_size))

        # Xavier init
        nn.init.xavier_uniform_(self.user_emb)
        nn.init.xavier_uniform_(self.item_emb)

        self.to(device)

    def forward(self, user_ids, pos_item_ids, neg_item_ids=None):
        """
        BPR Forward
        user_ids: (batch,)
        pos_item_ids: (batch,)
        neg_item_ids: (batch,) or (batch,neg_size)
        returns: pos_scores, neg_scores
        """
        user_ids = user_ids.to(self.device)
        pos_item_ids = pos_item_ids.to(self.device)

        u_e = self.user_emb[user_ids]      # (batch, emb_dim)
        p_e = self.item_emb[pos_item_ids]  # (batch, emb_dim)

        pos_scores = torch.sum(u_e*p_e, dim=-1)  # (batch,)

        if neg_item_ids is not None:
            neg_item_ids = neg_item_ids.to(self.device)
            if len(neg_item_ids.shape) == 2:
                # (batch, neg_size)
                neg_e = self.item_emb[neg_item_ids.view(-1)]
                neg_e = neg_e.view(*neg_item_ids.shape, self.embedding_size)  # (batch,neg_size,emb)
                # (batch,1,emb)*(batch,neg_size,emb) => (batch,neg_size)
                neg_scores = torch.sum(u_e.unsqueeze(1)*neg_e, dim=-1)
            else:
                # (batch,)
                neg_e = self.item_emb[neg_item_ids]
                neg_scores = torch.sum(u_e*neg_e, dim=-1)

            return pos_scores, neg_scores
        else:
            return pos_scores, None
        
    def evaluate(self, user_ids):
        user_ids = user_ids.to(self.device)
        user_feature = self.user_emb[user_ids]  # (batch_size, emb_dim)
        scores = torch.matmul(user_feature, self.item_emb.t())  # (batch_size, num_items)
        return scores

    def get_user_emb_matrix(self):
        """
        CPU -> numpy
        """
        return self.user_emb.detach().cpu().numpy()

    def get_item_emb_matrix(self):
        return self.item_emb.detach().cpu().numpy()
