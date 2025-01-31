import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, List
from collections import defaultdict

class MBGCN(nn.Module):
    """
    MB-GCN model for multi-behavior recommendation, referencing Jin et al. (SIGIR 2020).
    
    We do multi-layer propagation for:
        - user-item (behavior-aware)
        - item-item (behavior semantics)
    Then combine user-based CF score + item-based CF score to get final score.
    """
    """
    01-31 변경 사항:
    1) 메시지 패싱(encode)은 한 Epoch에 한 번만 호출되도록: self.embedding_cached = 0 -> encode() -> 1
    2) forward 시 이미 캐싱된 self.user_latent, self.item_latent, self.s_item_list를 사용
        => 더 이상 대규모 spmm 연산이 반복되지 않도록록
    """
    def __init__(
        self, 
        num_users: int,
        num_items: int,
        embedding_size: int=64,
        num_layers: int=2,
        behaviors: List[str]=None, # e.g. ["view", "cart", "purchase"]
        ui_mats: Dict[str, sp.csr_matrix]=None,
        ii_mats: Dict[str, sp.csr_matrix]=None,
        user_bhv_dict: Dict[str, Dict[int, List[int]]]=None,  # 필수 인자로 변경 가능
        node_dropout: float=0.2,
        message_dropout: float=0.2,
        alpha_learning: bool=True,
        lamb: float=0.5,
        item_cf_mode: str="original", # "original" or "unify"
        item_alpha: bool=False,  # default: no alpha at item update
        alpha_mode: str="global",     # "global" or "per_user"
        device: str="cpu"
    ):
        """
        Args:
            - num_users, num_items (int): total counts
            - embedding_size (int): dimension of embeddings
            - num_layers (int): GCN propagation depth
            - behaviors (List[str]): list of behavior names (keys in ui_mats, ii_mats)
            - ui_mats (Dict[str, sp.csr_matrix]): user-item adjacency (dict of CSR for each behavior)
            - ii_mats (Dict[str, sp.csr_matrix]): item-item adjacency (dict of CSR for each behavior)
            - user_bhv_dict (Dict[str, Dict[int, List[int]]]): 사용자별 행동별 아이템 리스트
            - node_dropout (float): ratio for node dropout
            - message_dropout (float): ratio for message dropout
            - alpha_learning (bool): whether to learn a global weight for each behavior
            - lamb (float): weighting factor for user-based CF vs item-based CF score
            - item_cf_mode (str):
                - "original" => distinct transform & M_t per behavior
                - "unify"    => single transform & single M for all behaviors
            - item_alpha (bool):
                - True  => use alpha at item embedding update
                - False => no alpha at item embedding update
            - alpha_mode (str):
                - "global"   => alpha[t] is global for all users
                - "per_user" => alpha[u,t] = w[t]*n_{u,t}/sum_m w[m]*n_{u,m}
            - device (str): "cpu" or "cuda"
        """
        super().__init__()
        if behaviors is None:
            behaviors = ["view", "cart", "purchase"]
            
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.behaviors = behaviors
        self.ui_mats = ui_mats if ui_mats else {}
        self.ii_mats = ii_mats if ii_mats else {}
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.alpha_learning = alpha_learning
        self.lamb = lamb
        self.item_cf_mode = item_cf_mode
        self.item_alpha = item_alpha
        self.alpha_mode = alpha_mode
        self.device = device
        
        # 1) LayerNorm을 각 레이어별로 정의
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_layers)])
        
        # 2) user & item base embeddings => p_u^(0), q_i^(0)
        self.user_emb = nn.Parameter(torch.empty(num_users, embedding_size))
        self.item_emb = nn.Parameter(torch.empty(num_items, embedding_size))
        nn.init.xavier_uniform_(self.user_emb, gain=1.0)
        nn.init.xavier_uniform_(self.item_emb, gain=1.0)
        
        # 3) s_{i,t} for item-based CF => behavior-specific item embeddings
        self.s_item_emb = nn.ParameterList()
        for _ in behaviors:
            p = nn.Parameter(torch.empty(num_items, embedding_size))
            nn.init.xavier_uniform_(p, gain=1.0)
            self.s_item_emb.append(p)
        
        # 4) alpha param (for user-item weighting)
        if self.alpha_learning:
            self.behavior_alpha = nn.Parameter(torch.ones(len(behaviors), dtype=torch.float32))
        else:
            self.register_buffer("behavior_alpha", torch.ones(len(behaviors), dtype=torch.float32))
        # user_count => n_{u,t} if alpha_mode="per_user"
        self.register_buffer("user_count", torch.zeros(len(behaviors), num_users, dtype=torch.float32))
        
        # 5) M_t or unify_M => item-based CF transformations
        if item_cf_mode == "original":
            # M_t: per behavior
            self.M_t = nn.Parameter(torch.empty(len(behaviors), embedding_size, embedding_size))
            nn.init.xavier_uniform_(self.M_t, gain=1.0)
        elif item_cf_mode == "unify":
            # single M
            self.unify_M = nn.Parameter(torch.empty(embedding_size, embedding_size))
            nn.init.xavier_uniform_(self.unify_M, gain=1.0)
        else:
            raise ValueError("item_cf_mode must be 'original' or 'unify'")
        
        # 6) message_dropout modules
        self.msg_dropout = nn.Dropout(p=self.message_dropout)

        # 7) user-item transforms
        self.ui_transforms = nn.ModuleList([
            nn.Linear(embedding_size, embedding_size, bias=False)
            for _ in range(num_layers)
        ])
        
        # 8) item-item transforms for behavior-based s_{i,t}
        self.ii_transforms_behavior = nn.ModuleList()
        for lidx in range(num_layers):
            layer_list = nn.ModuleList()
            for _ in behaviors:
                Wt = nn.Linear(embedding_size, embedding_size, bias=False)
                nn.init.xavier_uniform_(Wt.weight, gain=1.0)
                layer_list.append(Wt)
            self.ii_transforms_behavior.append(layer_list)
        
        # 9) 사용자별 행동 데이터 딕셔너리 설정
        if user_bhv_dict is not None:
            self.user_bhv_dict = user_bhv_dict
        else:
            self.user_bhv_dict = defaultdict(lambda: defaultdict(list))
        
        # Initialize encoding cache
        self.register_buffer("user_latent", torch.zeros(num_users, embedding_size, device=self.device))
        self.register_buffer("item_latent", torch.zeros(num_items, embedding_size, device=self.device))
        self.register_buffer("s_item_list", torch.zeros(len(behaviors), num_items, embedding_size, device=self.device))
        self.register_buffer("embedding_cached", torch.tensor(0, device=self.device))  # 0: not cached, 1: cached
        self.register_buffer("user_mean_emb", torch.zeros(len(behaviors), num_users, embedding_size, device=self.device))
        # 모든 파라미터와 모듈을 지정한 디바이스로 이동
        self.to(device)
        
    def calc_user_count(self):
        """
        Precompute n_{u,t} = # of items that user u has interacted with under behavior t
        We'll store it in self.user_count[t_idx,u].
        Must be called after we have self.ui_mats ready.
        """
        # user_count shape: (len(behaviors), num_users)
        user_count_temp = torch.zeros(len(self.behaviors), self.num_users, dtype=torch.float32, device=self.device)
        for t_idx, bname in enumerate(self.behaviors):
            row_sum = np.array(self.ui_mats[bname].sum(axis=1)).flatten()
            user_count_temp[t_idx] = torch.from_numpy(row_sum).float()
    
        self.user_count = user_count_temp  # 이미 device에 있음
        # print(f"[INFO] user_count Device: {self.user_count.device}")  # 🚀 디버깅용 출력
    
    def forward(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor):
        """
        For BPR usage: forward(user, item) => predicted score.
        We'll do multi-layer propagation for user_emb, item_emb,
        then compute user-based CF score + item-based CF score,
        combine by lamb.
        
            1) multi-layer propagation
            2) user-based CF score = dot(u_emb, i_emb)
            3) item-based CF score => depends on item_cf_mode
                - "original"
                - "unify"
            4) combine with lamb: final = lamb * user_cf + (1-lamb) * item_cf

        Returns: shape=(batch_size,)
        """
        """
        0131 변경사항:
        - 이미 encode()로부터 계산된 user_latent, item_latent, s_item_list를 사용
        """
        
        device = self.device  # Use model's device
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        
        # encode()가 되어 있지 않으면 에러
        if self.embedding_cached.item() == 0:
            raise RuntimeError("Please call model.encode() before forward! (Epoch-level encode)")

        # user-based CF
        u_vec = self.user_latent[user_idx] # (batch_size, emb_dim)
        i_vec = self.item_latent[item_idx] # (batch_size, emb_dim)
        user_cf_score = torch.sum(u_vec * i_vec, dim=1) # (batch_size,)
        
        # item-based CF
        if self.item_cf_mode == "original":
            item_cf_score = self.item_based_cf_score_original(user_idx, item_idx)
        else:
            item_cf_score = self.item_based_cf_score_unify(user_idx, item_idx)
        item_cf_score = item_cf_score.to(device)
            
        # combine
        score = self.lamb * user_cf_score + (1.0 - self.lamb) * item_cf_score
        
        # 디버그: 출력 크기 확인
        # print(f"  forward:")
        # print(f"    user_idx shape: {user_idx.shape}")               # (batch_size,)
        # print(f"    item_idx shape: {item_idx.shape}")               # (batch_size,)
        # print(f"    user_cf_score shape: {user_cf_score.shape}")     # (batch_size,)
        # print(f"    item_cf_score shape: {item_cf_score.shape}")     # (batch_size,)
        # print(f"    score shape: {score.shape}")                     # (batch_size,)
        
        return score
    
    def encode(self):
        """
        1) 전체 사용자/아이템 임베딩을 GCN propagate
        2) item-item GCN도 propagate
        3) self.user_latent, self.item_latent, self.s_item_list에 결과를 캐싱
        => Epoch에 1회 호출
        """
        self.eval()
        with torch.no_grad():
            user_latent, item_latent, s_item_list = self.propagate_embeddings()
        # 실제 버퍼에 copy_
        self.user_latent.copy_(user_latent)
        self.item_latent.copy_(item_latent)
        self.s_item_list.copy_(torch.stack(s_item_list))  # shape=(len(behaviors), num_items, emb_dim)
        self.embedding_cached.fill_(1)  # Mark as cached
        
        # === 0131 추가: 사용자별 아이템 평균 임베딩 precompute ===
        self.precompute_user_mean_emb()  
        
        self.train()
    
    def precompute_user_mean_emb(self):
        """
        self.s_item_list[b_idx] (shape: (num_items, emb_dim))가 이미 item-item GCN까지 반영된 상태.
        => user_bhv_dict[behavior][u]로 유저가 본 item indices.
        => 평균내어 user_mean_emb[b_idx, u] 저장.
        """
        with torch.no_grad():
            for b_idx, bname in enumerate(self.behaviors):
                # s_t: (num_items, emb_dim)
                s_t = self.s_item_list[b_idx]  
                # buffer to fill
                # shape: (num_users, emb_dim)
                user_mean_mat = torch.zeros(self.num_users, self.embedding_size, device=self.device)
                
                # Python 루프 (num_users 번) => 이후 forward 시점에는 안 돌음
                for u in range(self.num_users):
                    items_u = self.user_bhv_dict[bname][u]
                    if len(items_u) > 0:
                        emb_u = s_t[items_u]  # (n_u, emb_dim)
                        user_mean_mat[u] = emb_u.mean(dim=0)
                
                self.user_mean_emb[b_idx].copy_(user_mean_mat)
    
    ### item-based CF (original vs unify) ###
    
    def item_based_cf_score_original(
        self, 
        user_idx: torch.LongTensor, 
        item_idx: torch.LongTensor, 
        # s_item_embs: List[torch.Tensor]
    ):
        """
        eq. (10) from MB-GCN paper:
            y2(u,i) = sum_{t} sum_{j in N_t^I(u)} [ s_{j,t}^{*T} M_t s_{i,t}^{*}] / |N_t^I(u)|
            => Possibly multiply alpha[t] if self.item_alpha=True
        
         Args:
            - user_idx: Tensor of user indices, shape=(batch_size,)
            - item_idx: Tensor of item indices, shape=(batch_size,)
            - s_item_embs: List of tensors, each shape=(num_items, emb_dim)
        
        Returns:
            - scores: Tensor of shape=(batch_size,)
        """
        """
        0131 변경사항:
        - 이미 s_item_list[t_i], self.M_t[t_i] 가 self.encode()에서 계산됨
        """
        device = self.device
        batch_size = user_idx.size(0)
        alpha_vec = self.get_alpha() # shape=(len(behaviors),) or (len(behaviors), num_users) depending on alpha_mode
                
        sum_over_t = torch.zeros(batch_size, device=device)
        for t_i, bname in enumerate(self.behaviors):
            # s_i_t = self.s_item_list[t_i][item_idx]  # shape=(batch_size, emb_dim)
            # user_items = [self.user_bhv_dict[bname][u.item()] for u in user_idx]

            # # 디버그: 사용자별 아이템 리스트 확인
            # # print(f"    Behavior '{bname}':")
            # # for u in range(batch_size):
            #     # print(f"      User {user_idx[u].item()} items: {user_items[u]}")
    
            
            # # 빈 리스트를 안전하게 처리 (batch_size 유지)
            # user_items_emb = [
            #     self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 
            #     else torch.zeros(self.embedding_size, device=device) 
            #     for items in user_items
            # ]
            # user_items_emb = torch.stack(user_items_emb)  # shape=(batch_size, emb_dim)
            # # print(f"      user_items_emb shape: {user_items_emb.shape}")  # (batch_size, emb_dim)

            # transformed = F.relu(torch.matmul(user_items_emb, self.M_t[t_i]))  # 활성화 함수 추가
            # # print(f"      transformed shape: {transformed.shape}")    # (batch_size, emb_dim)
            # score_t = torch.sum(transformed * s_i_t, dim=1)  # shape=(batch_size,)
            # # print(f"      score_t shape: {score_t.shape}")          # (batch_size,)
            # user_mean_emb[t_i][user_idx] => (batch_size, emb_dim)
            user_emb_t = self.user_mean_emb[t_i][user_idx]
            # item_emb => s_item_list[t_i][item_idx] => (batch_size, emb_dim)
            i_emb_t = self.s_item_list[t_i][item_idx]
            
            # transform
            transformed = F.relu(torch.matmul(user_emb_t, self.M_t[t_i]))  # (batch_size, emb_dim)
            
            # dot product with i_emb_t
            score_t = torch.sum(transformed * i_emb_t, dim=1)  # (batch_size,)
            
            if self.item_alpha:
                if self.alpha_mode == "global":
                    sum_over_t = sum_over_t + alpha_vec[t_i] * score_t
                elif self.alpha_mode == "per_user":
                    alpha_per_user = alpha_vec[t_i][user_idx]  # shape=(batch_size,)
                    sum_over_t = sum_over_t + alpha_vec[t_i] * score_t
                else:
                    raise ValueError("alpha_mode must be 'global' or 'per_user'")
            else:
                sum_over_t = sum_over_t + score_t
            # print(f"      sum_over_t shape after update: {sum_over_t.shape}")  # (batch_size,)
        return sum_over_t
    
    
    def item_based_cf_score_unify(
        self, 
        user_idx: torch.LongTensor,
        item_idx: torch.LongTensor,
        # s_item_embs: List[torch.Tensor]
    ):
        """
        A simpler version of eq.(10), but with a SINGLE M for all behaviors:
        y2(u,i) = sum_{t} sum_{j in N_t^I(u)} [ s_{j,t}^T M s_{i,t} ] / |N_t^I(u)| * alpha[t]
        => same M for every t.
        """
        device = self.device
        batch_size = user_idx.size(0)
        alpha_vec = self.get_alpha()
        
        # single unify_M
        if not hasattr(self, "unify_M"):
            raise NotImplementedError("unify_M not found, item_cf_mode=unify?")
        
        sum_over_t = torch.zeros(batch_size, device=device)
        for t_i, bname in enumerate(self.behaviors):
            # s_i_t = self.s_item_list[t_i][item_idx]  # shape=(batch_size, emb_dim)
            # user_items = [self.user_bhv_dict[bname][u.item()] for u in user_idx]

            # # 빈 리스트를 안전하게 처리 (batch_size 유지)
            # user_items_emb = [
            #     self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 
            #     else torch.zeros(self.embedding_size, device=device) 
            #     for items in user_items
            # ]
            # user_items_emb = torch.stack(user_items_emb)  # shape=(batch_size, emb_dim)

            # transformed = F.relu(torch.matmul(user_items_emb, self.unify_M))
            # score_t = torch.sum(transformed * s_i_t, dim=1)  # shape=(batch_size,)
            user_emb_t = self.user_mean_emb[t_i][user_idx]
            i_emb_t = self.s_item_list[t_i][item_idx]
            transformed = F.relu(torch.matmul(user_emb_t, self.unify_M))
            score_t = torch.sum(transformed * i_emb_t, dim=1)

            if self.item_alpha:
                if self.alpha_mode == "global":
                    sum_over_t = sum_over_t + alpha_vec[t_i] * score_t
                elif self.alpha_mode == "per_user":
                    alpha_per_user = alpha_vec[t_i][user_idx]  # shape=(batch_size,)
                    sum_over_t = sum_over_t + alpha_per_user * score_t
                else:
                    raise ValueError("alpha_mode must be 'global' or 'per_user'")
            else:
                sum_over_t = sum_over_t + score_t

        return sum_over_t
    
    ### multi-layer propagation ###
    
    def propagate_embeddings(self):
        """
        (no_grad)에서 호출됨
        1) user-item GCN
        2) item-item GCN
        => 최종 user_latent, item_latent, s_item_list
        """
        user_latent = self.user_emb
        item_latent = self.item_emb
        s_item_list = [p for p in self.s_item_emb] # behavior-specific item emb
        
        for layer_idx in range(self.num_layers):
            # Debugging: print devices
            # print(f"propagate_embedding: {user_latent.device}, {item_latent.device}, {layer_idx}")
            user_latent, item_latent = self.propagate_user_item(user_latent, item_latent, layer_idx)
            
            # item-item for each behavior emb: s_{i,t} => always distinct
            s_item_list_new = []
            if self.ii_mats is not None:
                for b_idx, s_emb in enumerate(s_item_list):
                    s_new = self.propagate_item_item_behavior(s_emb, b_idx, layer_idx)
                    s_item_list_new.append(s_new)
            else:
                s_item_list_new.append(s_emb)
            s_item_list = s_item_list_new

            # 디버깅: 각 레이어별 임베딩의 평균과 표준편차 출력
            # print(f"Layer {layer_idx+1}/{self.num_layers}:")
            # print(f"  User Embeddings - Mean: {user_latent.mean().item():.4f}, Std: {user_latent.std().item():.4f}")
            # print(f"  Item Embeddings - Mean: {item_latent.mean().item():.4f}, Std: {item_latent.std().item():.4f}")
            # for b_idx, s_new in enumerate(s_item_list):
            #     print(f"  s_item_emb[{b_idx}] - Mean: {s_new.mean().item():.4f}, Std: {s_new.std().item():.4f}")
        
        return user_latent, item_latent, s_item_list
    
    def propagate_user_item(self, user_emb, item_emb, layer_idx: int):
        """
        behavior-aware user-item propagation for layer 'layer_idx'
        """
        device = user_emb.device
        # print(f"propagate_user_item: {device}")
        # alpha_vec을 함수 내에서 한 번만 가져오도록 변경 (최적화)
        alpha_vec = self.get_alpha()   # shape=(len(behaviors),) or (len(behaviors), num_users)
        
        # (1) user update
        user_agg = torch.zeros_like(user_emb, device=device)
        
        if self.alpha_mode == "global":
            for t_i, bname in enumerate(self.behaviors):
                mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
                user_part = self.spmm(mat, item_emb)  # PyTorch sparse matrix multiplication
                user_agg = user_agg + alpha_vec[t_i].item() * user_part  # Weighted sum
                
        else:  # alpha_mode == 'per_user'
            user_parts = []
            for t_i, bname in enumerate(self.behaviors):
                mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
                user_part = self.spmm(mat, item_emb)  # shape=(num_users, emb_dim)
                user_parts.append(user_part)
                
            user_parts = torch.stack(user_parts, dim=0)  # shape=(len(behaviors), num_users, emb_dim)
            
            # Compute per-user alpha weight
            # print(f"alpha_vec.shape = {alpha_vec.shape}\nalpha_vec.view(-1, 1).shape={alpha_vec.view(-1, 1).shape}")
            # print(f"self.user_count={self.user_count}")
            # print(f"self.user_count.shape={self.user_count.shape}")
            # denom = torch.clamp(torch.sum(alpha_vec.view(-1, 1) * self.user_count, dim=0, keepdim=True), min=1e-9)
            # alpha_ut = (alpha_vec.view(-1, 1) * self.user_count) / denom  # shape=(len(behaviors), num_users)
            denom = torch.clamp(torch.sum(alpha_vec * self.user_count, dim=0, keepdim=True), min=1e-9)
            alpha_ut = (alpha_vec * self.user_count) / denom  # shape=(len(behaviors), num_users)
            
            # Weighted sum of user embeddings
            user_agg = torch.sum(alpha_ut.unsqueeze(2) * user_parts, dim=0)  # shape=(num_users, emb_dim)
        
        # Transform and apply dropout
        transform_ui = self.ui_transforms[layer_idx]
        user_new = transform_ui(user_agg)
        user_new = F.relu(user_new)  # 활성화 함수 추가
        user_new = self.layer_norms[layer_idx](user_new)  # 정규화 추가
        user_new = self.msg_dropout(user_new)
        
        # 임베딩 값 모니터링
        # print(f"user_new[{layer_idx}] - mean: {user_new.mean().item()}, std: {user_new.std().item()}")
        
        # (2) item update
        item_agg = torch.zeros_like(item_emb, device=device)
        
        if not self.item_alpha:  # item_alpha=False: simple sum of behaviors
            for bname in self.behaviors:
                mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
                item_part = self.spmm(mat.T, user_new)  # PyTorch sparse multiplication
                item_agg = item_agg + item_part

        else:  # item_alpha=True: per-behavior weighted sum
            if self.alpha_mode == "global":
                for t_i, bname in enumerate(self.behaviors):
                    mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
                    item_part = self.spmm(mat.T, user_new)  # shape=(num_items, emb_dim)
                    item_agg = item_agg + alpha_vec[t_i] * item_part
            else:  # alpha_mode="per_user"
                item_parts = []
                for t_i, bname in enumerate(self.behaviors):
                    mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
                    item_part = self.spmm(mat.T, user_new)  # shape=(num_items, emb_dim)
                    item_parts.append(item_part)
                    
                item_parts = torch.stack(item_parts, dim=0)  # shape=(len(behaviors), num_items, emb_dim)

                # Compute per-behavior weight, not per-user alpha_ut
                item_agg = torch.sum(alpha_vec.view(-1, 1, 1) * item_parts, dim=0)  # shape=(num_items, emb_dim)
        
        # Transform and apply dropout
        transform_ui = self.ui_transforms[layer_idx]
        item_new = transform_ui(item_agg)
        item_new = F.relu(item_new)  # 활성화 함수 추가
        item_new = self.layer_norms[layer_idx](item_new)  # 정규화 추가
        item_new = self.msg_dropout(item_new)
        
        # 임베딩 값 모니터링
        # print(f"item_new[{layer_idx}] - mean: {item_new.mean().item()}, std: {item_new.std().item()}")
        
        return user_new, item_new
    
    def propagate_item_item_behavior(self, s_item_emb, behavior_idx: int, layer_idx: int):
        """
        Propagate item embeddings for a specific behavior and layer.
        
        Args:
            - s_item_emb: Tensor, shape=(num_items, emb_dim)
            - behavior_idx: int, index of the behavior
            - layer_idx: int, current layer index
        
        Returns:
            - s_new: Tensor, shape=(num_items, emb_dim)
        """
        device = s_item_emb.device  # Ensure correct device usage
        # print(f"propagate_item_item_behavior: {device}")
        bname = self.behaviors[behavior_idx]
        mat_ii = self.ii_mats.get(bname, sp.csr_matrix((self.num_items, self.num_items)))
        
        # Apply node dropout
        mat_drop = self.node_dropout_csr(mat_ii, self.node_dropout, device=device)
        
        # Perform sparse-dense matrix multiplication
        item_part = self.spmm(mat_drop, s_item_emb)  # shape=(num_items, emb_dim)
        
        # Apply behavior-specific transform and dropout
        transform_t = self.ii_transforms_behavior[layer_idx][behavior_idx]
        out = transform_t(item_part)
        out = F.relu(out)  # 활성화 함수 추가
        out = self.layer_norms[layer_idx](out)  # 정규화 추가
        out = self.msg_dropout(out)  # 드롭아웃 적용
        
        # 임베딩 값 모니터링
        # print(f"s_item_new[{layer_idx}, {behavior_idx}] - mean: {out.mean().item()}, std: {out.std().item()}")
        
        return out
    
    def get_scores(self, user_ids: torch.LongTensor):
        """
        user_ids에 대해 모든 아이템에 대한 예측 점수를 한번에 계산
        => evaluate에서 사용
        """
        device = self.device
        user_ids = user_ids.to(device)
        batch_size = user_ids.size(0)
        
        # Ensure that embeddings are computed
        if self.embedding_cached.item() == 0:
            raise RuntimeError("Please call model.encode() before get_scores!")
        
        # user-based CF scores: dot product between user_latent and item_latent
        user_emb = self.user_latent[user_ids]  # shape=(batch_size, emb_dim)
        item_emb = self.item_latent  # shape=(num_items, emb_dim)
        user_cf_scores = torch.matmul(user_emb, item_emb.t())  # shape=(batch_size, num_items)
        
        # item-based CF scores
        # (batch_size, num_items)
        item_cf_scores = self.compute_item_cf_scores_batch(user_ids)
        # # Vectorized computation
        # if self.item_cf_mode == "original":
        #     # Compute s_jt for all users and behaviors
        #     # Shape: (batch_size, len(behaviors), emb_dim)
        #     user_items_emb = []
        #     for t_i, bname in enumerate(self.behaviors):
        #         user_items = [self.user_bhv_dict[bname][u.item()] for u in user_ids]
        #         # Handle empty lists
        #         user_items_emb_t = [
        #             self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 
        #             else torch.zeros(self.embedding_size, device=device) 
        #             for items in user_items
        #         ]
        #         user_items_emb_t = torch.stack(user_items_emb_t)  # shape=(batch_size, emb_dim)
        #         user_items_emb.append(user_items_emb_t)
        #     user_items_emb = torch.stack(user_items_emb, dim=1)  # shape=(batch_size, len(behaviors), emb_dim)
            
        #     # Apply M_t and ReLU
        #     # 'bte, tef -> btf'
        #     transformed = torch.einsum('bte,tef->btf', user_items_emb, self.M_t)
        #     transformed = F.relu(transformed)  # shape=(batch_size, len(behaviors), emb_dim)
            
        #     # Compute scores
        #     # s_it: (batch_size, len(behaviors), num_items, emb_dim)
        #     s_it = self.s_item_list.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape=(batch_size, len(behaviors), num_items, emb_dim)
        #     # transformed: (batch_size, len(behaviors), emb_dim) -> (batch_size, len(behaviors), 1, emb_dim)
        #     transformed = transformed.unsqueeze(2)
        #     # Element-wise multiply and sum over emb_dim
        #     scores_t = torch.sum(transformed * s_it, dim=3)  # shape=(batch_size, len(behaviors), num_items)
            
        #     # Apply alpha
        #     if self.item_alpha:
        #         if self.alpha_mode == "global":
        #             alpha = F.softmax(self.behavior_alpha, dim=0).view(1, -1, 1)  # shape=(1, len(behaviors), 1)
        #             scores_t = scores_t * alpha
        #         elif self.alpha_mode == "per_user":
        #             alpha = self.get_alpha()  # shape=(len(behaviors), num_users)
        #             # Gather alpha for current batch users
        #             alpha = alpha[:, user_ids].transpose(0,1).unsqueeze(2)  # shape=(batch_size, len(behaviors), 1)
        #             scores_t = scores_t * alpha
        #     # Sum over behaviors
        #     item_cf_scores = scores_t.sum(dim=1)  # shape=(batch_size, num_items)
        
        # elif self.item_cf_mode == "unify":
        #     # Similar vectorized approach for 'unify' mode
        #     user_items_emb = []
        #     for t_i, bname in enumerate(self.behaviors):
        #         user_items = [self.user_bhv_dict[bname][u.item()] for u in user_ids]
        #         # Handle empty lists
        #         user_items_emb_t = [
        #             self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 
        #             else torch.zeros(self.embedding_size, device=device) 
        #             for items in user_items
        #         ]
        #         user_items_emb_t = torch.stack(user_items_emb_t)  # shape=(batch_size, emb_dim)
        #         user_items_emb.append(user_items_emb_t)
        #     user_items_emb = torch.stack(user_items_emb, dim=1)  # shape=(batch_size, len(behaviors), emb_dim)
            
        #     # Apply unify_M and ReLU
        #     transformed = torch.matmul(user_items_emb, self.unify_M)  # shape=(batch_size, len(behaviors), emb_dim)
        #     transformed = F.relu(transformed)  # shape=(batch_size, len(behaviors), emb_dim)
            
        #     # Compute scores
        #     s_it = self.s_item_list.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape=(batch_size, len(behaviors), num_items, emb_dim)
        #     transformed = transformed.unsqueeze(2)  # shape=(batch_size, len(behaviors), 1, emb_dim)
        #     scores_t = torch.sum(transformed * s_it, dim=3)  # shape=(batch_size, len(behaviors), num_items)
            
        #     # Apply alpha
        #     if self.item_alpha:
        #         if self.alpha_mode == "global":
        #             alpha = F.softmax(self.behavior_alpha, dim=0).view(1, -1, 1)  # shape=(1, len(behaviors), 1)
        #             scores_t = scores_t * alpha
        #         elif self.alpha_mode == "per_user":
        #             alpha = self.get_alpha()  # shape=(len(behaviors), num_users)
        #             # Gather alpha for current batch users
        #             alpha = alpha[:, user_ids].transpose(0,1).unsqueeze(2)  # shape=(batch_size, len(behaviors), 1)
        #             scores_t = scores_t * alpha
        #     # Sum over behaviors
        #     item_cf_scores = scores_t.sum(dim=1)  # shape=(batch_size, num_items)
        
        # else:
        #     raise ValueError("item_cf_mode must be 'original' or 'unify'")
        
        # Combine user-based CF and item-based CF
        lamb = torch.tensor(self.lamb, dtype=torch.float32, device=device)
        scores = lamb * user_cf_scores + (1.0 - lamb) * item_cf_scores  # shape=(batch_size, num_items)
        
        return scores
    
    def compute_item_cf_scores_batch(self, user_ids: torch.LongTensor):
        """
        user_ids (batch_size,)에 대해 모든 아이템 점수를 벡터화해서 계산
        """
        device = self.device
        batch_size = user_ids.size(0)
        max_iid = self.num_items
        
        # user별로, 각 behavior에서 봤던 아이템 임베딩 평균 -> M or M_t -> s_item_list와 곱
        # 벡터화 구현이 쉽지 않아서, 여기서는 다소 단순 for문 -> torch.stack로 처리
        # 대규모라면 더 효율적 방법 필요
        item_cf_scores = torch.zeros(batch_size, max_iid, device=device)
        alpha_vec = self.get_alpha()
        
        for t_i, bname in enumerate(self.behaviors):
            # s_item_list[t_i] shape = (num_items, emb_dim)
            # unify vs original 따로 처리
            # 여기선 "original"만 예시
            s_whole = self.s_item_list[t_i]  # (num_items, emb_dim)
            
            # user별 아이템 평균 임베딩
            user_items = [self.user_bhv_dict[bname][u.item()] for u in user_ids]
            user_items_emb = [
                self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 else torch.zeros(self.embedding_size, device=device)
                for items in user_items
            ]
            user_items_emb = torch.stack(user_items_emb, dim=0)  # (batch_size, emb_dim)
            
            # transform
            if self.item_cf_mode == "original":
                transformed = F.relu(torch.matmul(user_items_emb, self.M_t[t_i]))  # (batch_size, emb_dim)
            else:
                transformed = F.relu(torch.matmul(user_items_emb, self.unify_M))
            
            # s_whole: (num_items, emb_dim) -> broadcasting
            # (batch_size, 1, emb_dim) x (1, num_items, emb_dim)
            # 여기서는 trick으로 matmul이나 einsum
            # score_t[u,i] = <transformed[u], s_whole[i]>
            score_t = torch.matmul(transformed.unsqueeze(1), s_whole.t().unsqueeze(0)) 
            # => (batch_size, 1, num_items)
            score_t = score_t.squeeze(1)  # (batch_size, num_items)
            
            if self.item_alpha:
                if self.alpha_mode == "global":
                    score_t = alpha_vec[t_i] * score_t
                else:
                    # per_user
                    alpha_per_user = alpha_vec[t_i][user_ids].unsqueeze(1)
                    score_t = alpha_per_user * score_t
            
            item_cf_scores += score_t
        
        return item_cf_scores
            
    # def get_alpha(self):
    #     """
    #     Retrieve alpha values based on alpha_mode.
        
    #     Returns:
    #         - alpha_vec: Tensor
    #             - If alpha_mode == "global": shape=(len(behaviors),)
    #             - If alpha_mode == "per_user": shape=(len(behaviors), num_users)
    #     """
    #     if not self.alpha_learning:
    #         return torch.ones(len(self.behaviors), device=self.device) / len(self.behaviors)

    #     if hasattr(self, "alpha_cache") and self.alpha_cache is not None:
    #         return self.alpha_cache

    #     if self.alpha_mode == "global":
    #         self.alpha_cache = F.softmax(self.behavior_alpha, dim=0).to(self.device)  # shape=(len(behaviors),)
    #     elif self.alpha_mode == "per_user":
    #         # Compute per-user alpha
    #         # print(f"self.behavior_alpha.shape = {self.behavior_alpha.shape}\nalpha_vec.view(-1, 1).shape={self.behavior_alpha.view(-1, 1).shape}")
    #         # print(f"self.behavior_alpha.shape = {self.behavior_alpha.shape}\nalpha_vec.view(-1, 1).shape={self.behavior_alpha.view(-1, 1).shape}")
    #         # print(f"self.user_count={self.user_count}")
    #         # print(f"self.user_count.shape={self.user_count.shape}")
    #         denom = torch.clamp(torch.sum(self.behavior_alpha.view(-1, 1) * self.user_count, dim=0, keepdim=True), min=1e-9)  # shape=(1, num_users)
    #         self.alpha_cache = (self.behavior_alpha.view(-1, 1) * self.user_count) / denom  # shape=(len(behaviors), num_users)
    #     else:
    #         raise ValueError("alpha_mode must be 'global' or 'per_user'")
        
    #     self.alpha_cache = self.alpha_cache.detach()  # detach하여 그래프에서 분리
        
    #     return self.alpha_cache
    
    def get_alpha(self):
        """
        alpha_mode에 따른 alpha 계산
        """
        if not self.alpha_learning:
            return torch.ones(len(self.behaviors), device=self.device) / len(self.behaviors)
        
        if self.alpha_mode == "global":
            # global softmax
            return F.softmax(self.behavior_alpha, dim=0).to(self.device)
        elif self.alpha_mode == "per_user":
            # per_user => alpha[t] * user_count[t,u] / sum_t'(alpha[t']*count[t',u])
            # 그래프 연산 위해 detach는 안 함(원한다면 detach 필요)
            denom = torch.clamp(torch.sum(self.behavior_alpha.view(-1,1) * self.user_count, dim=0, keepdim=True), min=1e-9)
            alpha_ut = (self.behavior_alpha.view(-1,1) * self.user_count) / denom
            return alpha_ut  # shape=(len(behaviors), num_users)
        else:
            raise ValueError("alpha_mode must be 'global' or 'per_user'")
    
    
    def node_dropout_csr(self, mat: sp.csr_matrix, dropout_rate: float, device):
        """
        Apply node dropout on sparse matrix in CSR format.
        
        Args:
            mat (sp.csr_matrix): Sparse matrix in CSR format.
            dropout_rate (float): Dropout ratio.
            device (str): "cpu" or "cuda".
        
        Returns:
            torch.sparse_coo_tensor: Node-dropped sparse tensor.
        """
        if dropout_rate <= 0.0:
            indices = torch.from_numpy(np.vstack((mat.row, mat.col))).long().to(device)
            values = torch.from_numpy(mat.data).float().to(device)
            return torch.sparse_coo_tensor(
                indices, 
                values, 
                torch.Size(mat.shape), 
                dtype=torch.float32, 
                device=device
            ).coalesce()

        mat_coo = mat.tocoo()
        
        # Sparse matrix 데이터를 유지하면서 dropout 적용
        indices = torch.from_numpy(np.vstack((mat_coo.row, mat_coo.col))).long().to(device)
        values = torch.from_numpy(mat_coo.data).float().to(device)

        # Dropout mask 적용
        mask = (torch.rand(len(values), device=device) > dropout_rate).float()
        values = values * mask

        # PyTorch sparse tensor로 변환
        sparse_mat = torch.sparse_coo_tensor(
            indices, values, torch.Size(mat.shape), dtype=torch.float32, device=device
        ).coalesce()

        return sparse_mat
    
    def spmm(self, sp_matrix: torch.sparse_coo_tensor, dense_tensor: torch.Tensor):
        """
        Sparse-dense matrix multiplication on device.

        Args:
            sp_matrix (torch.sparse_coo_tensor): Sparse tensor of shape (M, N).
            dense_tensor (torch.Tensor): Dense tensor of shape (N, D).

        Returns:
            torch.Tensor: Resulting dense tensor of shape (M, D), on device.
        """
        if sp_matrix._nnz() == 0:
            return torch.zeros(sp_matrix.size(0), dense_tensor.size(1), dtype=torch.float32, device=dense_tensor.device)
        result = torch.sparse.mm(sp_matrix, dense_tensor)
        return result
