# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import scipy.sparse as sp
# from typing import Dict, List
# from collections import defaultdict

# class MBGCN(nn.Module):
#     """
#     MB-GCN model for multi-behavior recommendation, referencing Jin et al. (SIGIR 2020).
    
#     We do multi-layer propagation for:
#         - user-item (behavior-aware)
#         - item-item (behavior semantics)
#     Then combine user-based CF score + item-based CF score to get final score.
#     """
#     """
#     01-31 변경 사항:
#     1) 메시지 패싱(encode)은 한 Epoch에 한 번만 호출되도록: self.embedding_cached = 0 -> encode() -> 1
#     2) forward 시 이미 캐싱된 self.user_latent, self.item_latent, self.s_item_list를 사용
#         => 더 이상 대규모 spmm 연산이 반복되지 않도록록
#     """
#     def __init__(
#         self, 
#         num_users: int,
#         num_items: int,
#         embedding_size: int=64,
#         num_layers: int=2,
#         behaviors: List[str]=None, # e.g. ["view", "cart", "purchase"]
#         ui_mats: Dict[str, sp.csr_matrix]=None,
#         ii_mats: Dict[str, sp.csr_matrix]=None,
#         user_bhv_dict: Dict[str, Dict[int, List[int]]]=None,  # 필수 인자로 변경 가능
#         use_pretrain=False,
#         pretrain_user_emb=None,
#         pretrain_item_emb=None,
#         node_dropout: float=0.2,
#         message_dropout: float=0.2,
#         alpha_learning: bool=True,
#         lamb: float=0.5,
#         item_cf_mode: str="original", # "original" or "unify"
#         item_alpha: bool=False,  # default: no alpha at item update
#         alpha_mode: str="global",     # "global" or "per_user"
#         device: str="cpu"
#     ):
#         """
#         Args:
#             - num_users, num_items (int): total counts
#             - embedding_size (int): dimension of embeddings
#             - num_layers (int): GCN propagation depth
#             - behaviors (List[str]): list of behavior names (keys in ui_mats, ii_mats)
#             - ui_mats (Dict[str, sp.csr_matrix]): user-item adjacency (dict of CSR for each behavior)
#             - ii_mats (Dict[str, sp.csr_matrix]): item-item adjacency (dict of CSR for each behavior)
#             - user_bhv_dict (Dict[str, Dict[int, List[int]]]): 사용자별 행동별 아이템 리스트
#             - node_dropout (float): ratio for node dropout
#             - message_dropout (float): ratio for message dropout
#             - alpha_learning (bool): whether to learn a global weight for each behavior
#             - lamb (float): weighting factor for user-based CF vs item-based CF score
#             - item_cf_mode (str):
#                 - "original" => distinct transform & M_t per behavior
#                 - "unify"    => single transform & single M for all behaviors
#             - item_alpha (bool):
#                 - True  => use alpha at item embedding update
#                 - False => no alpha at item embedding update
#             - alpha_mode (str):
#                 - "global"   => alpha[t] is global for all users
#                 - "per_user" => alpha[u,t] = w[t]*n_{u,t}/sum_m w[m]*n_{u,m}
#             - device (str): "cpu" or "cuda"
#         """
#         super().__init__()
#         if behaviors is None:
#             behaviors = ["view", "cart", "purchase"]
            
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embedding_size = embedding_size
#         self.num_layers = num_layers
#         self.behaviors = behaviors
#         self.ui_mats = ui_mats if ui_mats else {}
#         self.ii_mats = ii_mats if ii_mats else {}
#         self.node_dropout = node_dropout
#         self.message_dropout = message_dropout
#         self.alpha_learning = alpha_learning
#         self.lamb = lamb
#         self.item_cf_mode = item_cf_mode
#         self.item_alpha = item_alpha
#         self.alpha_mode = alpha_mode
#         self.device = device
        
#         # 1) LayerNorm을 각 레이어별로 정의
#         self.layer_norms = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_layers)])
        
#         # 2) user & item base embeddings => p_u^(0), q_i^(0)
#         if use_pretrain and (pretrain_user_emb is not None) and (pretrain_item_emb is not None):
#             # pretrain 임베딩 사용
#             self.user_emb = nn.Parameter(torch.tensor(pretrain_user_emb, dtype=torch.float32))
#             self.item_emb = nn.Parameter(torch.tensor(pretrain_item_emb, dtype=torch.float32))
#         else:
#             # Xavier init
#             self.user_emb = nn.Parameter(torch.empty(num_users, embedding_size))
#             self.item_emb = nn.Parameter(torch.empty(num_items, embedding_size))
#             nn.init.xavier_uniform_(self.user_emb, gain=1.0)
#             nn.init.xavier_uniform_(self.item_emb, gain=1.0)
        
#         # 3) s_{i,t} for item-based CF => behavior-specific item embeddings
#         self.s_item_emb = nn.ParameterList()
#         for _ in behaviors:
#             p = nn.Parameter(torch.empty(num_items, embedding_size))
#             nn.init.xavier_uniform_(p, gain=1.0)
#             self.s_item_emb.append(p)
        
#         # 4) alpha param (for user-item weighting)
#         if self.alpha_learning:
#             self.behavior_alpha = nn.Parameter(torch.ones(len(behaviors), dtype=torch.float32))
#         else:
#             self.register_buffer("behavior_alpha", torch.ones(len(behaviors), dtype=torch.float32))
#         # user_count => n_{u,t} if alpha_mode="per_user"
#         self.register_buffer("user_count", torch.zeros(len(behaviors), num_users, dtype=torch.float32))
        
#         # 5) M_t or unify_M => item-based CF transformations
#         if item_cf_mode == "original":
#             # M_t: per behavior
#             self.M_t = nn.Parameter(torch.empty(len(behaviors), embedding_size, embedding_size))
#             nn.init.xavier_uniform_(self.M_t, gain=1.0)
#         elif item_cf_mode == "unify":
#             # single M
#             self.unify_M = nn.Parameter(torch.empty(embedding_size, embedding_size))
#             nn.init.xavier_uniform_(self.unify_M, gain=1.0)
#         else:
#             raise ValueError("item_cf_mode must be 'original' or 'unify'")
        
#         # 6) message_dropout modules
#         self.msg_dropout = nn.Dropout(p=self.message_dropout)

#         # 7) user-item transforms
#         self.ui_transforms = nn.ModuleList([
#             nn.Linear(embedding_size, embedding_size, bias=False)
#             for _ in range(num_layers)
#         ])
        
#         # 8) item-item transforms for behavior-based s_{i,t}
#         self.ii_transforms_behavior = nn.ModuleList()
#         for lidx in range(num_layers):
#             layer_list = nn.ModuleList()
#             for _ in behaviors:
#                 Wt = nn.Linear(embedding_size, embedding_size, bias=False)
#                 nn.init.xavier_uniform_(Wt.weight, gain=1.0)
#                 layer_list.append(Wt)
#             self.ii_transforms_behavior.append(layer_list)
        
#         # 9) 사용자별 행동 데이터 딕셔너리 설정
#         if user_bhv_dict is not None:
#             self.user_bhv_dict = user_bhv_dict
#         else:
#             self.user_bhv_dict = defaultdict(lambda: defaultdict(list))
        
#         # Initialize encoding cache
#         self.register_buffer("user_latent", torch.zeros(num_users, embedding_size, device=self.device))
#         self.register_buffer("item_latent", torch.zeros(num_items, embedding_size, device=self.device))
#         self.register_buffer("s_item_list", torch.zeros(len(behaviors), num_items, embedding_size, device=self.device))
#         self.register_buffer("embedding_cached", torch.tensor(0, device=self.device))  # 0: not cached, 1: cached
#         self.register_buffer("user_mean_emb", torch.zeros(len(behaviors), num_users, embedding_size, device=self.device))
#         # 모든 파라미터와 모듈을 지정한 디바이스로 이동
#         self.to(device)
        
#     def calc_user_count(self):
#         """
#         Precompute n_{u,t} = # of items that user u has interacted with under behavior t
#         We'll store it in self.user_count[t_idx,u].
#         Must be called after we have self.ui_mats ready.
#         """
#         # user_count shape: (len(behaviors), num_users)
#         user_count_temp = torch.zeros(len(self.behaviors), self.num_users, dtype=torch.float32, device=self.device)
#         for t_idx, bname in enumerate(self.behaviors):
#             row_sum = np.array(self.ui_mats[bname].sum(axis=1)).flatten()
#             user_count_temp[t_idx] = torch.from_numpy(row_sum).float()
    
#         self.user_count = user_count_temp  # 이미 device에 있음
#         # print(f"[INFO] user_count Device: {self.user_count.device}")  # 🚀 디버깅용 출력
    
#     def forward(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor):
#         """
#         For BPR usage: forward(user, item) => predicted score.
#         We'll do multi-layer propagation for user_emb, item_emb,
#         then compute user-based CF score + item-based CF score,
#         combine by lamb.
        
#             1) multi-layer propagation
#             2) user-based CF score = dot(u_emb, i_emb)
#             3) item-based CF score => depends on item_cf_mode
#                 - "original"
#                 - "unify"
#             4) combine with lamb: final = lamb * user_cf + (1-lamb) * item_cf

#         Returns: shape=(batch_size,)
#         """
#         """
#         0131 변경사항:
#         - 이미 encode()로부터 계산된 user_latent, item_latent, s_item_list를 사용
#         """
        
#         device = self.device  # Use model's device
#         user_idx = user_idx.to(device)
#         item_idx = item_idx.to(device)
        
#         # encode()가 되어 있지 않으면 에러
#         if self.embedding_cached.item() == 0:
#             raise RuntimeError("Please call model.encode() before forward! (Epoch-level encode)")

#         # user-based CF
#         u_vec = self.user_latent[user_idx] # (batch_size, emb_dim)
#         i_vec = self.item_latent[item_idx] # (batch_size, emb_dim)
#         user_cf_score = torch.sum(u_vec * i_vec, dim=1) # (batch_size,)
        
#         # item-based CF
#         if self.item_cf_mode == "original":
#             item_cf_score = self.item_based_cf_score_original(user_idx, item_idx)
#         else:
#             item_cf_score = self.item_based_cf_score_unify(user_idx, item_idx)
#         item_cf_score = item_cf_score.to(device)
            
#         # combine
#         score = self.lamb * user_cf_score + (1.0 - self.lamb) * item_cf_score
        
#         # 디버그: 출력 크기 확인
#         # print(f"  forward:")
#         # print(f"    user_idx shape: {user_idx.shape}")               # (batch_size,)
#         # print(f"    item_idx shape: {item_idx.shape}")               # (batch_size,)
#         # print(f"    user_cf_score shape: {user_cf_score.shape}")     # (batch_size,)
#         # print(f"    item_cf_score shape: {item_cf_score.shape}")     # (batch_size,)
#         # print(f"    score shape: {score.shape}")                     # (batch_size,)
        
#         return score
    
#     def encode(self):
#         """
#         1) 전체 사용자/아이템 임베딩을 GCN propagate
#         2) item-item GCN도 propagate
#         3) self.user_latent, self.item_latent, self.s_item_list에 결과를 캐싱
#         => Epoch에 1회 호출
#         """
#         self.eval()
#         with torch.no_grad():
#             user_latent, item_latent, s_item_list = self.propagate_embeddings()
#         # 실제 버퍼에 copy_
#         self.user_latent.copy_(user_latent)
#         self.item_latent.copy_(item_latent)
#         self.s_item_list.copy_(torch.stack(s_item_list))  # shape=(len(behaviors), num_items, emb_dim)
#         self.embedding_cached.fill_(1)  # Mark as cached
        
#         # === 0131 추가: 사용자별 아이템 평균 임베딩 precompute ===
#         self.precompute_user_mean_emb()  
        
#         self.train()
    
#     def precompute_user_mean_emb(self):
#         """
#         self.s_item_list[b_idx] (shape: (num_items, emb_dim))가 이미 item-item GCN까지 반영된 상태.
#         => user_bhv_dict[behavior][u]로 유저가 본 item indices.
#         => 평균내어 user_mean_emb[b_idx, u] 저장.
#         """
#         with torch.no_grad():
#             for b_idx, bname in enumerate(self.behaviors):
#                 # s_t: (num_items, emb_dim)
#                 s_t = self.s_item_list[b_idx]  
#                 # buffer to fill
#                 # shape: (num_users, emb_dim)
#                 user_mean_mat = torch.zeros(self.num_users, self.embedding_size, device=self.device)
                
#                 # Python 루프 (num_users 번) => 이후 forward 시점에는 안 돌음
#                 for u in range(self.num_users):
#                     items_u = self.user_bhv_dict[bname][u]
#                     if len(items_u) > 0:
#                         emb_u = s_t[items_u]  # (n_u, emb_dim)
#                         user_mean_mat[u] = emb_u.mean(dim=0)
                
#                 self.user_mean_emb[b_idx].copy_(user_mean_mat)
    
#     ### item-based CF (original vs unify) ###
    
#     def item_based_cf_score_original(
#         self, 
#         user_idx: torch.LongTensor, 
#         item_idx: torch.LongTensor, 
#         # s_item_embs: List[torch.Tensor]
#     ):
#         """
#         eq. (10) from MB-GCN paper:
#             y2(u,i) = sum_{t} sum_{j in N_t^I(u)} [ s_{j,t}^{*T} M_t s_{i,t}^{*}] / |N_t^I(u)|
#             => Possibly multiply alpha[t] if self.item_alpha=True
        
#          Args:
#             - user_idx: Tensor of user indices, shape=(batch_size,)
#             - item_idx: Tensor of item indices, shape=(batch_size,)
#             - s_item_embs: List of tensors, each shape=(num_items, emb_dim)
        
#         Returns:
#             - scores: Tensor of shape=(batch_size,)
#         """
#         """
#         0131 변경사항:
#         - 이미 s_item_list[t_i], self.M_t[t_i] 가 self.encode()에서 계산됨
#         """
#         device = self.device
#         batch_size = user_idx.size(0)
#         alpha_vec = self.get_alpha() # shape=(len(behaviors),) or (len(behaviors), num_users) depending on alpha_mode
                
#         sum_over_t = torch.zeros(batch_size, device=device)
#         for t_i, bname in enumerate(self.behaviors):
#             # s_i_t = self.s_item_list[t_i][item_idx]  # shape=(batch_size, emb_dim)
#             # user_items = [self.user_bhv_dict[bname][u.item()] for u in user_idx]

#             # # 디버그: 사용자별 아이템 리스트 확인
#             # # print(f"    Behavior '{bname}':")
#             # # for u in range(batch_size):
#             #     # print(f"      User {user_idx[u].item()} items: {user_items[u]}")
    
            
#             # # 빈 리스트를 안전하게 처리 (batch_size 유지)
#             # user_items_emb = [
#             #     self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 
#             #     else torch.zeros(self.embedding_size, device=device) 
#             #     for items in user_items
#             # ]
#             # user_items_emb = torch.stack(user_items_emb)  # shape=(batch_size, emb_dim)
#             # # print(f"      user_items_emb shape: {user_items_emb.shape}")  # (batch_size, emb_dim)

#             # transformed = F.relu(torch.matmul(user_items_emb, self.M_t[t_i]))  # 활성화 함수 추가
#             # # print(f"      transformed shape: {transformed.shape}")    # (batch_size, emb_dim)
#             # score_t = torch.sum(transformed * s_i_t, dim=1)  # shape=(batch_size,)
#             # # print(f"      score_t shape: {score_t.shape}")          # (batch_size,)
#             # user_mean_emb[t_i][user_idx] => (batch_size, emb_dim)
#             user_emb_t = self.user_mean_emb[t_i][user_idx]
#             # item_emb => s_item_list[t_i][item_idx] => (batch_size, emb_dim)
#             i_emb_t = self.s_item_list[t_i][item_idx]
            
#             # transform
#             transformed = F.relu(torch.matmul(user_emb_t, self.M_t[t_i]))  # (batch_size, emb_dim)
            
#             # dot product with i_emb_t
#             score_t = torch.sum(transformed * i_emb_t, dim=1)  # (batch_size,)
            
#             if self.item_alpha:
#                 if self.alpha_mode == "global":
#                     sum_over_t = sum_over_t + alpha_vec[t_i] * score_t
#                 elif self.alpha_mode == "per_user":
#                     alpha_per_user = alpha_vec[t_i][user_idx]  # shape=(batch_size,)
#                     sum_over_t = sum_over_t + alpha_per_user * score_t
#                 else:
#                     raise ValueError("alpha_mode must be 'global' or 'per_user'")
#             else:
#                 sum_over_t = sum_over_t + score_t
#             # print(f"      sum_over_t shape after update: {sum_over_t.shape}")  # (batch_size,)
#         return sum_over_t
    
    
#     def item_based_cf_score_unify(
#         self, 
#         user_idx: torch.LongTensor,
#         item_idx: torch.LongTensor,
#         # s_item_embs: List[torch.Tensor]
#     ):
#         """
#         A simpler version of eq.(10), but with a SINGLE M for all behaviors:
#         y2(u,i) = sum_{t} sum_{j in N_t^I(u)} [ s_{j,t}^T M s_{i,t} ] / |N_t^I(u)| * alpha[t]
#         => same M for every t.
#         """
#         device = self.device
#         batch_size = user_idx.size(0)
#         alpha_vec = self.get_alpha()
        
#         # single unify_M
#         if not hasattr(self, "unify_M"):
#             raise NotImplementedError("unify_M not found, item_cf_mode=unify?")
        
#         sum_over_t = torch.zeros(batch_size, device=device)
#         for t_i, bname in enumerate(self.behaviors):
#             # s_i_t = self.s_item_list[t_i][item_idx]  # shape=(batch_size, emb_dim)
#             # user_items = [self.user_bhv_dict[bname][u.item()] for u in user_idx]

#             # # 빈 리스트를 안전하게 처리 (batch_size 유지)
#             # user_items_emb = [
#             #     self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 
#             #     else torch.zeros(self.embedding_size, device=device) 
#             #     for items in user_items
#             # ]
#             # user_items_emb = torch.stack(user_items_emb)  # shape=(batch_size, emb_dim)

#             # transformed = F.relu(torch.matmul(user_items_emb, self.unify_M))
#             # score_t = torch.sum(transformed * s_i_t, dim=1)  # shape=(batch_size,)
#             user_emb_t = self.user_mean_emb[t_i][user_idx]
#             i_emb_t = self.s_item_list[t_i][item_idx]
#             transformed = F.relu(torch.matmul(user_emb_t, self.unify_M))
#             score_t = torch.sum(transformed * i_emb_t, dim=1)

#             if self.item_alpha:
#                 if self.alpha_mode == "global":
#                     sum_over_t = sum_over_t + alpha_vec[t_i] * score_t
#                 elif self.alpha_mode == "per_user":
#                     alpha_per_user = alpha_vec[t_i][user_idx]  # shape=(batch_size,)
#                     sum_over_t = sum_over_t + alpha_per_user * score_t
#                 else:
#                     raise ValueError("alpha_mode must be 'global' or 'per_user'")
#             else:
#                 sum_over_t = sum_over_t + score_t

#         return sum_over_t
    
#     ### multi-layer propagation ###
    
#     def propagate_embeddings(self):
#         """
#         (no_grad)에서 호출됨
#         1) user-item GCN
#         2) item-item GCN
#         => 최종 user_latent, item_latent, s_item_list
#         """
#         user_latent = self.user_emb
#         item_latent = self.item_emb
#         s_item_list = [p for p in self.s_item_emb] # behavior-specific item emb
        
#         for layer_idx in range(self.num_layers):
#             # Debugging: print devices
#             # print(f"propagate_embedding: {user_latent.device}, {item_latent.device}, {layer_idx}")
#             user_latent, item_latent = self.propagate_user_item(user_latent, item_latent, layer_idx)
            
#             # item-item for each behavior emb: s_{i,t} => always distinct
#             s_item_list_new = []
#             if self.ii_mats is not None:
#                 for b_idx, s_emb in enumerate(s_item_list):
#                     s_new = self.propagate_item_item_behavior(s_emb, b_idx, layer_idx)
#                     s_item_list_new.append(s_new)
#             else:
#                 s_item_list_new.append(s_emb)
#             s_item_list = s_item_list_new

#             # 디버깅: 각 레이어별 임베딩의 평균과 표준편차 출력
#             # print(f"Layer {layer_idx+1}/{self.num_layers}:")
#             # print(f"  User Embeddings - Mean: {user_latent.mean().item():.4f}, Std: {user_latent.std().item():.4f}")
#             # print(f"  Item Embeddings - Mean: {item_latent.mean().item():.4f}, Std: {item_latent.std().item():.4f}")
#             # for b_idx, s_new in enumerate(s_item_list):
#             #     print(f"  s_item_emb[{b_idx}] - Mean: {s_new.mean().item():.4f}, Std: {s_new.std().item():.4f}")
        
#         return user_latent, item_latent, s_item_list
    
#     def propagate_user_item(self, user_emb, item_emb, layer_idx: int):
#         """
#         behavior-aware user-item propagation for layer 'layer_idx'
#         """
#         device = user_emb.device
#         # print(f"propagate_user_item: {device}")
#         # alpha_vec을 함수 내에서 한 번만 가져오도록 변경 (최적화)
#         alpha_vec = self.get_alpha()   # shape=(len(behaviors),) or (len(behaviors), num_users)
        
#         # (1) user update
#         user_agg = torch.zeros_like(user_emb, device=device)
        
#         if self.alpha_mode == "global":
#             for t_i, bname in enumerate(self.behaviors):
#                 mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                 user_part = self.spmm(mat, item_emb)  # PyTorch sparse matrix multiplication
#                 user_agg = user_agg + alpha_vec[t_i].item() * user_part  # Weighted sum
                
#         else:  # alpha_mode == 'per_user'
#             user_parts = []
#             for t_i, bname in enumerate(self.behaviors):
#                 mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                 user_part = self.spmm(mat, item_emb)  # shape=(num_users, emb_dim)
#                 user_parts.append(user_part)
                
#             user_parts = torch.stack(user_parts, dim=0)  # shape=(len(behaviors), num_users, emb_dim)
            
#             # Compute per-user alpha weight
#             # print(f"alpha_vec.shape = {alpha_vec.shape}\nalpha_vec.view(-1, 1).shape={alpha_vec.view(-1, 1).shape}")
#             # print(f"self.user_count={self.user_count}")
#             # print(f"self.user_count.shape={self.user_count.shape}")
#             # denom = torch.clamp(torch.sum(alpha_vec.view(-1, 1) * self.user_count, dim=0, keepdim=True), min=1e-9)
#             # alpha_ut = (alpha_vec.view(-1, 1) * self.user_count) / denom  # shape=(len(behaviors), num_users)
#             denom = torch.clamp(torch.sum(alpha_vec * self.user_count, dim=0, keepdim=True), min=1e-9)
#             alpha_ut = (alpha_vec * self.user_count) / denom  # shape=(len(behaviors), num_users)
            
#             # Weighted sum of user embeddings
#             user_agg = torch.sum(alpha_ut.unsqueeze(2) * user_parts, dim=0)  # shape=(num_users, emb_dim)
        
#         # Transform and apply dropout
#         transform_ui = self.ui_transforms[layer_idx]
#         user_new = transform_ui(user_agg)
#         user_new = F.relu(user_new)  # 활성화 함수 추가
#         user_new = self.layer_norms[layer_idx](user_new)  # 정규화 추가
#         user_new = self.msg_dropout(user_new)
        
#         # 임베딩 값 모니터링
#         # print(f"user_new[{layer_idx}] - mean: {user_new.mean().item()}, std: {user_new.std().item()}")
        
#         # (2) item update
#         item_agg = torch.zeros_like(item_emb, device=device)
        
#         if not self.item_alpha:  # item_alpha=False: simple sum of behaviors
#             for bname in self.behaviors:
#                 mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                 item_part = self.spmm(mat.T, user_new)  # PyTorch sparse multiplication
#                 item_agg = item_agg + item_part

#         else:  # item_alpha=True: per-behavior weighted sum
#             if self.alpha_mode == "global":
#                 for t_i, bname in enumerate(self.behaviors):
#                     mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                     item_part = self.spmm(mat.T, user_new)  # shape=(num_items, emb_dim)
#                     item_agg = item_agg + alpha_vec[t_i] * item_part
#             else:  # alpha_mode="per_user"
#                 item_parts = []
#                 for t_i, bname in enumerate(self.behaviors):
#                     mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                     item_part = self.spmm(mat.T, user_new)  # shape=(num_items, emb_dim)
#                     item_parts.append(item_part)
                    
#                 item_parts = torch.stack(item_parts, dim=0)  # shape=(len(behaviors), num_items, emb_dim)

#                 # Compute per-behavior weight, not per-user alpha_ut
#                 item_agg = torch.sum(alpha_vec.view(-1, 1, 1) * item_parts, dim=0)  # shape=(num_items, emb_dim)
        
#         # Transform and apply dropout
#         transform_ui = self.ui_transforms[layer_idx]
#         item_new = transform_ui(item_agg)
#         item_new = F.relu(item_new)  # 활성화 함수 추가
#         item_new = self.layer_norms[layer_idx](item_new)  # 정규화 추가
#         item_new = self.msg_dropout(item_new)
        
#         # 임베딩 값 모니터링
#         # print(f"item_new[{layer_idx}] - mean: {item_new.mean().item()}, std: {item_new.std().item()}")
        
#         return user_new, item_new
    
#     def propagate_item_item_behavior(self, s_item_emb, behavior_idx: int, layer_idx: int):
#         """
#         Propagate item embeddings for a specific behavior and layer.
        
#         Args:
#             - s_item_emb: Tensor, shape=(num_items, emb_dim)
#             - behavior_idx: int, index of the behavior
#             - layer_idx: int, current layer index
        
#         Returns:
#             - s_new: Tensor, shape=(num_items, emb_dim)
#         """
#         device = s_item_emb.device  # Ensure correct device usage
#         # print(f"propagate_item_item_behavior: {device}")
#         bname = self.behaviors[behavior_idx]
#         mat_ii = self.ii_mats.get(bname, sp.csr_matrix((self.num_items, self.num_items)))
        
#         # Apply node dropout
#         mat_drop = self.node_dropout_csr(mat_ii, self.node_dropout, device=device)
        
#         # Perform sparse-dense matrix multiplication
#         item_part = self.spmm(mat_drop, s_item_emb)  # shape=(num_items, emb_dim)
        
#         # Apply behavior-specific transform and dropout
#         transform_t = self.ii_transforms_behavior[layer_idx][behavior_idx]
#         out = transform_t(item_part)
#         out = F.relu(out)  # 활성화 함수 추가
#         out = self.layer_norms[layer_idx](out)  # 정규화 추가
#         out = self.msg_dropout(out)  # 드롭아웃 적용
        
#         # 임베딩 값 모니터링
#         # print(f"s_item_new[{layer_idx}, {behavior_idx}] - mean: {out.mean().item()}, std: {out.std().item()}")
        
#         return out
    
#     def get_scores(self, user_ids: torch.LongTensor):
#         """
#         user_ids에 대해 모든 아이템에 대한 예측 점수를 한번에 계산
#         => evaluate에서 사용
#         """
#         device = self.device
#         user_ids = user_ids.to(device)
#         batch_size = user_ids.size(0)
        
#         # Ensure that embeddings are computed
#         if self.embedding_cached.item() == 0:
#             raise RuntimeError("Please call model.encode() before get_scores!")
        
#         # user-based CF scores: dot product between user_latent and item_latent
#         user_emb = self.user_latent[user_ids]  # shape=(batch_size, emb_dim)
#         item_emb = self.item_latent  # shape=(num_items, emb_dim)
#         user_cf_scores = torch.matmul(user_emb, item_emb.t())  # shape=(batch_size, num_items)
        
#         # item-based CF scores
#         # (batch_size, num_items)
#         item_cf_scores = self.compute_item_cf_scores_batch(user_ids)
#         # # Vectorized computation
#         # if self.item_cf_mode == "original":
#         #     # Compute s_jt for all users and behaviors
#         #     # Shape: (batch_size, len(behaviors), emb_dim)
#         #     user_items_emb = []
#         #     for t_i, bname in enumerate(self.behaviors):
#         #         user_items = [self.user_bhv_dict[bname][u.item()] for u in user_ids]
#         #         # Handle empty lists
#         #         user_items_emb_t = [
#         #             self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 
#         #             else torch.zeros(self.embedding_size, device=device) 
#         #             for items in user_items
#         #         ]
#         #         user_items_emb_t = torch.stack(user_items_emb_t)  # shape=(batch_size, emb_dim)
#         #         user_items_emb.append(user_items_emb_t)
#         #     user_items_emb = torch.stack(user_items_emb, dim=1)  # shape=(batch_size, len(behaviors), emb_dim)
            
#         #     # Apply M_t and ReLU
#         #     # 'bte, tef -> btf'
#         #     transformed = torch.einsum('bte,tef->btf', user_items_emb, self.M_t)
#         #     transformed = F.relu(transformed)  # shape=(batch_size, len(behaviors), emb_dim)
            
#         #     # Compute scores
#         #     # s_it: (batch_size, len(behaviors), num_items, emb_dim)
#         #     s_it = self.s_item_list.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape=(batch_size, len(behaviors), num_items, emb_dim)
#         #     # transformed: (batch_size, len(behaviors), emb_dim) -> (batch_size, len(behaviors), 1, emb_dim)
#         #     transformed = transformed.unsqueeze(2)
#         #     # Element-wise multiply and sum over emb_dim
#         #     scores_t = torch.sum(transformed * s_it, dim=3)  # shape=(batch_size, len(behaviors), num_items)
            
#         #     # Apply alpha
#         #     if self.item_alpha:
#         #         if self.alpha_mode == "global":
#         #             alpha = F.softmax(self.behavior_alpha, dim=0).view(1, -1, 1)  # shape=(1, len(behaviors), 1)
#         #             scores_t = scores_t * alpha
#         #         elif self.alpha_mode == "per_user":
#         #             alpha = self.get_alpha()  # shape=(len(behaviors), num_users)
#         #             # Gather alpha for current batch users
#         #             alpha = alpha[:, user_ids].transpose(0,1).unsqueeze(2)  # shape=(batch_size, len(behaviors), 1)
#         #             scores_t = scores_t * alpha
#         #     # Sum over behaviors
#         #     item_cf_scores = scores_t.sum(dim=1)  # shape=(batch_size, num_items)
        
#         # elif self.item_cf_mode == "unify":
#         #     # Similar vectorized approach for 'unify' mode
#         #     user_items_emb = []
#         #     for t_i, bname in enumerate(self.behaviors):
#         #         user_items = [self.user_bhv_dict[bname][u.item()] for u in user_ids]
#         #         # Handle empty lists
#         #         user_items_emb_t = [
#         #             self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 
#         #             else torch.zeros(self.embedding_size, device=device) 
#         #             for items in user_items
#         #         ]
#         #         user_items_emb_t = torch.stack(user_items_emb_t)  # shape=(batch_size, emb_dim)
#         #         user_items_emb.append(user_items_emb_t)
#         #     user_items_emb = torch.stack(user_items_emb, dim=1)  # shape=(batch_size, len(behaviors), emb_dim)
            
#         #     # Apply unify_M and ReLU
#         #     transformed = torch.matmul(user_items_emb, self.unify_M)  # shape=(batch_size, len(behaviors), emb_dim)
#         #     transformed = F.relu(transformed)  # shape=(batch_size, len(behaviors), emb_dim)
            
#         #     # Compute scores
#         #     s_it = self.s_item_list.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape=(batch_size, len(behaviors), num_items, emb_dim)
#         #     transformed = transformed.unsqueeze(2)  # shape=(batch_size, len(behaviors), 1, emb_dim)
#         #     scores_t = torch.sum(transformed * s_it, dim=3)  # shape=(batch_size, len(behaviors), num_items)
            
#         #     # Apply alpha
#         #     if self.item_alpha:
#         #         if self.alpha_mode == "global":
#         #             alpha = F.softmax(self.behavior_alpha, dim=0).view(1, -1, 1)  # shape=(1, len(behaviors), 1)
#         #             scores_t = scores_t * alpha
#         #         elif self.alpha_mode == "per_user":
#         #             alpha = self.get_alpha()  # shape=(len(behaviors), num_users)
#         #             # Gather alpha for current batch users
#         #             alpha = alpha[:, user_ids].transpose(0,1).unsqueeze(2)  # shape=(batch_size, len(behaviors), 1)
#         #             scores_t = scores_t * alpha
#         #     # Sum over behaviors
#         #     item_cf_scores = scores_t.sum(dim=1)  # shape=(batch_size, num_items)
        
#         # else:
#         #     raise ValueError("item_cf_mode must be 'original' or 'unify'")
        
#         # Combine user-based CF and item-based CF
#         lamb = torch.tensor(self.lamb, dtype=torch.float32, device=device)
#         scores = lamb * user_cf_scores + (1.0 - lamb) * item_cf_scores  # shape=(batch_size, num_items)
        
#         return scores
    
#     def compute_item_cf_scores_batch(self, user_ids: torch.LongTensor):
#         """
#         user_ids (batch_size,)에 대해 모든 아이템 점수를 벡터화해서 계산
#         """
#         device = self.device
#         batch_size = user_ids.size(0)
#         max_iid = self.num_items
        
#         # user별로, 각 behavior에서 봤던 아이템 임베딩 평균 -> M or M_t -> s_item_list와 곱
#         # 벡터화 구현이 쉽지 않아서, 여기서는 다소 단순 for문 -> torch.stack로 처리
#         # 대규모라면 더 효율적 방법 필요
#         item_cf_scores = torch.zeros(batch_size, max_iid, device=device)
#         alpha_vec = self.get_alpha()
        
#         for t_i, bname in enumerate(self.behaviors):
#             # s_item_list[t_i] shape = (num_items, emb_dim)
#             # unify vs original 따로 처리
#             # 여기선 "original"만 예시
#             s_whole = self.s_item_list[t_i]  # (num_items, emb_dim)
            
#             # user별 아이템 평균 임베딩
#             user_items = [self.user_bhv_dict[bname][u.item()] for u in user_ids]
#             user_items_emb = [
#                 self.s_item_emb[t_i][items].mean(dim=0) if len(items) > 0 else torch.zeros(self.embedding_size, device=device)
#                 for items in user_items
#             ]
#             user_items_emb = torch.stack(user_items_emb, dim=0)  # (batch_size, emb_dim)
            
#             # transform
#             if self.item_cf_mode == "original":
#                 transformed = F.relu(torch.matmul(user_items_emb, self.M_t[t_i]))  # (batch_size, emb_dim)
#             else:
#                 transformed = F.relu(torch.matmul(user_items_emb, self.unify_M))
            
#             # s_whole: (num_items, emb_dim) -> broadcasting
#             # (batch_size, 1, emb_dim) x (1, num_items, emb_dim)
#             # 여기서는 trick으로 matmul이나 einsum
#             # score_t[u,i] = <transformed[u], s_whole[i]>
#             score_t = torch.matmul(transformed.unsqueeze(1), s_whole.t().unsqueeze(0)) 
#             # => (batch_size, 1, num_items)
#             score_t = score_t.squeeze(1)  # (batch_size, num_items)
            
#             if self.item_alpha:
#                 if self.alpha_mode == "global":
#                     score_t = alpha_vec[t_i] * score_t
#                 else:
#                     # per_user
#                     alpha_per_user = alpha_vec[t_i][user_ids].unsqueeze(1)
#                     score_t = alpha_per_user * score_t
            
#             item_cf_scores += score_t
        
#         return item_cf_scores
            
#     # def get_alpha(self):
#     #     """
#     #     Retrieve alpha values based on alpha_mode.
        
#     #     Returns:
#     #         - alpha_vec: Tensor
#     #             - If alpha_mode == "global": shape=(len(behaviors),)
#     #             - If alpha_mode == "per_user": shape=(len(behaviors), num_users)
#     #     """
#     #     if not self.alpha_learning:
#     #         return torch.ones(len(self.behaviors), device=self.device) / len(self.behaviors)

#     #     if hasattr(self, "alpha_cache") and self.alpha_cache is not None:
#     #         return self.alpha_cache

#     #     if self.alpha_mode == "global":
#     #         self.alpha_cache = F.softmax(self.behavior_alpha, dim=0).to(self.device)  # shape=(len(behaviors),)
#     #     elif self.alpha_mode == "per_user":
#     #         # Compute per-user alpha
#     #         # print(f"self.behavior_alpha.shape = {self.behavior_alpha.shape}\nalpha_vec.view(-1, 1).shape={self.behavior_alpha.view(-1, 1).shape}")
#     #         # print(f"self.behavior_alpha.shape = {self.behavior_alpha.shape}\nalpha_vec.view(-1, 1).shape={self.behavior_alpha.view(-1, 1).shape}")
#     #         # print(f"self.user_count={self.user_count}")
#     #         # print(f"self.user_count.shape={self.user_count.shape}")
#     #         denom = torch.clamp(torch.sum(self.behavior_alpha.view(-1, 1) * self.user_count, dim=0, keepdim=True), min=1e-9)  # shape=(1, num_users)
#     #         self.alpha_cache = (self.behavior_alpha.view(-1, 1) * self.user_count) / denom  # shape=(len(behaviors), num_users)
#     #     else:
#     #         raise ValueError("alpha_mode must be 'global' or 'per_user'")
        
#     #     self.alpha_cache = self.alpha_cache.detach()  # detach하여 그래프에서 분리
        
#     #     return self.alpha_cache
    
#     def get_alpha(self):
#         """
#         alpha_mode에 따른 alpha 계산
#         """
#         if not self.alpha_learning:
#             return torch.ones(len(self.behaviors), device=self.device) / len(self.behaviors)
        
#         if self.alpha_mode == "global":
#             # global softmax
#             return F.softmax(self.behavior_alpha, dim=0).to(self.device)
#         elif self.alpha_mode == "per_user":
#             # per_user => alpha[t] * user_count[t,u] / sum_t'(alpha[t']*count[t',u])
#             # 그래프 연산 위해 detach는 안 함(원한다면 detach 필요)
#             denom = torch.clamp(torch.sum(self.behavior_alpha.view(-1,1) * self.user_count, dim=0, keepdim=True), min=1e-9)
#             alpha_ut = (self.behavior_alpha.view(-1,1) * self.user_count) / denom
#             return alpha_ut  # shape=(len(behaviors), num_users)
#         else:
#             raise ValueError("alpha_mode must be 'global' or 'per_user'")
    
    
#     def node_dropout_csr(self, mat: sp.csr_matrix, dropout_rate: float, device):
#         """
#         Apply node dropout on sparse matrix in CSR format.
        
#         Args:
#             mat (sp.csr_matrix): Sparse matrix in CSR format.
#             dropout_rate (float): Dropout ratio.
#             device (str): "cpu" or "cuda".
        
#         Returns:
#             torch.sparse_coo_tensor: Node-dropped sparse tensor.
#         """
#         if dropout_rate <= 0.0:
#             indices = torch.from_numpy(np.vstack((mat.row, mat.col))).long().to(device)
#             values = torch.from_numpy(mat.data).float().to(device)
#             return torch.sparse_coo_tensor(
#                 indices, 
#                 values, 
#                 torch.Size(mat.shape), 
#                 dtype=torch.float32, 
#                 device=device
#             ).coalesce()

#         mat_coo = mat.tocoo()
        
#         # Sparse matrix 데이터를 유지하면서 dropout 적용
#         indices = torch.from_numpy(np.vstack((mat_coo.row, mat_coo.col))).long().to(device)
#         values = torch.from_numpy(mat_coo.data).float().to(device)

#         # Dropout mask 적용
#         mask = (torch.rand(len(values), device=device) > dropout_rate).float()
#         values = values * mask

#         # PyTorch sparse tensor로 변환
#         sparse_mat = torch.sparse_coo_tensor(
#             indices, values, torch.Size(mat.shape), dtype=torch.float32, device=device
#         ).coalesce()

#         return sparse_mat
    
#     def spmm(self, sp_matrix: torch.sparse_coo_tensor, dense_tensor: torch.Tensor):
#         """
#         Sparse-dense matrix multiplication on device.

#         Args:
#             sp_matrix (torch.sparse_coo_tensor): Sparse tensor of shape (M, N).
#             dense_tensor (torch.Tensor): Dense tensor of shape (N, D).

#         Returns:
#             torch.Tensor: Resulting dense tensor of shape (M, D), on device.
#         """
#         if sp_matrix._nnz() == 0:
#             return torch.zeros(sp_matrix.size(0), dense_tensor.size(1), dtype=torch.float32, device=dense_tensor.device)
#         result = torch.sparse.mm(sp_matrix, dense_tensor)
#         return result


# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import scipy.sparse as sp
# from typing import Dict, List
# from collections import defaultdict

# class MBGCN(nn.Module):
#     """
#     MB-GCN model (Jin et al., SIGIR 2020) with multi-layer propagation and
#     user-based CF + item-based CF combination.
#     """

#     def __init__(
#         self, 
#         num_users: int,
#         num_items: int,
#         embedding_size: int = 64,
#         num_layers: int = 2,
#         behaviors: List[str] = None,
#         ui_mats: Dict[str, sp.csr_matrix] = None,
#         ii_mats: Dict[str, sp.csr_matrix] = None,
#         user_bhv_dict: Dict[str, Dict[int, List[int]]] = None,
#         use_pretrain = False,
#         pretrain_user_emb = None,
#         pretrain_item_emb = None,
#         node_dropout: float = 0.2,
#         message_dropout: float = 0.2,
#         alpha_learning: bool = True,
#         lamb: float = 0.5,
#         item_cf_mode: str = "bigmat", # "bigmat" (행렬화 방식), 필요시 "original"/"unify" 등 확장 가능
#         item_alpha: bool = False,
#         alpha_mode: str = "global",      # "global" or "per_user"
#         device: str = "cpu"
#     ):
#         super().__init__()
#         if behaviors is None:
#             behaviors = ["view", "cart", "purchase"]
            
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embedding_size = embedding_size
#         self.num_layers = num_layers
#         self.behaviors = behaviors
#         self.ui_mats = ui_mats if ui_mats else {}
#         self.ii_mats = ii_mats if ii_mats else {}
#         self.node_dropout = node_dropout
#         self.message_dropout = message_dropout
#         self.alpha_learning = alpha_learning
#         self.lamb = lamb
#         self.item_cf_mode = item_cf_mode
#         self.item_alpha = item_alpha
#         self.alpha_mode = alpha_mode
#         self.device = device

#         # LayerNorm (per-layer)
#         self.layer_norms = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_layers)])
        
#         # Base user/item embeddings 초기화: MF로 사전학습한 임베딩을 사용할 경우, np.load로 로드.
#         if use_pretrain and pretrain_user_emb and pretrain_item_emb:
#             # pretrain_user_emb과 pretrain_item_emb는 npy 파일 경로
#             user_emb_data = np.load(pretrain_user_emb)
#             item_emb_data = np.load(pretrain_item_emb)
#             self.user_emb = nn.Parameter(torch.tensor(user_emb_data, dtype=torch.float32))
#             self.item_emb = nn.Parameter(torch.tensor(item_emb_data, dtype=torch.float32))
#         else:
#             self.user_emb = nn.Parameter(torch.empty(num_users, embedding_size))
#             self.item_emb = nn.Parameter(torch.empty(num_items, embedding_size))
#             nn.init.xavier_uniform_(self.user_emb, gain=1.0)
#             nn.init.xavier_uniform_(self.item_emb, gain=1.0)
        
#         # s_{i,t}: behavior-specific item embeddings
#         self.s_item_emb = nn.ParameterList()
#         for _ in behaviors:
#             p = nn.Parameter(torch.empty(num_items, embedding_size))
#             nn.init.xavier_uniform_(p, gain=1.0)
#             self.s_item_emb.append(p)
        
#         # alpha param (for user-item weighting)
#         if self.alpha_learning:
#             self.behavior_alpha = nn.Parameter(torch.ones(len(behaviors), dtype=torch.float32))
#         else:
#             self.register_buffer("behavior_alpha", torch.ones(len(behaviors), dtype=torch.float32))
        
#         # user_count => n_{u,t} if alpha_mode="per_user"
#         self.register_buffer("user_count", torch.zeros(len(behaviors), num_users, dtype=torch.float32))
        
#         # M_t or unify_M for item-based CF
#         # if item_cf_mode == "original":
#         #     self.M_t = nn.Parameter(torch.empty(len(behaviors), embedding_size, embedding_size))
#         #     nn.init.xavier_uniform_(self.M_t, gain=1.0)
#         # elif item_cf_mode == "unify":
#         #     self.unify_M = nn.Parameter(torch.empty(embedding_size, embedding_size))
#         #     nn.init.xavier_uniform_(self.unify_M, gain=1.0)
#         # else:
#         #     raise ValueError("item_cf_mode must be 'original' or 'unify'")
#         self.M_t = nn.Parameter(torch.empty(len(behaviors), embedding_size, embedding_size))
#         nn.init.xavier_uniform_(self.M_t, gain=1.0)
        
#         # message_dropout
#         self.msg_dropout = nn.Dropout(p=self.message_dropout)

#         # user-item transforms (one per layer)
#         self.ui_transforms = nn.ModuleList([
#             nn.Linear(embedding_size, embedding_size, bias=False)
#             for _ in range(num_layers)
#         ])
        
#         # item-item transforms (behavior-based) per layer
#         self.ii_transforms_behavior = nn.ModuleList()
#         for lidx in range(num_layers):
#             layer_list = nn.ModuleList()
#             for _ in behaviors:
#                 Wt = nn.Linear(embedding_size, embedding_size, bias=False)
#                 nn.init.xavier_uniform_(Wt.weight, gain=1.0)
#                 layer_list.append(Wt)
#             self.ii_transforms_behavior.append(layer_list)
        
#         # 사용자별 행동 데이터
#         if user_bhv_dict is not None:
#             self.user_bhv_dict = user_bhv_dict
#         else:
#             self.user_bhv_dict = defaultdict(lambda: defaultdict(list))
        
#         # Cache for epoch-level
#         self.register_buffer("user_latent", torch.zeros(num_users, embedding_size, device=self.device))
#         self.register_buffer("item_latent", torch.zeros(num_items, embedding_size, device=self.device))
#         self.register_buffer("s_item_list", torch.zeros(len(behaviors), num_items, embedding_size, device=self.device))
#         self.register_buffer("embedding_cached", torch.tensor(0, device=self.device))  
        
#         # (Optional) user_mean_emb: 현재는 item-based CF에서 "평균 임베딩" 접근에 사용
#         #  그러나 논문 eq.(10) 을 엄밀히 구현하려면, 아래 precompute_user_mean_emb를 안 쓰거나
#         #  별도 sum-of-item 임베딩 방식을 써야 함.
#         # self.register_buffer("user_mean_emb",
#         #                      torch.zeros(len(behaviors), num_users, embedding_size, device=self.device))

#         self.to(device)

#     def calc_user_count(self):
#         """
#         Precompute n_{u,t} = # of items that user u has interacted with under behavior t
#         => used in per_user alpha mode
#         """
#         user_count_temp = torch.zeros(len(self.behaviors), self.num_users, dtype=torch.float32, device=self.device)
#         for t_idx, bname in enumerate(self.behaviors):
#             row_sum = np.array(self.ui_mats[bname].sum(axis=1)).flatten()
#             user_count_temp[t_idx] = torch.from_numpy(row_sum).float()
    
#         self.user_count = user_count_temp

#     def forward(self, user_idx: torch.LongTensor, pos_item_idx: torch.LongTensor, neg_item_idx: torch.LongTensor = None):
#         """
#         BPR(혹은 다른 Loss) 학습 시 batch-wise forward
#           1) multi-layer propagation은 encode()로 사전 계산 (embedding_cached)
#           2) user_cf_score + item_cf_score -> lamb로 combine
#         """
#         """
#         0201 변경사항:
#         - pos_item_idx: (batch_size,)
#         - neg_item_idx: (batch_size,)

#         Returns:
#             - pos_scores: (batch_size,)
#             - neg_scores: (batch_size,)
#         """
#         device = self.device
#         user_idx = user_idx.to(device)
#         pos_item_idx = pos_item_idx.to(device)
#         if neg_item_idx is not None:
#             neg_item_idx = neg_item_idx.to(device)
        
#         if self.embedding_cached.item() == 0:
#             raise RuntimeError("Please call model.encode() before forward!")
        
#         # user-based CF
#         # u_vec = self.user_latent[user_idx]
#         # i_vec = self.item_latent[item_idx]
#         # user_cf_score = torch.sum(u_vec * i_vec, dim=1)
        
#         # 전체 아이템에 대한 점수 계산 (user-based CF + item-based CF의 조합)
#         # get_scores()는 (batch_size, num_items) shape의 점수 행렬을 반환함
#         all_scores = self.get_scores(user_idx) # (batch_size, num_items)
#         # 양의 아이템 점수: 각 배치의 pos_item_idx 위치의 값을 추출
#         pos_scores = all_scores.gather(dim=1, index=pos_item_idx.unsqueeze(1)).squeeze(1)
        
#         if neg_item_idx is not None:
#             # 음의 아이템 점수도 동일하게 gather
#             neg_scores = all_scores.gather(dim=1, index=neg_item_idx.unsqueeze(1)).squeeze(1)
#             return pos_scores, neg_scores
#         else:
#             return pos_scores, None
#     def evaluate(self, user_ids: torch.LongTensor):
#         """
#         0201 추가사항:
#         Args:
#             user_ids (torch.LongTensor): (batch_size,) 사용자 인덱스
#         Returns:
#             scores (torch.Tensor): (batch_size, num_items) 각 사용자의 아이템 점수 행렬
#         """
#         return self.get_scores(user_ids)

#     def encode(self):
#         """
#         1) multi-layer GCN propagate (user-item, item-item)
#         2) cache user_latent, item_latent, s_item_list
#         => epoch마다 1회만
#         """
#         self.eval()
#         with torch.no_grad():
#             user_latent, item_latent, s_item_list = self.propagate_embeddings()

#         self.user_latent.copy_(user_latent)
#         self.item_latent.copy_(item_latent)
#         self.s_item_list.copy_(torch.stack(s_item_list))  
#         self.embedding_cached.fill_(1)

#         # (optional) 만약 "평균 임베딩" 접근을 계속 쓰고 싶다면:
#         # self.precompute_user_mean_emb()  

#         self.train()

#     # def precompute_user_mean_emb(self):
#     #     """
#     #     user_bhv_dict[behavior][u] => user가 본 item list
#     #     => s_item_list[b_idx]에서 해당 아이템 임베딩을 평균.
#     #     => item_based_cf_score_original()의 간소화 용도로 사용.
#     #     """
#     #     with torch.no_grad():
#     #         for b_idx, bname in enumerate(self.behaviors):
#     #             s_t = self.s_item_list[b_idx]
#     #             user_mean_mat = torch.zeros(self.num_users, self.embedding_size, device=self.device)
                
#     #             for u in range(self.num_users):
#     #                 items_u = self.user_bhv_dict[bname][u]
#     #                 if len(items_u) > 0:
#     #                     emb_u = s_t[items_u]
#     #                     user_mean_mat[u] = emb_u.mean(dim=0)
                
#     #             self.user_mean_emb[b_idx].copy_(user_mean_mat)
    
#     ### ---------------- 아이템 기반 CF: big matrix approach -----------------
#     def build_item_cf_matrix_big(self):
#         """
#         아이템 기반 CF 점수 행렬 (num_users x num_items)을 한 번에 구성

#         Steps:
#          1) behavior별로 item-item score S_t = (s_item_list[t] @ M_t) @ s_item_list[t].T
#             => shape=(num_items, num_items)
#          2) user-item adjacency => row-normalize => A_t (num_users, num_items)
#          3) A_t @ S_t => (num_users, num_items) => user_item_score_t
#          4) alpha[t] 곱 (글로벌 or per_user)
#          5) sum_t => item_cf_matrix
#         => O(num_items^2) space/time
#         """
#         device = self.device
#         num_b = len(self.behaviors)
#         alpha_vec = self.get_alpha()  # global => shape=(num_b,) / per_user => (num_b, num_users)

#         # 최종 item-based CF => shape=(num_users, num_items)
#         item_cf_mat = torch.zeros(self.num_users, self.num_items, device=device)

#         for t_i, bname in enumerate(self.behaviors):
#             # 1) item-item score => shape=(num_items, num_items)
#             #    s_item_list[t_i] => (num_items, emb_dim)
#             #    M_t[t_i] => (emb_dim, emb_dim)
#             s_i_t = self.s_item_list[t_i]
#             M_t = self.M_t[t_i]
#             item_item_score = self._build_item_item_score_big(s_i_t, M_t)  # (num_items, num_items)

#             # 2) user-item adjacency -> row-normalize => shape=(num_users, num_items)
#             A_t = self._build_user_item_norm_mat(self.ui_mats[bname].to(device))  # sparse_coo_tensor

#             # 3) user_item_score_t = A_t mm item_item_score => shape=(num_users, num_items)
#             user_item_score_t = torch.sparse.mm(A_t, item_item_score)

#             # 4) alpha => item_alpha=True => multiply alpha[t_i]
#             if self.item_alpha:
#                 if self.alpha_mode == "global":
#                     # alpha_vec[t_i] => scalar
#                     user_item_score_t = alpha_vec[t_i] * user_item_score_t
#                 else:
#                     # per_user => alpha_vec[t_i, u]
#                     # shape=(num_users,) => row-wise multiply
#                     alpha_user = alpha_vec[t_i]  # (num_users,)
#                     user_item_score_t = alpha_user.unsqueeze(1) * user_item_score_t

#             # 5) accumulate
#             item_cf_mat += user_item_score_t

#         return item_cf_mat

#     def _build_item_item_score_big(self, s_item_mat: torch.Tensor, M: torch.Tensor):
#         """
#         s_item_mat: shape=(num_items, emb_dim)
#         M: shape=(emb_dim, emb_dim)

#         Returns: item_item_score => (num_items, num_items)
#            S[i,j] = s_item_mat[i,:] @ M @ s_item_mat[j,:]^T
#         => O(num_items^2)
#         """
#         # (num_items, emb_dim) x (emb_dim, emb_dim) => (num_items, emb_dim)
#         transform = s_item_mat @ M
#         # => (num_items, emb_dim) x (emb_dim, num_items) => (num_items, num_items)
#         return transform @ s_item_mat.t()

#     def _build_user_item_norm_mat(self, ui_mat):
#         """
#         0201 변경사항:
#             - user-item adjacency 행렬 ui_mat을 row-normalize 하여,
#             - 각 행의 합이 1이 되는 sparse_coo_tensor (또는 dense tensor)를 반환
#         Args:
#             - ui_mat: torch.Tensor (희소 또는 밀집) 또는 scipy.sparse matrix
#         """
#         device = self.device  # 항상 self.device 사용
#         if isinstance(ui_mat, torch.Tensor):
#             if ui_mat.is_sparse:
#                 ui_mat = ui_mat.coalesce()
#                 indices = ui_mat._indices().to(device)
#                 values = ui_mat._values().to(device)
#                 row_sum = torch.zeros(ui_mat.size(0), device=device)
#                 row_sum = row_sum.scatter_add_(0, indices[0], values)
#                 row_sum = row_sum.clamp(min=1e-9)
#                 normalized_values = values / row_sum[indices[0]]
#                 return torch.sparse_coo_tensor(indices, normalized_values, ui_mat.size(), device=device).coalesce()
#             else:
#                 ui_mat = ui_mat.to(device).float()
#                 row_sum = ui_mat.sum(dim=1, keepdim=True).clamp(min=1e-9)
#                 return ui_mat / row_sum
#         else:
#             # scipy.sparse matrix의 경우
#             mat_coo = ui_mat.tocoo()
#             row_sum = np.asarray(ui_mat.sum(axis=1)).flatten()
#             row_sum = np.maximum(row_sum, 1e-9)
#             norm_vals = []
#             for r, val in zip(mat_coo.row, mat_coo.data):
#                 norm_vals.append(val / row_sum[r])
#             norm_vals = np.array(norm_vals, dtype=np.float32)
#             indices = torch.from_numpy(np.vstack((mat_coo.row, mat_coo.col))).long().to(device)
#             values = torch.from_numpy(norm_vals).float().to(device)
#             return torch.sparse_coo_tensor(indices, values, size=(ui_mat.shape[0], ui_mat.shape[1]), dtype=torch.float32, device=device).coalesce()
    
#     ########################################
#     #     item-based CF (original / unify)
#     ########################################
#     # def item_based_cf_score_original(self, user_idx, item_idx):
#     #     """
#     #     논문 eq.(10) 근접 구현:
#     #       y2(u, i) = sum_t( alpha[t] * sum_{j in N_t^I(u)} ( s_{j,t}^T * M_t * s_{i,t} ) / |N_t^I(u)| )
#     #     => 아래서는 j별로 합산하는 버전을 직접 구현 (평균 임베딩 사용X)
#     #     => 단, item_alpha, alpha_mode, user_mean_emb 등과 충돌이 없도록 주의.
#     #     """
#     #     device = user_idx.device
#     #     batch_size = user_idx.size(0)
#     #     alpha_vec = self.get_alpha()

#     #     sum_over_t = torch.zeros(batch_size, device=device, dtype=torch.float)

#     #     for t_i, bname in enumerate(self.behaviors):
#     #         # s_item_list[t_i] = s_{i,t}
#     #         s_i_t = self.s_item_list[t_i][item_idx]   # (batch_size, emb_dim)
#     #         M_t = self.M_t[t_i]                       # (emb_dim, emb_dim)

#     #         # 각 user에 대해, N_t^I(u) 아이템들의 임베딩을 개별적으로 transform
#     #         # (방법1) for-loop (batch_size) → for-loop (j items)
#     #         # (방법2) gather+matmul
#     #         # 여기서는 간단히 Python loop로 구현 (소규모 데이터 가정)
#     #         scores_t = torch.zeros(batch_size, device=device)

#     #         for b_i in range(batch_size):
#     #             u_id = user_idx[b_i].item()
#     #             items_u = self.user_bhv_dict[bname][u_id]  # j in N_t^I(u)
#     #             if len(items_u) == 0:
#     #                 continue

#     #             # s_j,t shape=(|N_t^I(u)|, emb_dim)
#     #             s_j_t = self.s_item_list[t_i][items_u]
#     #             # transform => s_j_t @ M_t => shape=(|N_t^I(u)|, emb_dim)
#     #             transformed_j = torch.matmul(s_j_t, M_t)
#     #             # dot-product vs s_i_t[b_i]
#     #             dot_j = torch.sum(transformed_j * s_i_t[b_i], dim=1)  # (|N_t^I(u)|,)
#     #             # 평균
#     #             mean_j = dot_j.mean()
#     #             scores_t[b_i] = mean_j

#     #         # item_alpha (if needed)
#     #         if self.item_alpha:
#     #             if self.alpha_mode == "global":
#     #                 scores_t = alpha_vec[t_i] * scores_t
#     #             else:  # per_user
#     #                 alpha_user = alpha_vec[t_i][user_idx]  # shape=(batch_size,)
#     #                 scores_t = alpha_user * scores_t
#     #         else:
#     #             # alpha 미사용인 경우, t별 합산
#     #             pass

#     #         sum_over_t += scores_t

#     #     return sum_over_t

#     # def item_based_cf_score_unify(self, user_idx, item_idx):
#     #     """
#     #     single M for all behaviors
#     #     y2(u,i) = sum_{t} alpha[t] * sum_{j in N_t^I(u)} ( s_{j,t}^T * unify_M * s_{i,t} ) / |N_t^I(u)|
#     #     """
#     #     device = user_idx.device
#     #     batch_size = user_idx.size(0)
#     #     alpha_vec = self.get_alpha()

#     #     unify = self.unify_M
#     #     sum_over_t = torch.zeros(batch_size, device=device)

#     #     for t_i, bname in enumerate(self.behaviors):
#     #         s_i_t = self.s_item_list[t_i][item_idx]
#     #         scores_t = torch.zeros(batch_size, device=device)

#     #         for b_i in range(batch_size):
#     #             u_id = user_idx[b_i].item()
#     #             items_u = self.user_bhv_dict[bname][u_id]
#     #             if len(items_u) == 0:
#     #                 continue

#     #             s_j_t = self.s_item_list[t_i][items_u]
#     #             transformed_j = torch.matmul(s_j_t, unify)
#     #             dot_j = torch.sum(transformed_j * s_i_t[b_i], dim=1)
#     #             mean_j = dot_j.mean()
#     #             scores_t[b_i] = mean_j

#     #         # item_alpha
#     #         if self.item_alpha:
#     #             if self.alpha_mode == "global":
#     #                 scores_t = alpha_vec[t_i] * scores_t
#     #             else:
#     #                 alpha_user = alpha_vec[t_i][user_idx]
#     #                 scores_t = alpha_user * scores_t

#     #         sum_over_t += scores_t

#     #     return sum_over_t

#     ########################################
#     # multi-layer GCN propagation
#     ########################################
#     def propagate_embeddings(self):
#         """
#         (no_grad)에서 호출됨
#         1) user-item GCN
#         2) item-item GCN
#         => 최종 user_latent, item_latent, s_item_list
#         """
#         user_latent = self.user_emb
#         item_latent = self.item_emb
#         s_item_list = [p for p in self.s_item_emb]

#         for layer_idx in range(self.num_layers):
#             user_latent, item_latent = self.propagate_user_item(user_latent, item_latent, layer_idx)
            
#             s_item_list_new = []
#             if self.ii_mats is not None:
#                 for b_idx, s_emb in enumerate(s_item_list):
#                     s_new = self.propagate_item_item_behavior(s_emb, b_idx, layer_idx)
#                     s_item_list_new.append(s_new)
#             else:
#                 s_item_list_new = s_item_list
            
#             s_item_list = s_item_list_new

#         return user_latent, item_latent, s_item_list

#     def propagate_user_item(self, user_emb, item_emb, layer_idx: int):
#         device = user_emb.device
#         alpha_vec = self.get_alpha()

#         # (1) user update
#         user_agg = torch.zeros_like(user_emb, device=device)

#         if self.alpha_mode == "global":
#             # sum_{t} alpha[t] * (mat * item_emb)
#             for t_i, bname in enumerate(self.behaviors):
#                 mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                 user_part = self.spmm(mat, item_emb)
#                 user_agg += alpha_vec[t_i].item() * user_part
#         else:
#             # per_user: alpha[t_i, u]
#             user_parts = []
#             for t_i, bname in enumerate(self.behaviors):
#                 mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                 user_part = self.spmm(mat, item_emb)
#                 user_parts.append(user_part)

#             user_parts = torch.stack(user_parts, dim=0)
#             denom = torch.clamp(torch.sum(alpha_vec * self.user_count, dim=0, keepdim=True), min=1e-9)
#             alpha_ut = (alpha_vec * self.user_count) / denom
#             # alpha_ut shape=(len(behaviors), num_users)
#             user_agg = torch.sum(alpha_ut.unsqueeze(2) * user_parts, dim=0)

#         # transform
#         transform_ui = self.ui_transforms[layer_idx]
#         user_new = transform_ui(user_agg)
#         user_new = F.relu(user_new)
#         user_new = self.layer_norms[layer_idx](user_new)
#         user_new = self.msg_dropout(user_new)

#         # (2) item update
#         item_agg = torch.zeros_like(item_emb, device=device)

#         if not self.item_alpha:
#             # simple sum
#             for bname in self.behaviors:
#                 mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                 item_part = self.spmm(mat.T, user_new)
#                 item_agg += item_part
#         else:
#             # item_alpha = True
#             if self.alpha_mode == "global":
#                 for t_i, bname in enumerate(self.behaviors):
#                     mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                     item_part = self.spmm(mat.T, user_new) # shape=(num_items, emb_dim)
#                     item_agg += alpha_vec[t_i] * item_part
#             else:
#                 item_parts = []
#                 for t_i, bname in enumerate(self.behaviors):
#                     mat = self.node_dropout_csr(self.ui_mats[bname], self.node_dropout, device=device)
#                     item_part = self.spmm(mat.T, user_new) # shape=(num_items, emb_dim)
#                     item_parts.append(item_part)

#                 item_parts = torch.stack(item_parts, dim=0) # shape=(len(behaviors), num_items, emb_dim)
#                 print(f"alpha_vec.shape = {alpha_vec.shape}")
#                 print(f"alpha_vec.view(-1,1,1).shape = {alpha_vec.view(-1,1,1).shape}")
#                 print(f"item_parts.shape = {item_parts.shape}")
#                 print(f"item_parts = {item_parts}")
#                 item_agg = torch.sum(alpha_vec.view(-1,1,1)* item_parts, dim=0)

#         transform_ui2 = self.ui_transforms[layer_idx]
#         item_new = transform_ui2(item_agg)
#         item_new = F.relu(item_new)
#         item_new = self.layer_norms[layer_idx](item_new)
#         item_new = self.msg_dropout(item_new)

#         return user_new, item_new

#     def propagate_item_item_behavior(self, s_item_emb, behavior_idx: int, layer_idx: int):
#         device = s_item_emb.device
#         bname = self.behaviors[behavior_idx]
#         mat_ii = self.ii_mats.get(bname, sp.csr_matrix((self.num_items, self.num_items)))

#         mat_drop = self.node_dropout_csr(mat_ii, self.node_dropout, device=device)
#         item_part = self.spmm(mat_drop, s_item_emb)

#         transform_t = self.ii_transforms_behavior[layer_idx][behavior_idx]
#         out = transform_t(item_part)
#         out = F.relu(out)
#         out = self.layer_norms[layer_idx](out)
#         out = self.msg_dropout(out)
#         return out

#     def get_scores(self, user_ids: torch.LongTensor):
#         """
#         Evaluate 시 전체 아이템 점수(batch_size, num_items)
#         => user-based CF + item-based CF => lamb vs (1-lamb)
#         => encode() 후 호출
#         """
#         if self.embedding_cached.item() == 0:
#             raise RuntimeError("Please call encode() first")

#         device = self.device
#         user_ids = user_ids.to(device)
#         batch_size = user_ids.size(0)

#         # user-based
#         user_emb = self.user_latent[user_ids]         # (batch_size, emb_dim)
#         item_emb = self.item_latent                   # (num_items, emb_dim)
#         user_cf_scores = torch.matmul(user_emb, item_emb.t())  # (batch_size, num_items)

#         # item-based
#         # 아래 compute_item_cf_scores_batch()를 개선(“sum-of-j” 방식)
#         # item_cf_scores = self.compute_item_cf_scores_batch(user_ids)
        
#         # item-based => shape=(num_users, num_items)
#         big_item_cf = self.build_item_cf_matrix_big()   # (num_users, num_items)
#         # row slice => (batch, num_items)
#         item_cf_scores = big_item_cf[user_ids, :]

#         # lamb_t = torch.tensor(self.lamb, dtype=torch.float32, device=device)
#         # scores = lamb_t * user_cf_scores + (1.0 - lamb_t) * item_cf_scores
#         lam = torch.tensor(self.lamb, dtype=torch.float32, device=device)
#         scores = lam * user_cf_scores + (1.0 - lam) * item_cf_scores
#         return scores

#     # def compute_item_cf_scores_batch(self, user_ids: torch.LongTensor):
#     #     """
#     #     user_ids : (batch_size,)
#     #     => item-based CF 점수를 (batch_size, num_items)로 반환
#     #     => eq.(10)을 batch-wise로 구현한 형태
#     #     """
#     #     device = user_ids.device
#     #     batch_size = user_ids.size(0)
#     #     max_iid = self.num_items
#     #     alpha_vec = self.get_alpha()
        
#     #     # 결과 저장
#     #     item_cf_scores = torch.zeros(batch_size, max_iid, device=device)

#     #     if self.item_cf_mode == "original":
#     #         # per-behavior transform M_t
#     #         for t_i, bname in enumerate(self.behaviors):
#     #             # for each user in batch, compute sum_{j} [ s_{j,t}^T M_t s_{i,t} ] / |N_t^I(u)|
#     #             M_t = self.M_t[t_i]
#     #             for b_i in range(batch_size):
#     #                 u_id = user_ids[b_i].item()
#     #                 items_u = self.user_bhv_dict[bname][u_id]
#     #                 if len(items_u) == 0:
#     #                     continue
#     #                 # shape=(|Nu|, emb_dim)
#     #                 s_j_t = self.s_item_list[t_i][items_u]
#     #                 # transform
#     #                 transform_j = torch.matmul(s_j_t, M_t)   # (|Nu|, emb_dim)

#     #                 # target items => 0..(num_items-1)
#     #                 # broadcast dot product vs each item i
#     #                 # (|Nu|, emb_dim) x (emb_dim, num_items)
#     #                 # => (|Nu|, num_items)
#     #                 # => sum over j => (num_items,)
#     #                 # => then / |Nu|
#     #                 # 1) expand transform_j => (|Nu|, 1, emb_dim)
#     #                 # 2) item embedding s_{i,t_i} => (num_items, emb_dim)
#     #                 #    => transpose => (emb_dim, num_items)
#     #                 all_item_emb = self.s_item_list[t_i]    # (num_items, emb_dim)
#     #                 dot_j_all = torch.matmul(transform_j, all_item_emb.t())  # (|Nu|, num_items)
#     #                 # mean over j
#     #                 mean_j_all = dot_j_all.mean(dim=0)      # (num_items,)

#     #                 item_cf_scores[b_i] += mean_j_all
                
#     #             # alpha 적용 (if item_alpha=True)
#     #             if self.item_alpha:
#     #                 if self.alpha_mode == "global":
#     #                     # alpha_vec[t_i] is scalar
#     #                     item_cf_scores *= alpha_vec[t_i]
#     #                 else:  # per_user
#     #                     # alpha_vec[t_i, user_ids]
#     #                     alpha_u = alpha_vec[t_i][user_ids]   # (batch_size,)
#     #                     # 곱하려면 broadcast
#     #                     item_cf_scores = item_cf_scores * alpha_u.unsqueeze(1)

#     #     else:
#     #         # unify_M
#     #         unify = self.unify_M
#     #         for t_i, bname in enumerate(self.behaviors):
#     #             for b_i in range(batch_size):
#     #                 u_id = user_ids[b_i].item()
#     #                 items_u = self.user_bhv_dict[bname][u_id]
#     #                 if len(items_u) == 0:
#     #                     continue
#     #                 s_j_t = self.s_item_list[t_i][items_u]
#     #                 transform_j = torch.matmul(s_j_t, unify)
#     #                 all_item_emb = self.s_item_list[t_i]
#     #                 dot_j_all = torch.matmul(transform_j, all_item_emb.t())
#     #                 mean_j_all = dot_j_all.mean(dim=0)
#     #                 item_cf_scores[b_i] += mean_j_all

#     #             if self.item_alpha:
#     #                 if self.alpha_mode == "global":
#     #                     item_cf_scores *= alpha_vec[t_i]
#     #                 else:
#     #                     alpha_u = alpha_vec[t_i][user_ids]
#     #                     item_cf_scores = item_cf_scores * alpha_u.unsqueeze(1)
        
#     #     return item_cf_scores
    
#     def get_alpha(self):
#         """
#         alpha_mode에 따른 alpha 계산:
#          - global: softmax(behavior_alpha)
#          - per_user: alpha[t,u] = (behavior_alpha[t] * user_count[t,u]) / sum_t'(behavior_alpha[t'] * user_count[t',u])
#         """
#         if not self.alpha_learning:
#             # 그냥 균일하게
#             return torch.ones(len(self.behaviors), device=self.device) / len(self.behaviors)

#         if self.alpha_mode == "global":
#             return F.softmax(self.behavior_alpha, dim=0).to(self.device)
#         elif self.alpha_mode == "per_user":
#             denom = torch.clamp(torch.sum(self.behavior_alpha.view(-1,1) * self.user_count, dim=0, keepdim=True), min=1e-9)
#             alpha_ut = (self.behavior_alpha.view(-1,1)* self.user_count) / denom
#             return alpha_ut
#         else:
#             raise ValueError("alpha_mode must be 'global' or 'per_user'")

#     def node_dropout_csr(self, mat, dropout_rate: float, device):
#         """
#         0201 변경사항:
#         - user-item adjacency 행렬 mat에 대해 dropout을 적용한 후, torch sparse tensor를 반환.
#         - mat이 이미 torch sparse tensor인 경우와 scipy.sparse matrix인 경우를 구분하여 처리.
#         """
#         if isinstance(mat, torch.Tensor):
#             if mat.is_sparse:
#                 mat = mat.coalesce()
#                 if dropout_rate <= 0.0:
#                     return mat.to(device)
#                 indices = mat._indices().to(device)
#                 values = mat._values().to(device)
#                 mask = (torch.rand(values.size(), device=device) > dropout_rate).float()
#                 new_values = values * mask
#                 return torch.sparse_coo_tensor(indices, new_values, mat.size(), device=device).coalesce()
#             else:
#                 # Dense tensor인 경우: mat을 device로 옮기고 float형으로 캐스팅한 후 dropout 적용
#                 mat = mat.to(device).float()
#                 if dropout_rate <= 0.0:
#                     return mat
#                 mask = (torch.rand_like(mat) > dropout_rate).float()
#                 return mat * mask
#         else:
#             # scipy.sparse matrix의 경우
#             mat_coo = mat.tocoo()
#             indices = torch.from_numpy(np.vstack((mat_coo.row, mat_coo.col))).long().to(device)
#             values = torch.from_numpy(mat_coo.data).float().to(device)
#             mask = (torch.rand(len(values), device=device) > dropout_rate).float()
#             values = values * mask
#             sparse_mat = torch.sparse_coo_tensor(
#                 indices, values, torch.Size(mat.shape), dtype=torch.float32, device=device
#             ).coalesce()
#             return sparse_mat

#     def spmm(self, sp_matrix: torch.Tensor, dense_tensor: torch.Tensor):
#         """
#         0201 변경사항:
#             - sp_matrix와 dense_tensor 간의 행렬 곱셈을 수행
#             - 만약 sp_matrix가 sparse tensor라면 torch.sparse.mm()을 사용하고,
#             - 그렇지 않으면 일반적인 dense matrix multiplication을 수행
#         """
#         if isinstance(sp_matrix, torch.Tensor) and sp_matrix.is_sparse:
#             if sp_matrix._nnz() == 0:
#                 return torch.zeros(sp_matrix.size(0), dense_tensor.size(1), dtype=torch.float32, device=dense_tensor.device)
#             return torch.sparse.mm(sp_matrix, dense_tensor)
#         else:
#             # sp_matrix가 dense tensor인 경우 dense 곱셈 사용
#             return torch.matmul(sp_matrix, dense_tensor)


# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import scipy.sparse as sp
# from typing import Dict, List
# from collections import defaultdict

# class MBGCN(nn.Module):
#     """
#     MBGCN model (Jin et al., SIGIR 2020) with multi-layer propagation
#     combining user-based CF and item-based CF. 
#     개선 사항:
#       - 아이템-아이템 전파는 각 behavior에 대해 간단히 dense 행렬 I를 사용하여 구현
#       - 각 layer마다 user 및 behavior-specific item 임베딩을 갱신하고,
#         최종 item 임베딩은 behavior-specific 임베딩의 평균으로 사용
#     """
#     def __init__(
#         self, 
#         num_users: int,
#         num_items: int,
#         embedding_size: int = 64,
#         num_layers: int = 2,
#         behaviors: List[str] = None,
#         ui_mats: Dict[str, torch.Tensor] = None,   # 각 behavior의 user-item adjacency, torch.sparse_coo_tensor
#         ii_mats: Dict[str, torch.Tensor] = None,   # 각 behavior의 item-item 관계, dense tensor로 가정 (또는 None이면 identity)
#         use_pretrain: bool = False,
#         pretrain_user_emb: str = None,
#         pretrain_item_emb: str = None,
#         node_dropout: float = 0.2,
#         message_dropout: float = 0.2,
#         lamb: float = 0.5,
#         L2_norm: float = 1e-4,  # 추가: L2 정규화 계수
#         device: str = "cpu"
#     ):
#         super().__init__()
#         if behaviors is None:
#             behaviors = ["buy", "cart", "click", "collect"]
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embedding_size = embedding_size
#         self.num_layers = num_layers
#         self.behaviors = behaviors
#         self.lamb = lamb
#         self.node_dropout = node_dropout
#         self.message_dropout = message_dropout
#         self.device = device
#         self.L2_norm = L2_norm  # L2 정규화 계수 저장

#         # Base user/item embeddings 초기화 (MF와 유사하게)
#         if use_pretrain and pretrain_user_emb and pretrain_item_emb:
#             user_emb_data = np.load(pretrain_user_emb)
#             item_emb_data = np.load(pretrain_item_emb)
#             self.user_emb = nn.Parameter(torch.tensor(user_emb_data, dtype=torch.float32, device=device))
#             self.item_emb = nn.Parameter(torch.tensor(item_emb_data, dtype=torch.float32, device=device))
#         else:
#             self.user_emb = nn.Parameter(torch.empty(num_users, embedding_size, device=device))
#             self.item_emb = nn.Parameter(torch.empty(num_items, embedding_size, device=device))
#             nn.init.xavier_uniform_(self.user_emb, gain=1.0)
#             nn.init.xavier_uniform_(self.item_emb, gain=1.0)
        
#         # Behavior-specific item embeddings: 초기에는 global item embedding을 복사합니다.
#         self.s_item_emb = nn.ParameterList([nn.Parameter(self.item_emb.clone()) for _ in behaviors])
#         # self.s_item_emb = nn.ParameterList([
#         #     nn.Parameter(torch.empty(num_items, embedding_size, device=device))
#         #     for _ in behaviors
#         # ])
#         # for param in self.s_item_emb:
#         #     nn.init.xavier_uniform_(param, gain=1.0)
        
#         # user-item adjacency matrices: ui_mats는 dictionary, 각 값은 torch.sparse_coo_tensor (device에 올림)
#         self.ui_mats = {}
#         if ui_mats is not None:
#             for b in behaviors:
#                 self.ui_mats[b] = ui_mats.get(b, 
#                     torch.sparse_coo_tensor(
#                         torch.empty(2,0, device=device),
#                         torch.empty(0, device=device),
#                         size=(num_users, num_items),
#                         device=device
#                     )
#                 ).to(device)
#         else:
#             self.ui_mats = {}  # 실제로는 반드시 제공되어야 함
        
#         # 아이템-아이템 관계 행렬: ii_mats는 dictionary, 각 값은 dense tensor (또는 None이면 identity)
#         self.ii_mats = {}
#         if ii_mats is not None:
#             for b in behaviors:
#                 # ii_mats[b]가 제공되지 않으면, identity 행렬을 사용
#                 self.ii_mats[b] = ii_mats.get(b, torch.eye(num_items, device=device))
#         else:
#             # 모든 behavior에 대해 identity
#             for b in behaviors:
#                 self.ii_mats[b] = torch.eye(num_items, device=device)
        
#         # Dropout layers
#         # self.dropout = nn.Dropout(p=self.message_dropout)
        
#         # Transformation for user-item propagation: one linear layer per propagation layer.
#         self.ui_transforms = nn.ModuleList([nn.Linear(embedding_size, embedding_size, bias=False) for _ in range(num_layers)])
        
#         # Transformation for item-item propagation per behavior per layer.
#         self.ii_transforms = nn.ModuleList([
#             nn.ModuleList([nn.Linear(embedding_size, embedding_size, bias=False) for _ in behaviors])
#             for _ in range(num_layers)
#         ])
        
#         # Learned behavior weights (alpha) for combining behaviors in user propagation.
#         self.alpha = nn.Parameter(torch.ones(len(behaviors), device=device))
        
#         # Final transformation for combining propagation results
#         self.final_W = nn.Linear(embedding_size, embedding_size, bias=False)
#         nn.init.xavier_uniform_(self.final_W.weight, gain=1.0)
        
#         # 캐시 (encode 호출 후 채워질 값들)
#         self.register_buffer("user_latent", torch.zeros(num_users, embedding_size, device=device))
#         self.register_buffer("item_latent", torch.zeros(num_items, embedding_size, device=device))
#         # 각 behavior별 s_item_emb 결과를 저장 (list of tensors)
#         self.register_buffer("embedding_cached", torch.tensor(0, device=device))
        
#         self.to(device)
        
#         print(f"user_emb mean={self.user_emb.mean().item():.4f}, std={self.user_emb.std().item():.4f}")
#         print(f"item_emb mean={self.item_emb.mean().item():.4f}, std={self.item_emb.std().item():.4f}")
#         for i, emb in enumerate(self.s_item_emb):
#             print(f"s_item_emb[{i}] mean={emb.mean().item():.4f}, std={emb.std().item():.4f}")
#         for i, layer in enumerate(self.ui_transforms):
#             print(f"ui_transforms[{i}].weight: mean={layer.weight.mean().item():.4f}, std={layer.weight.std().item():.4f}")
#         for l, layer_list in enumerate(self.ii_transforms):
#             for i, layer in enumerate(layer_list):
#                 print(f"ii_transforms[{l}][{i}].weight: mean={layer.weight.mean().item():.4f}, std={layer.weight.std().item():.4f}")
#         print(f"final_W.weight: mean={self.final_W.weight.mean().item():.4f}, std={self.final_W.weight.std().item():.4f}")
        
#     def apply_dropout(self, mat, dropout_rate, device):
#         """
#         mat이 torch.sparse_coo_tensor인 경우, 값들에 dropout을 적용합니다.
#         만약 dense tensor라면 element-wise dropout.
#         """
#         # if isinstance(mat, torch.Tensor) and mat.is_sparse:
#         #     mat = mat.coalesce()
#         #     indices = mat._indices().to(device)
#         #     values = mat._values().to(device)
#         #     mask = (torch.rand(values.size(), device=device) > dropout_rate).float()
#         #     new_values = values * mask
#         #     return torch.sparse_coo_tensor(indices, new_values, mat.size(), device=device).coalesce()
#         # else:
#         #     # dense tensor
#         #     return F.dropout(mat.to(device).float(), p=dropout_rate)
#         return mat
    
#     def propagate_embeddings(self):
#         """
#         multi-layer propagation:
#          - User-item propagation: 각 behavior의 user-item sparse 행렬 A^b를 이용해, 
#            각 사용자에 대해 이웃 아이템 임베딩을 가중 평균한 후, learned transformation 적용.
#          - Item-item propagation: 각 behavior에 대해, dense item-item 행렬 I^b를 사용해, 
#            behavior-specific item 임베딩을 갱신.
#          - 각 layer마다 전파 후, 다음 layer의 입력으로 사용.
#         최종적으로, self.user_latent, self.item_latent, 그리고 각 behavior의 s_item_emb를 반환.
#         """
#         user_emb = self.user_emb  # (num_users, emb_dim)
#         item_emb = self.item_emb  # (num_items, emb_dim)
#         s_item_emb = [p for p in self.s_item_emb]  # list of tensors, each (num_items, emb_dim)
        
#         for layer in range(self.num_layers):
#             # User-item propagation
#             aggregated_user = torch.zeros_like(user_emb, device=self.device)
#             for b, bname in enumerate(self.behaviors):
#                 A = self.ui_mats[bname]  # sparse tensor (num_users, num_items)
#                 A_drop = self.apply_dropout(A, self.node_dropout, self.device)
#                 neigh = torch.sparse.mm(A_drop, item_emb)  # (num_users, emb_dim)
#                 aggregated_user += self.alpha[b] * neigh
#             aggregated_user = aggregated_user / (self.alpha.sum() + 1e-8)
#             # Transformation and non-linearity
#             user_emb = self.ui_transforms[layer](aggregated_user)
#             # user_emb = F.relu(user_emb)
#             # user_emb = self.dropout(user_emb)
            
#             # Item-item propagation per behavior
#             new_s_item_emb = []
#             for b, bname in enumerate(self.behaviors):
#                 # dense item-item matrix: I = ii_mats[bname]
#                 I = self.ii_mats[bname].to(self.device).float()
#                 # I_drop = F.dropout(I, p=self.node_dropout)
#                 I_drop = I
#                 # propagate behavior-specific item embedding
#                 neigh_item = torch.matmul(I_drop, s_item_emb[b])  # (num_items, emb_dim)
#                 new_s = self.ii_transforms[layer][b](neigh_item)
#                 # new_s = F.relu(new_s)
#                 # new_s = self.dropout(new_s)
#                 new_s_item_emb.append(new_s)
#             s_item_emb = new_s_item_emb
#             # Optionally update global item embedding as average of behavior-specific ones
#             item_emb = torch.stack(s_item_emb, dim=0).mean(dim=0)
        
#         print(f"Layer {layer}: aggregated_user mean={aggregated_user.mean().item():.4f}, std={aggregated_user.std().item():.4f}")
#         print(f"Layer {layer}: item_emb mean={item_emb.mean().item():.4f}, std={item_emb.std().item():.4f}")
#         for b, s in enumerate(s_item_emb):
#             print(f"Layer {layer}: s_item_emb[{b}] mean={s.mean().item():.4f}, std={s.std().item():.4f}")
        
#         return user_emb, item_emb, s_item_emb
    
#     def encode(self):
#         """
#         Propagate embeddings once (per epoch) and cache the results.
#         """
#         self.eval()
#         with torch.no_grad():
#             user_latent, item_latent, s_item_list = self.propagate_embeddings()
#         self.user_latent.copy_(user_latent)
#         self.item_latent.copy_(item_latent)
#         for i, s in enumerate(s_item_list):
#             self.s_item_emb[i].data.copy_(s)
#         self.embedding_cached.fill_(1)
#         self.train()
    
#     def get_scores(self, user_ids: torch.LongTensor):
#         """
#         Compute the final scores for given user_ids.
#         For MF part, use inner product of cached user and item latent embeddings.
#         For MBGCN, one can also incorporate behavior-specific propagation if desired.
#         여기서는 간단하게 user_latent와 item_latent 내적만 사용합니다.
#         """
#         if self.embedding_cached.item() == 0:
#             raise RuntimeError("Please call encode() before evaluate!")
#         user_ids = user_ids.to(self.device)
#         scores = torch.matmul(self.user_latent[user_ids], self.item_latent.t())
#         return scores
    
#     def evaluate(self, user_ids: torch.LongTensor):
#         """
#         평가 시 전체 아이템 점수를 계산하여 반환합니다.
#         """
#         return self.get_scores(user_ids)
    
#     def regularize(self, user_embeddings, item_embeddings):
#         """
#         L2 regularization on user, item, and behavior-specific item embeddings.
#         """
#         reg = torch.norm(self.user_emb, p=2)**2 + torch.norm(self.item_emb, p=2)**2
#         for emb in self.s_item_emb:
#             reg += torch.norm(emb, p=2)**2
#         for layer in self.ui_transforms:
#             for param in layer.parameters():
#                 reg += torch.norm(param, p=2)**2
#         for layer_list in self.ii_transforms:
#             for layer in layer_list:
#                 for param in layer.parameters():
#                     reg += torch.norm(param, p=2)**2
#         reg += torch.norm(self.final_W.weight, p=2)**2
#         return self.L2_norm * reg
    
#     def predict(self, user_embedding, item_embedding):
#         """
#         Prediction function: inner product along embedding dimension.
#         user_embedding: (batch, k, emb_dim)
#         item_embedding: (batch, k, emb_dim)
#         """
#         return torch.sum(user_embedding * item_embedding, dim=2)
    
#     def forward(self, user_ids: torch.LongTensor, pos_item_ids: torch.LongTensor, neg_item_ids: torch.LongTensor = None):
#         """
#         BPR forward: 
#           - 먼저, 반드시 self.encode()를 호출하여 캐시된 임베딩을 사용해야 합니다.
#           - 전체 아이템 점수를 계산한 후, pos, neg 위치의 값을 gather하여 반환합니다.
#         """
#         user_ids = user_ids.to(self.device)
#         pos_item_ids = pos_item_ids.to(self.device)
#         if neg_item_ids is not None:
#             neg_item_ids = neg_item_ids.to(self.device)
#         if self.training:
#             # 학습 시, grad가 흐르도록 propagation 수행 (torch.no_grad() 사용 X)
#             user_latent, item_latent, _ = self.propagate_embeddings()
#         else:
#             if self.embedding_cached.item() == 0:
#                 raise RuntimeError("Please call encode() before forward in evaluation mode!")
#             user_latent = self.user_latent
#             item_latent = self.item_latent
        
#         # 전체 아이템 점수 행렬 계산
#         scores = torch.matmul(user_latent[user_ids], item_latent.t())
#         pos_scores = scores.gather(dim=1, index=pos_item_ids.unsqueeze(1)).squeeze(1)
#         if neg_item_ids is not None:
#             neg_scores = scores.gather(dim=1, index=neg_item_ids.unsqueeze(1)).squeeze(1)
#             return pos_scores, neg_scores
#         else:
#             return pos_scores, None

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from collections import defaultdict

# ----------------- 디버깅용 헬퍼 함수 -----------------
def node_dropout_spmatrix(sp_matrix: torch.Tensor, dropout_rate: float, device: torch.device) -> torch.Tensor:
    """
    sp_matrix에 dropout을 적용합니다.
    - sp_matrix가 sparse_coo_tensor이면 dropout 후 coalesce() 적용.
    - sp_matrix가 dense이면 element-wise dropout 적용.
    디버깅: 입력 sparse tensor의 layout, nnz 등을 출력합니다.
    """
    # print("[DEBUG] node_dropout_spmatrix: 시작")
    if dropout_rate <= 0.0:
        # print("[DEBUG] dropout_rate <= 0.0, 그대로 반환")
        return sp_matrix.to(device)
    
    if sp_matrix.is_sparse:
        try:
            sp_matrix = sp_matrix.coalesce()
        except Exception as e:
            print(f"[DEBUG] sp_matrix.coalesce() 에러: {e}")
        nnz_before = sp_matrix._nnz()
        # print(f"[DEBUG] 입력 sparse matrix: layout={sp_matrix.layout}, nnz={nnz_before}")
        indices = sp_matrix._indices().to(device)
        values = sp_matrix._values().to(device)
        mask = (torch.rand(values.size(), device=device) > dropout_rate).float()
        new_values = values * mask
        nnz_after = (mask > 0).sum().item()
        # print(f"[DEBUG] dropout 후 nnz (예상): {nnz_after}")
        out = torch.sparse_coo_tensor(indices, new_values, sp_matrix.size(), device=device)
        out = out.coalesce()
        # print(f"[DEBUG] 출력 sparse matrix: layout={out.layout}, nnz={out._nnz()}")
        return out
    else:
        # dense tensor
        out = F.dropout(sp_matrix.to(device).float(), p=dropout_rate)
        # print(f"[DEBUG] dense matrix dropout 적용: shape={out.shape}, mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        return out

def spmm(sp_matrix: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """
    sp_matrix: torch.sparse_coo_tensor 또는 dense tensor.
    dense: shape=(N, emb)
    Returns: shape=(M, emb)
    """
    if sp_matrix.is_sparse:
        sp_matrix = sp_matrix.coalesce()
        if sp_matrix._nnz() == 0:
            return torch.zeros(sp_matrix.size(0), dense.size(1), device=dense.device)
        return torch.sparse.mm(sp_matrix, dense)
    else:
        return torch.matmul(sp_matrix, dense)
# -------------------------------------------------------


class MBGCN(nn.Module):
    """
    MBGCN (Jin et al., SIGIR 2020) with:
      - multi-layer user-item propagation (behavior-aware)
      - multi-layer item-item propagation (behavior-specific)
      - caching of embeddings after encode()
    모든 인접행렬은 torch.sparse_coo_tensor를 사용.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        behaviors: List[str],
        ui_mats: Dict[str, torch.sparse_coo_tensor],  # user-item adjacency per behavior
        ii_mats: Dict[str, torch.sparse_coo_tensor],      # item-item adjacency per behavior (dense 혹은 sparse)
        embedding_size: int = 64,
        num_layers: int = 2,
        alpha_mode: str = "global",  # "global" 또는 "per_user"
        alpha_learning: bool = True,
        node_dropout: float = 0.1,
        message_dropout: float = 0.1,
        lamb: float = 0.5,
        device: str = "cpu"
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.behaviors = behaviors
        self.ui_mats = ui_mats
        self.ii_mats = ii_mats
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.alpha_mode = alpha_mode
        self.alpha_learning = alpha_learning
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.lamb = lamb
        self.device = torch.device(device) if isinstance(device, str) else device
        # 아이템-아이템 관계 행렬 설정 (기존 코드 아래에 추가)
        if ii_mats is not None:
            for b in behaviors:
                # ii_mats[b]가 제공되지 않으면 identity 행렬 사용
                self.ii_mats[b] = ii_mats.get(b, torch.eye(num_items, device=device))
        else:
            for b in behaviors:
                self.ii_mats[b] = torch.eye(num_items, device=device)

        # === ii_mats 디버깅 출력 추가 ===
        # print("==== ii_mats 상태 확인 ====")
        # for key, mat in self.ii_mats.items():
        #     # ii_mats가 sparse 텐서인지 확인
        #     if isinstance(mat, torch.Tensor):
        #         print(f"ii_mats[{key}]: shape={mat.shape}, is_sparse={mat.is_sparse}")
        #         if not mat.is_sparse:
        #             print(f"   mean={mat.float().mean().item():.4f}, std={mat.float().std().item():.4f}")
        #     else:
        #         print(f"ii_mats[{key}] is not a torch.Tensor!")
        # print("========================")
        
        self.item_norms = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_layers)])
        # Base embeddings 초기화
        gain = 0.1
        self.user_emb = nn.Parameter(torch.empty(num_users, embedding_size))
        self.item_emb = nn.Parameter(torch.empty(num_items, embedding_size))
        nn.init.xavier_uniform_(self.user_emb, gain=gain)
        nn.init.xavier_uniform_(self.item_emb, gain=gain)

        # behavior-specific item embeddings
        self.s_item_emb = nn.ParameterList([nn.Parameter(self.item_emb.clone()) for _ in behaviors])
        # (참고: 초기값이 평균 0인 것은 정상이므로 그대로 사용)

        # alpha parameter for user-item weighting
        if self.alpha_learning:
            self.behavior_alpha = nn.Parameter(torch.ones(len(behaviors)))
        else:
            self.register_buffer("behavior_alpha", torch.ones(len(behaviors)))
        # per-user alpha: user_count
        self.register_buffer("user_count", torch.zeros(len(behaviors), num_users))

        # user-item transforms (one per layer)
        self.ui_transforms = nn.ModuleList([
            nn.Linear(embedding_size, embedding_size, bias=False) for _ in range(num_layers)
        ])
        # ui_transforms 가중치도 Xavier 초기화 (gain 적용)
        for layer in self.ui_transforms:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        # item-item transforms per behavior per layer
        self.ii_transforms = nn.ModuleList([
            nn.ModuleList([nn.Linear(embedding_size, embedding_size, bias=False) for _ in behaviors])
            for _ in range(num_layers)
        ])
        for block in self.ii_transforms:
            for layer in block:
                nn.init.xavier_uniform_(layer.weight, gain=gain)

        # message dropout
        self.msg_drop = nn.Dropout(p=self.message_dropout)

        # caching buffers (for encode)
        self.register_buffer("user_latent", torch.zeros(num_users, embedding_size))
        self.register_buffer("item_latent", torch.zeros(num_items, embedding_size))
        self.register_buffer("cached", torch.tensor(0))  # 0: not cached, 1: cached

        self.to(self.device)

        # 디버깅: 초기 임베딩 상태 출력
        # print("==== 초기 임베딩 상태 ====")
        # print(f"user_emb: mean={self.user_emb.mean().item():.4f}, std={self.user_emb.std().item():.4f}")
        # print(f"item_emb: mean={self.item_emb.mean().item():.4f}, std={self.item_emb.std().item():.4f}")
        # for i, emb in enumerate(self.s_item_emb):
            # print(f"s_item_emb[{i}]: mean={emb.mean().item():.4f}, std={emb.std().item():.4f}")
        # print("========================")

    def calc_user_count(self):
        """
        user_count[t, u] = # of items user u has for behavior t.
        """
        for t_idx, bname in enumerate(self.behaviors):
            sp_tensor = self.ui_mats[bname].coalesce()
            row_idx = sp_tensor._indices()[0]
            vals = sp_tensor._values()
            count = torch.zeros(self.num_users, device=self.device)
            count = count.index_add_(0, row_idx, vals)
            self.user_count[t_idx] = count
            # print(f"[DEBUG] user_count for {bname}: mean={count.mean().item():.4f}, max={count.max().item()}")

    def get_alpha(self):
        """
        global: softmax(behavior_alpha) → shape=(T,)
        per_user: shape=(T, num_users)
        """
        if not self.alpha_learning:
            return torch.ones(len(self.behaviors), device=self.device) / len(self.behaviors)
        if self.alpha_mode == "global":
            a = F.softmax(self.behavior_alpha, dim=0)
            # print(f"[DEBUG] global alpha: {a}")
            return a
        else:
            denom = torch.clamp(
                torch.sum(self.behavior_alpha.view(-1, 1) * self.user_count, dim=0, keepdim=True),
                min=1e-9
            )
            alpha_ut = (self.behavior_alpha.view(-1, 1) * self.user_count) / denom
            # print(f"[DEBUG] per_user alpha: mean={alpha_ut.mean().item():.4f}")
            return alpha_ut

    def encode(self):
        """
        매 epoch마다 1회 호출: propagate_embeddings()를 통해 최종 임베딩을 계산하여 캐싱
        """
        self.eval()
        with torch.no_grad():
            user_latent, item_latent, s_list = self.propagate_embeddings()
        self.user_latent.copy_(user_latent)
        self.item_latent.copy_(item_latent)
        for i, s in enumerate(s_list):
            self.s_item_emb[i].data.copy_(s)
        self.cached.fill_(1)
        # print("[DEBUG] encode() 완료: user_latent mean={:.4f}, item_latent mean={:.4f}".format(
            # self.user_latent.mean().item(), self.item_latent.mean().item()))
        self.train()

    def propagate_embeddings(self):
        """
        multi-layer propagation:
          - 사용자: 각 behavior별 ui_mats를 사용하여 아이템 임베딩을 집계하고 ui_transforms로 변환.
          - 아이템: 각 behavior별 ui_mats^T를 사용하여 사용자 임베딩 집계.
          - 아이템-아이템: 각 behavior별 ii_mats에 대해 item-item 전파.
        디버깅 코드를 추가하여 각 레이어 종료 후 평균과 표준편차 출력.
        """
        user_emb = self.user_emb
        item_emb = self.item_emb
        s_list = [p for p in self.s_item_emb]

        for layer_idx in range(self.num_layers):
            # print(f"---- Layer {layer_idx} 시작 ----")
            alpha = self.get_alpha()  # global: (T,), per_user: (T, num_users)
            # (1) 사용자 업데이트
            user_agg = torch.zeros_like(user_emb, device=self.device)
            if self.alpha_mode == "global":
                for t_i, bname in enumerate(self.behaviors):
                    A_sp = self.ui_mats[bname]
                    A_sp = node_dropout_spmatrix(A_sp, self.node_dropout, self.device)
                    try:
                        A_sp = A_sp.coalesce()
                    except Exception as e:
                        print(f"[DEBUG] Layer {layer_idx} coalesce 에러 for {bname}: {e}")
                    nnz = A_sp._nnz() if A_sp.is_sparse else "dense"
                    # print(f"[DEBUG] Layer {layer_idx} - ui_mats[{bname}] nnz={nnz}")
                    part = spmm(A_sp, item_emb)
                    user_agg += alpha[t_i] * part
            else:
                user_parts = []
                for t_i, bname in enumerate(self.behaviors):
                    A_sp = node_dropout_spmatrix(self.ui_mats[bname], self.node_dropout, self.device)
                    A_sp = A_sp.coalesce()
                    part = spmm(A_sp, item_emb)
                    user_parts.append(part)
                user_parts = torch.stack(user_parts, dim=0)
                user_agg = torch.sum(alpha.unsqueeze(-1) * user_parts, dim=0)
            # 사용자 선형 변환 및 활성화 (디버깅을 위해 ReLU와 dropout 주석 처리 가능)
            u_new = self.ui_transforms[layer_idx](user_agg)
            u_new = F.leaky_relu(u_new, negative_slope=0.1)
            u_new = self.msg_drop(u_new)
            # print(f"[DEBUG] Layer {layer_idx} - 사용자 업데이트: mean={u_new.mean().item():.4f}, std={u_new.std().item():.4f}")
            
            # (2) 아이템 업데이트
            item_agg = torch.zeros_like(item_emb, device=self.device)
            for bname in self.behaviors:
                A_sp = node_dropout_spmatrix(self.ui_mats[bname], self.node_dropout, self.device)
                A_sp = A_sp.coalesce()
                part = spmm(A_sp.transpose(0,1), u_new)
                item_agg += part
            i_new = self.ui_transforms[layer_idx](item_agg)
            i_new = F.leaky_relu(i_new, negative_slope=0.1)
            i_new = self.item_norms[layer_idx](i_new)  # 추가: LayerNorm 적용
            i_new = self.msg_drop(i_new)
            # print(f"[DEBUG] Layer {layer_idx} - 아이템 업데이트: mean={i_new.mean().item():.4f}, std={i_new.std().item():.4f}")

            # (3) 아이템-아이템 전파 (behavior-specific)
            s_list_new = []
            for t_i, bname in enumerate(self.behaviors):
                s_i = s_list[t_i]
                A_ii = self.ii_mats[bname].to(self.device).float()
                # A_ii가 sparse인 경우 dropout 적용; dense인 경우 그냥 사용
                A_ii_dropped = node_dropout_spmatrix(A_ii, self.node_dropout, self.device)
                if A_ii_dropped.is_sparse:
                    try:
                        A_ii_dropped = A_ii_dropped.coalesce()
                    except Exception as e:
                        print(f"[DEBUG] Layer {layer_idx} ii_mats[{bname}] coalesce 에러: {e}")
                # dense-dense 곱셈
                if A_ii_dropped.is_sparse:
                    A_ii_dropped = A_ii_dropped.coalesce()
                    item_part = torch.sparse.mm(A_ii_dropped, s_i)
                else:
                    # 만약 dense이고, dropout을 적용하면 너무 많이 0으로 바뀐다면 dropout을 스킵:
                    item_part = torch.matmul(A_ii, s_i)
                # 디버깅: item_part 출력
                # print(f"[DEBUG] Layer {layer_idx} - behavior '{bname}': item_part mean={item_part.mean().item():.4f}, std={item_part.std().item():.4f}")
                s_out = self.ii_transforms[layer_idx][t_i](item_part)
                # residual 연결에 스케일 팩터 적용 (예: 0.5)
                s_out = s_i + 0.5*s_out
                # print(f"[DEBUG] Layer {layer_idx} - behavior '{bname}': s_out (pre-activation) mean={s_out.mean().item():.4f}, std={s_out.std().item():.4f}")
                s_out = F.leaky_relu(s_out, negative_slope=0.1)
                s_out = self.msg_drop(s_out)
                s_list_new.append(s_out)
                # print(f"[DEBUG] Layer {layer_idx} - s_item_emb[{t_i}]: mean={s_out.mean().item():.4f}, std={s_out.std().item():.4f}")
            # s_list = s_list_new

            # Global item embedding 업데이트: behavior별 임베딩 평균
            item_emb = torch.stack(s_list, dim=0).mean(dim=0)
            # 사용자, 아이템 임베딩을 다음 레이어 입력으로 사용
            user_emb = u_new
            # print(f"---- Layer {layer_idx} 종료: user_emb mean={u_new.mean().item():.4f}, std={u_new.std().item():.4f}; item_emb mean={item_emb.mean().item():.4f}, std={item_emb.std().item():.4f} ----")

        return user_emb, item_emb, s_list

    def get_scores(self, user_ids: torch.LongTensor):
        """
        최종 점수: cached된 user_latent와 item_latent의 내적 (user-based CF)
        """
        if self.cached.item() == 0:
            raise RuntimeError("Please call encode() first")
        user_ids = user_ids.to(self.device)
        scores = torch.matmul(self.user_latent[user_ids], self.item_latent.t())
        # print(f"[DEBUG] get_scores: scores mean={scores.mean().item():.4f}, std={scores.std().item():.4f}")
        return scores

    def forward(self, user_ids: torch.LongTensor, pos_item_ids: torch.LongTensor, neg_item_ids: torch.LongTensor = None):
        """
        BPR forward: gather pos/neg 점수를 get_scores()로부터.
        디버깅: pos, neg 점수의 통계 출력.
        """
        user_ids = user_ids.to(self.device)
        pos_item_ids = pos_item_ids.to(self.device)
        if neg_item_ids is not None:
            neg_item_ids = neg_item_ids.to(self.device)
        # 학습 시: 캐시 대신 최신 propagate_embeddings() 사용
        if self.training:
            user_latent, item_latent, _ = self.propagate_embeddings()
        else:
            if self.cached.item() == 0:
                raise RuntimeError("Please call encode() before forward in evaluation mode!")
            user_latent = self.user_latent
            item_latent = self.item_latent

        scores = torch.matmul(user_latent[user_ids], item_latent.t())
        # print(f"[DEBUG] forward: scores mean={scores.mean().item():.4f}, std={scores.std().item():.4f}")
        pos_scores = scores.gather(dim=1, index=pos_item_ids.unsqueeze(1)).squeeze(1)
        if neg_item_ids is not None:
            neg_scores = scores.gather(dim=1, index=neg_item_ids.unsqueeze(1)).squeeze(1)
            # print(f"[DEBUG] forward: pos_scores mean={pos_scores.mean().item():.4f}, neg_scores mean={neg_scores.mean().item():.4f}")
            return pos_scores, neg_scores
        else:
            # print(f"[DEBUG] forward: pos_scores mean={pos_scores.mean().item():.4f}")
            return pos_scores, None

    def regularize(self):
        """
        L2 정규화: base embedding, s_item_emb, ui_transforms, ii_transforms, final_W 등.
        디버깅: 정규화 항 출력.
        """
        reg = torch.norm(self.user_emb, p=2).pow(2) + torch.norm(self.item_emb, p=2).pow(2)
        for emb in self.s_item_emb:
            reg += torch.norm(emb, p=2).pow(2)
        for layer in self.ui_transforms:
            reg += torch.norm(layer.weight, p=2).pow(2)
        for block in self.ii_transforms:
            for sublayer in block:
                reg += torch.norm(sublayer.weight, p=2).pow(2)
        # 만약 final_W가 있다면 추가 (여기서는 생략)
        # print(f"[DEBUG] regularization term: {reg.item():.4f}")
        return reg

    def evaluate(self, user_ids: torch.LongTensor):
        return self.get_scores(user_ids)

