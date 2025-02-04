import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === Base Model (변경 없음) ===
class ModelBase(nn.Module):
    def __init__(self, args: argparse.Namespace, trainset, device: torch.device):
        super().__init__()
        self.embed_size = args.embedding_size
        self.L2_norm = args.L2_norm
        self.device = device
        self.num_users = trainset.num_users
        self.num_items = trainset.num_items

        if args.create_embeddings:
            self.item_embedding = nn.Parameter(torch.empty(self.num_items, self.embed_size, device=self.device))
            nn.init.xavier_normal_(self.item_embedding)
            self.user_embedding = nn.Parameter(torch.empty(self.num_users, self.embed_size, device=self.device))
            nn.init.xavier_normal_(self.user_embedding)
        else:
            load_path = os.path.join(args.pretrain_path, 'model.pkl')
            load_data = torch.load(load_path, map_location='cpu')
            if not args.pretrain_frozen:
                self.item_embedding = nn.Parameter(F.normalize(load_data['item_embedding']).to(self.device))
                self.user_embedding = nn.Parameter(F.normalize(load_data['user_embedding']).to(self.device))
            else:
                self.item_embedding = F.normalize(load_data['item_embedding']).to(self.device)
                self.user_embedding = F.normalize(load_data['user_embedding']).to(self.device)

    def propagate(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def regularize(self, user_embeddings, item_embeddings):
        return self.L2_norm * ((user_embeddings ** 2).sum() + (item_embeddings ** 2).sum())

    def forward(self, users, items):
        user_emb_final, item_emb_final, item_item_emb_dict = self.propagate()
        # user_emb_final: (num_batch, d_total), item_emb_final: (num_items_selected, d_total)
        batch_user_emb = user_emb_final[users].unsqueeze(1).expand(-1, items.shape[1], -1)
        batch_item_emb = item_emb_final[items]
        score1 = torch.sum(batch_user_emb * batch_item_emb, dim=2)
        
        # score2: item-based CF scoring (계산 방식은 기존 코드와 유사하게)
        # 각 행동 t에 대해, 사용자 u가 행동 t로 상호작용한 아이템들의 item-item 임베딩과 후보 아이템의 item-item 임베딩을 투영 후 내적을 수행.
        # 여기서는 간단하게 각 행동별 최종 임베딩을 사용하여 평균을 내는 예시로 구현합니다.
        score2 = 0
        for t, s_emb in item_item_emb_dict.items():
            # s_emb: (num_items, d_total)
            # relation_dict[t]는 사용자-아이템 관계 sparse tensor (노드 드롭아웃은 이미 propagation 시 적용됨)
            rel = self.relation_dict[t]
            # 사용자 u의 행동 t에 의한 이웃 아이템 임베딩 평균 (크기: num_users x d_total)
            # (분자: rel * s_emb, 분모: user_behavior_degree[:, t].unsqueeze(-1))
            eps = 1e-8
            user_deg = self.user_behavior_degree[:, t].unsqueeze(-1).float() + eps
            user_item_agg = torch.sparse.mm(rel, s_emb) / user_deg  # (num_users, d_total)
            # 투영: 기존 코드에서는 추가 projection 행렬(self.item_behavior_W)을 사용함.
            # 여기서는 각 행동별로 저장된 self.item_behavior_W_score[t] (예시로)로 투영한다고 가정.
            proj = self.item_behavior_W_score[t]  # (d_total, d_total)
            user_proj = torch.mm(user_item_agg, proj)  # (num_users, d_total)
            # 후보 아이템의 s_emb 투영
            item_proj = torch.mm(s_emb, proj)  # (num_items, d_total)
            # 최종 내적
            score2 += torch.mm(user_proj[users], item_proj.t())[torch.arange(users.shape[0]).unsqueeze(1), items]
        # score2를 행동 수로 나누어 평균 (만약 행동별 임베딩이 모두 존재한다면)
        score2 = score2 / len(self.relation_dict)

        # 최종 점수: λ * score1 + (1 - λ) * score2
        scores = self.lamb * score1 + (1 - self.lamb) * score2
        L2_loss = self.regularize(batch_user_emb, batch_item_emb)
        return scores, L2_loss

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

# === MBGCN Model (Multi-layer propagation 적용) ===
class MBGCN(ModelBase):
    def __init__(self, args: argparse.Namespace, trainset, device: torch.device):
        super().__init__(args, trainset, device)
        # 학습 데이터 및 그래프 관련 변수
        self.mgnn_weight = args.mgnn_weight
        self.relation_dict = trainset.relation_dict      # dict: 각 행동별 사용자-아이템 관계 (sparse tensor)
        self.item_graph = trainset.item_graph            # dict: 각 행동별 아이템-아이템 그래프 (sparse tensor)
        self.train_matrix = trainset.train_matrix.to(self.device)
        self.relation = trainset.relation                # (필요시 사용)
        self.lamb = args.lamb
        self.item_graph_degree = trainset.item_graph_degree  # dict: 각 행동별 아이템 그래프의 degree
        self.user_behavior_degree = trainset.user_behavior_degree.to(self.device)  # (num_users, num_behaviors)
        self.num_layers = args.num_layers

        # 드롭아웃 설정 (각 레이어에 적용)
        self.message_drop = nn.Dropout(p=args.message_dropout)
        self.node_drop = nn.ModuleDict({
            t: nn.Dropout(p=args.node_dropout) for t in self.relation_dict
        })
        self.train_node_drop = nn.Dropout(p=args.node_dropout)

        # -- 파라미터 초기화 --
        # --- Multi-layer용 선형 변환 행렬 초기화 ---
        # User-Item propagation: 각 레이어마다 W^(l): (d x d)
        self.user_item_W_list = nn.ParameterList([
            nn.Parameter(torch.empty(self.embed_size, self.embed_size, device=self.device))
            for _ in range(self.num_layers)
        ])
        for W in self.user_item_W_list:
            nn.init.xavier_normal_(W)

        # Item-Item propagation: 각 행동별, 각 레이어마다 W_t^(l): (d x d)
        # self.item_item_W: dict, key: 행동 t, value: ParameterList (num_layers, (d x d))
        self.item_item_W = {}
        for t in self.relation_dict:
            param_list = nn.ParameterList([
                nn.Parameter(torch.empty(self.embed_size, self.embed_size, device=self.device))
                for _ in range(self.num_layers)
            ])
            for param in param_list:
                nn.init.xavier_normal_(param)
            self.item_item_W[t] = param_list
        self.item_item_W = nn.ModuleDict(self.item_item_W)

        # score2에서 사용할 추가 투영 행렬 (논문과 코드에서 사용했던 self.item_behavior_W 역할)
        # 각 행동 t마다 (d_total x d_total)로 정의 (d_total = embed_size * (num_layers+1))
        # 여기서는 간단하게 d_total x d_total로 초기화
        self.item_behavior_W_score = {}
        self.d_total = self.embed_size * (self.num_layers + 1)
        for t in self.relation_dict:
            proj = nn.Parameter(torch.empty(self.d_total, self.d_total, device=self.device))
            nn.init.xavier_normal_(proj)
            self.item_behavior_W_score[t] = proj
        self.item_behavior_W_score = nn.ParameterDict(self.item_behavior_W_score)

        # _to_gpu: 모든 sparse tensor를 device로 이동
        self._to_gpu()

        # 기존 코드에서 사용하던 mgnn_weight는 행동별 중요도(w_t)를 학습하는 파라미터입니다.
        # shape: (num_behaviors,). 여기서도 그대로 Parameter로 설정.
        self.mgnn_weight = nn.Parameter(torch.tensor(self.mgnn_weight, dtype=torch.float32, device=self.device))

    def _to_gpu(self):
        for key in self.relation_dict:
            self.relation_dict[key] = self.relation_dict[key].to(self.device)
        for key in self.item_graph:
            self.item_graph[key] = self.item_graph[key].to(self.device)
        for key in self.item_graph_degree:
            self.item_graph_degree[key] = self.item_graph_degree[key].to(self.device)
            
    def _compute_user_behavior_weight(self):
        # mgnn_weight: (num_behaviors,) → softmax 후 unsqueeze(-1)로 (num_behaviors, 1)
        weight = torch.softmax(self.mgnn_weight, dim=0).unsqueeze(-1)  # (num_behaviors, 1)
        # total_weight: (num_users, 1)
        total_weight = torch.mm(self.user_behavior_degree.float(), weight)
        # user_behavior_weight: 각 column마다 weight를 곱한 후 총합으로 나누기 위해, 
        # elementwise 계산이 더 편리하므로 weight를 다시 (1, num_behaviors)로 변환
        weight_row = weight.transpose(0, 1)  # (1, num_behaviors)
        user_behavior_weight = (self.user_behavior_degree.float() * weight_row) / (total_weight + 1e-8)
        return user_behavior_weight

    def propagate(self):
        """
        Multi-layer propagation.
        각 레이어마다 다음을 수행:
          1. User-Item propagation: 각 행동 t에 대해, 
             p_{u,t}^{(l)} = aggregate(아이템 임베딩 I^{(l)}의 이웃, relation_dict[t] 적용, node dropout, message dropout)
             → 사용자별 가중치(α_{u,t})로 가중합 → 선형변환(W^(l)) 적용하여 U^{(l+1)} 도출.
          2. Item propagation from user side: (train_matrix를 통해)
             I_user^{(l+1)} = aggregate(사용자 임베딩 U^{(l)}) → 선형변환(W^(l)) 적용.
          3. Item-Item propagation: 각 행동 t에 대해,
             s_{i,t}^{(l+1)} = W_t^(l) * aggregate(아이템 I^{(l)}의 이웃, item_graph[t] 적용)
          4. 각 레이어마다 dropout 적용.
        최종적으로, 레이어 0 (초기 임베딩)부터 레이어 L까지의 임베딩을 concat하여 반환.
        """
        eps = 1e-8
        num_behaviors = self.user_behavior_degree.shape[1]
        # 초기 임베딩: 레이어 0
        U_list = [self.user_embedding]  # list of (num_users, d)
        I_list = [self.item_embedding]  # list of (num_items, d)
        # 각 행동 t에 대한 item-item 임베딩도 초기값으로 self.item_embedding을 사용 (dict: key-> list)
        S_dict = { t: [self.item_embedding] for t in self.relation_dict }

        # 사전에 행동별 사용자 가중치 (α_{u,t}) 계산 (모든 레이어에서 동일하다고 가정)
        user_behavior_weight = self._compute_user_behavior_weight()  # (num_users, num_behaviors)

        # multi-layer propagation 반복
        U_current = self.user_embedding
        I_current = self.item_embedding
        for l in range(self.num_layers):
            # --- User-Item propagation ---
            # 각 행동 t별로 사용자에 대해 이웃 아이템 임베딩 집계 (node dropout, message dropout 적용)
            agg_list = []
            for idx, t in enumerate(self.relation_dict):
                # relation matrix에 node dropout 적용
                rel = self.relation_dict[t]
                # 각 행동 t에 대해, dropout은 각 값에 대해 적용
                # (이미 ModuleDict로 정의된 dropout 사용)
                indices = rel._indices()
                values = rel._values()
                values = self.node_drop[t](values)
                rel_drop = torch.sparse_coo_tensor(indices, values, rel.shape, device=self.device)
                # 집계: p_{u,t}^{(l)} = rel_drop * I_current, 정규화 (사용자별 행동 degree)
                user_deg = self.user_behavior_degree[:, idx].unsqueeze(-1).float() + eps
                p_ut = torch.sparse.mm(rel_drop, I_current) / user_deg
                p_ut = self.message_drop(p_ut)
                agg_list.append(p_ut)  # list의 각 원소: (num_users, d)
            # 스택해서 (num_users, num_behaviors, d)
            P_ut = torch.stack(agg_list, dim=1)
            # 각 사용자 u에 대해, 행동별 가중치 (α_{u,t}) 적용 → (num_users, d)
            # user_behavior_weight: (num_users, num_behaviors) → unsqueeze(-1)하여 곱함
            U_neighbor = torch.sum(user_behavior_weight.unsqueeze(-1) * P_ut, dim=1)
            # 선형 변환: W^(l) (d x d) 적용
            U_new = torch.matmul(U_neighbor, self.user_item_W_list[l])
            # --- Item propagation from user side ---
            # train_matrix: (num_users, num_items), 여기에도 node dropout 적용
            indices = self.train_matrix._indices()
            values = self.train_matrix._values()
            values = self.train_node_drop(values)
            train_matrix_drop = torch.sparse_coo_tensor(indices, values, self.train_matrix.shape, device=self.device, dtype=torch.float32)
            # I_user = (train_matrix_drop^T * U_current) / (아이템별 사용자 상호작용 수)
            # (여기서는 train_matrix의 열 degree를 사용)
            # train_matrix_drop.sum(dim=0) (sparse tensor의 경우 직접 계산 필요하므로, trainset에서 미리 계산해두었다고 가정)
            item_deg = self.train_matrix.sum(dim=0).to_dense().unsqueeze(-1).float() + eps
            I_user = torch.sparse.mm(train_matrix_drop.t(), U_current) / item_deg
            I_user = self.message_drop(I_user)
            I_new = torch.matmul(I_user, self.user_item_W_list[l])
            # --- Item-Item propagation (각 행동별) ---
            # 각 행동 t에 대해, s_{i,t}^{(l+1)} = W_t^(l) * ( (item_graph[t] * I_current) / (item_graph_degree[t] + eps) )
            S_new_dict = {}
            for t in self.item_graph:
                # item_graph[t]: (num_items, num_items) sparse tensor, apply dropout if 원하는 경우
                graph = self.item_graph[t]
                indices = graph._indices()
                values = graph._values()
                # (노드 dropout을 적용할 수 있음; 여기서는 간단히 사용)
                values = self.node_drop[t](values)
                graph_drop = torch.sparse_coo_tensor(indices, values, graph.shape, device=self.device)
                deg = self.item_graph_degree[t].unsqueeze(-1).float() + eps
                agg_item = torch.sparse.mm(graph_drop, I_current) / deg  # (num_items, d)
                agg_item = self.message_drop(agg_item)
                # 선형 변환: 각 행동별 W_t^(l) 적용
                S_new = torch.matmul(agg_item, self.item_item_W[t][l])
                S_new_dict[t] = S_new  # (num_items, d)
            # --- 업데이트: 다음 레이어 입력은 propagation 결과 ---
            U_current = U_new
            I_current = I_new
            # 각 리스트에 현재 레이어 임베딩 저장
            U_list.append(U_current)
            I_list.append(I_current)
            for t in S_dict:
                S_dict[t].append(S_new_dict[t])
        # 최종 임베딩: 각 레이어(0부터 L까지) 임베딩을 concat (axis=1)
        U_final = torch.cat(U_list, dim=1)  # (num_users, d*(L+1))
        I_final = torch.cat(I_list, dim=1)  # (num_items, d*(L+1))
        # 각 행동별 item-item 임베딩 최종: concat
        S_final_dict = {}
        for t in S_dict:
            S_final_dict[t] = torch.cat(S_dict[t], dim=1)  # (num_items, d*(L+1))
        return U_final, I_final, S_final_dict

    def evaluate(self, user):
        # 평가 시에는 propagate()를 통해 전체 아이템에 대한 임베딩을 얻고,
        # forward와 유사하게 점수를 계산합니다.
        U_final, I_final, S_final_dict = self.propagate()
        user_feature = U_final[user]  # (num_users_eval, d_total)
        score1 = torch.matmul(user_feature, I_final.t())
        score2 = 0
        for t, s_emb in S_final_dict.items():
            user_deg = self.user_behavior_degree[:, list(self.relation_dict.keys()).index(t)].unsqueeze(-1).float() + 1e-8
            user_item_agg = torch.sparse.mm(self.relation_dict[t], s_emb) / user_deg
            proj = self.item_behavior_W_score[t]
            user_proj = torch.matmul(user_item_agg, proj)
            item_proj = torch.matmul(s_emb, proj)
            score2 += torch.matmul(user_proj[user], item_proj.t())
        score2 = score2 / len(self.relation_dict)
        scores = self.lamb * score1 + (1 - self.lamb) * score2
        return scores

