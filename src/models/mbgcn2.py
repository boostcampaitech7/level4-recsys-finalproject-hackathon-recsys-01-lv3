import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp  # (필요시 사용)

# === Base Model ===
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
        """
        raw embeddings -> embeddings for predicting  
        (반환: (user_embedding, item_embedding))
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """
        예측 대상 임베딩 -> 점수 계산  
        (반환: 점수 텐서)
        """
        raise NotImplementedError

    def regularize(self, user_embeddings, item_embeddings):
        """
        임베딩에 대한 정규화 항 (기본: L2 정규화)
        """
        return self.L2_norm * ((user_embeddings ** 2).sum() + (item_embeddings ** 2).sum())

    def forward(self, users, items):
        # propagate()로부터 예측에 필요한 user, item 임베딩을 얻음
        users_feature, item_feature = self.propagate()
        item_embeddings = item_feature[items]
        # users_feature는 (batch, embed_size)일 수 있으므로 expand를 통해 item 개수 만큼 복제
        user_embeddings = users_feature[users].expand(-1, items.shape[1], -1)
        pred = self.predict(user_embeddings, item_embeddings)
        L2_loss = self.regularize(user_embeddings, item_embeddings)
        return pred, L2_loss

    def evaluate(self, *args, **kwargs):
        """
        테스트 시 전체 item에 대한 점수를 계산하는 함수 (모델별로 구현)
        """
        raise NotImplementedError

# === MF Model (Matrix Factorization) ===
class MF(ModelBase):
    def __init__(self, args: argparse.Namespace, trainset, device: torch.device):
        super().__init__(args, trainset, device)

    def propagate(self, task: str = 'train'):
        return self.user_embedding, self.item_embedding

    def predict(self, user_embedding, item_embedding):
        # 각 사용자-아이템 쌍의 내적을 계산
        return torch.sum(user_embedding * item_embedding, dim=2)

    def evaluate(self, propagate_result, users):
        users_feature, item_feature = propagate_result
        user_feature = users_feature[users]
        scores = torch.mm(user_feature, item_feature.t())
        return scores

# === MBGCN Model ===
class MBGCN(ModelBase):
    def __init__(self, args: argparse.Namespace, trainset, device: torch.device):
        super().__init__(args, trainset, device)
        # 학습 데이터 및 그래프 관련 변수
        self.relation_dict = trainset.relation_dict  # dict: 각 행동 별 사용자-아이템 관계 (sparse tensor)
        self.mgnn_weight = args.mgnn_weight  # 예: list 또는 numpy array
        self.item_graph = trainset.item_graph  # dict: 각 행동별 아이템-아이템 그래프
        self.train_matrix = trainset.train_matrix.to(self.device)
        self.relation = trainset.relation
        self.lamb = args.lamb
        self.item_graph_degree = trainset.item_graph_degree
        self.user_behavior_degree = trainset.user_behavior_degree.to(self.device)

        # 드롭아웃 설정
        self.message_drop = nn.Dropout(p=args.message_dropout)
        self.train_node_drop = nn.Dropout(p=args.node_dropout)
        self.node_drop = nn.ModuleList([nn.Dropout(p=args.node_dropout) for _ in self.relation_dict])

        self._to_gpu()
        self._param_init()

    def _to_gpu(self):
        # 모든 sparse tensor를 device로 이동
        for key in self.relation_dict:
            self.relation_dict[key] = self.relation_dict[key].to(self.device)
        for key in self.item_graph:
            self.item_graph[key] = self.item_graph[key].to(self.device)
        for key in self.item_graph_degree:
            self.item_graph_degree[key] = self.item_graph_degree[key].to(self.device)

    def _decode_weight(self):
        # softmax를 통해 사용자 행동 가중치를 산출 (필요 시 호출)
        weight = torch.softmax(self.mgnn_weight, dim=0).unsqueeze(-1)
        total_weight = torch.mm(self.user_behavior_degree, weight)
        self.user_behavior_weight = self.user_behavior_degree.float() / (total_weight + 1e-8)

    def _param_init(self):
        # mgnn_weight를 Parameter로 변환
        self.mgnn_weight = nn.Parameter(torch.tensor(self.mgnn_weight, dtype=torch.float32, device=self.device))
        # 각 행동에 대한 아이템 행동 변환 행렬 초기화
        self.item_behavior_W = nn.ParameterList([
            nn.Parameter(torch.empty(self.embed_size * 2, self.embed_size * 2, device=self.device))
            for _ in self.mgnn_weight
        ])
        for param in self.item_behavior_W:
            nn.init.xavier_normal_(param)
        # 아이템 전파 행렬 초기화
        self.item_propagate_W = nn.ParameterList([
            nn.Parameter(torch.empty(self.embed_size, self.embed_size, device=self.device))
            for _ in self.mgnn_weight
        ])
        for param in self.item_propagate_W:
            nn.init.xavier_normal_(param)
        # 최종 feature 결합을 위한 두 개의 변환 행렬:
        # 사용자 전파(feature from multi-behavior)는 embed_size*2 (예: 128)에서 embed_size (64)로 변환
        self.W_user = nn.Parameter(torch.empty(self.embed_size * 2, self.embed_size, device=self.device))
        nn.init.xavier_normal_(self.W_user)
        # 아이템 전파(feature from train_matrix)는 embed_size (64)에서 embed_size (64)로 변환
        self.W_item = nn.Parameter(torch.empty(self.embed_size, self.embed_size, device=self.device))
        nn.init.xavier_normal_(self.W_item)

    def forward(self, user, item):
        # 학습 행렬에 대해 node dropout 적용
        indices = self.train_matrix._indices()
        values = self.train_matrix._values()
        values = self.train_node_drop(values)
        train_matrix = torch.sparse_coo_tensor(indices, values, size=self.train_matrix.shape, dtype=torch.float32, device=self.device)

        # 디버깅: train_matrix shape 및 nnz 출력
        # print("DEBUG: train_matrix shape:", train_matrix.shape, "nnz:", train_matrix._nnz())
        
        # 사용자 행동 가중치 계산
        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behavior_degree, weight)
        user_behavior_weight = self.user_behavior_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)

        # 각 행동(relation)에 대해 아이템 및 사용자 전파를 수행
        score2 = None
        for i, key in enumerate(self.relation_dict):
            # relation 행렬에 노드 드롭아웃 적용
            rel_indices = self.relation_dict[key]._indices()
            rel_values = self.relation_dict[key]._values()
            rel_values = self.node_drop[i](rel_values)
            tmp_relation_matrix = torch.sparse_coo_tensor(rel_indices, rel_values, size=self.relation_dict[key].shape, dtype=torch.float32, device=self.device)

            # 디버깅: 각 relation의 sparse 행렬 shape 확인
            # print(f"DEBUG: Relation '{key}' shape:", tmp_relation_matrix.shape)
            
            # 아이템 전파: (아이템 그래프 * item_embedding) 정규화 후 행렬 곱
            tmp_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8),
                self.item_propagate_W[i]
            )
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)
            tmp_item_embedding = tmp_item_propagation[item]
            # 디버깅: tmp_item_embedding shape
            # print(f"DEBUG: For relation '{key}', tmp_item_embedding shape:", tmp_item_embedding.shape)
            
            # 사용자 주변 아이템 정보 집계
            tmp_user_neighbour = torch.mm(tmp_relation_matrix, self.item_embedding) / (
                self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8
            )
            tmp_user_item_neighbour_p = torch.mm(tmp_relation_matrix, tmp_item_propagation) / (
                self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8
            )
            tbehavior_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behavior_W[i])
            # 기존: tbehavior_item_projection[user].expand(-1, item.shape[1], -1)
            indexed_proj = tbehavior_item_projection[user]  # shape: [batch_size, feature_dim]
            # 디버깅: indexed_proj shape
            # print(f"DEBUG: For relation '{key}', indexed_proj shape:", indexed_proj.shape)
            
            # 새 차원 추가 후 expand: shape: [batch_size, item.shape[1], feature_dim]
            tuser_tbehavior_item_projection = indexed_proj.unsqueeze(1).expand(-1, item.shape[1], -1)
            # 두 텐서의 element-wise 곱: tmp_item_embedding의 shape도 [batch_size, item.shape[1], feature_dim]여야 함
            product = tuser_tbehavior_item_projection * tmp_item_embedding  # shape: [batch_size, 1, 128]
            # sum(dim=2)를 사용하지 않고, singleton 차원을 제거하여 [batch_size, feature_dim] 형태로 만듦
            score_component = product.squeeze(1)  # 결과 shape: [batch_size, 128]
            # 디버깅: score_component shape
            # print(f"DEBUG: For relation '{key}', score_component shape:", score_component.shape)
            
            if score2 is None:
                score2 = score_component
            else:
                score2 += score_component
        # Normalization over number of relations
        score2 = score2 / len(self.mgnn_weight)  # score2: [batch_size, embed_size*2] i.e. [batch_size, 128]
        # print("DEBUG: score2 shape after normalization:", score2.shape)
        
        # 사용자-아이템 전파 (train_matrix로부터 사용자 feature 추출)
        item_feature_temp = torch.mm(train_matrix.t(), self.user_embedding)  # [num_items, embed_size] i.e. [num_items, 64]
        item_feature_prop = torch.mm(item_feature_temp, self.W_item)          # [num_items, 64]
        # 사용자 propagated feature: transform score2 from 128 to 64
        user_feature_prop = torch.mm(score2, self.W_user)                      # [batch_size, 64]
        # 디버깅: user_feature_prop, item_feature_prop shape
        # print("DEBUG: user_feature_prop shape:", user_feature_prop.shape)
        # print("DEBUG: item_feature_prop shape:", item_feature_prop.shape)
        
        # 최종 feature 결합: 원래 embedding과 전파된 feature를 concat
        user_feature = torch.cat((self.user_embedding[user], user_feature_prop), dim=1)  # [num_users, 64+64=128] for training, but indexing uses batch user indices
        item_feature = torch.cat((self.item_embedding[item].squeeze(1), item_feature_prop[item].squeeze(1)), dim=1)  # [num_items, 64+64=128]
        # print("DEBUG: user_feature shape before dropout:", user_feature.shape)
        # print("DEBUG: item_feature shape before dropout:", item_feature.shape)

        # 메시지 드롭아웃 적용
        user_feature = self.message_drop(user_feature)
        item_feature = self.message_drop(item_feature)

        # 최종 예측: 
        # user_feature와 item_feature는 모두 [batch_size, feature_dim] (예: [4096, 128])입니다.
        # y1: 사용자 기반 점수로 두 feature의 element-wise 곱을 모두 합산하여 스칼라 값을 얻습니다.
        score1 = torch.sum(user_feature * item_feature, dim=1, keepdim=True)  # [batch_size, 1]

        scores = score1 + self.lamb * score2
        L2_loss = self.regularize(user_feature, item_feature)
        # print("DEBUG: L2_loss:", L2_loss.item())

        return scores, L2_loss

    def evaluate(self, user):
        # 평가에서는 train_matrix를 그대로 사용하여 아이템 feature를 추출
        item_feature_temp = torch.mm(self.train_matrix.t(), self.user_embedding)
        item_feature_prop = torch.mm(item_feature_temp, self.W_item)
        item_feature = torch.cat((self.item_embedding, item_feature_prop), dim=1)  # [num_items, embed_size+embed_size]

        # 사용자 propagated feature (score2) 계산: 학습 시와 유사하게 for 루프 적용
        score2 = None
        for i, key in enumerate(self.relation_dict):
            tmp_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8),
                self.item_propagate_W[i]
            )
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1) # [num_items, embed_size*2]
            tmp_user_item_neighbour_p = torch.mm(self.relation_dict[key], tmp_item_propagation) / (
                self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8
            )
            tbehavior_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behavior_W[i]) # [num_users, embed_size*2]
            # indexed_proj = tbehavior_item_projection[user]  # shape: [batch_size, embed_size*2]
            # tuser_tbehavior_item_projection = indexed_proj.unsqueeze(1).expand(-1, 1, -1)
            # product = tuser_tbehavior_item_projection * tmp_item_propagation[user].unsqueeze(1)
            # score_component = product.squeeze(1)  # shape: [batch_size, embed_size*2]
            # 이제 batch 사용자에 대해 선택 (user는 사용자 인덱스 tensor)
            if score2 is None:
                score2 = tbehavior_item_projection[user]
            else:
                score2 += tbehavior_item_projection[user]
        score2 = score2 / len(self.mgnn_weight)  # [batch_size, embed_size*2]
        
        # 만약 score2가 1D라면 2D로 보정 (batch_size가 1일 경우 발생할 수 있음)
        if score2.dim() == 1:
            score2 = score2.unsqueeze(0)
        
        # 사용자 propagated feature: transform score2 from embed_size*2 to embed_size
        user_feature_prop = torch.mm(score2, self.W_user)  # [batch_size, embed_size]
        # 최종 user feature: 배치 내 유저 임베딩과 전파된 feature를 concat
        user_feature = torch.cat((self.user_embedding[user], user_feature_prop), dim=1)  # [batch_size, embed_size*2]
        # 최종 예측 점수: user_feature와 전체 item_feature의 내적 계산
        scores = torch.mm(user_feature, item_feature.t())  # [batch_size, num_items]
        print(f"DEBUG: evaluate() scores shape: {scores.shape}")
        return scores
    # def evaluate(self, user):
    #     # 전체 아이템 인덱스 생성 (shape: [1, num_items])
    #     all_items = torch.arange(self.num_items, device=self.device).unsqueeze(0)
    #     # user는 이미 [batch_size] 형태라고 가정
    #     scores, _ = self.forward(user, all_items)
    #     # scores의 shape은 [batch_size, num_items]
    #     return scores
