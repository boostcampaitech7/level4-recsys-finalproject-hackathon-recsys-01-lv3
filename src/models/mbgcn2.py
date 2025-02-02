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
        user_embeddings = users_feature[users].unsqueeze(1).expand(-1, items.shape[1], -1)
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

    def evaluate(self, users):
        users_feature, item_feature = self.propagate()
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
        self.W = nn.Parameter(torch.empty(self.embed_size, self.embed_size, device=self.device))
        nn.init.xavier_normal_(self.W)

    def forward(self, user, item):
        # train_matrix: node dropout 적용
        indices = self.train_matrix._indices()
        values = self.train_matrix._values()
        values = self.train_node_drop(values)
        train_matrix = torch.sparse_coo_tensor(indices, values, size=self.train_matrix.shape, dtype=torch.float32, device=self.device)

        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behavior_degree, weight)
        user_behavior_weight = self.user_behavior_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)

        for i, key in enumerate(self.relation_dict):
            # relation_dict에 node dropout 적용
            rel = self.relation_dict[key]
            indices = rel._indices()
            values = rel._values()
            values = self.node_drop[i](values)
            tmp_relation_matrix = torch.sparse_coo_tensor(indices, values, size=rel.shape, device=self.device)

            # 아이템 전파: 아이템 임베딩과 전파 행렬 곱 연산
            tmp_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8),
                self.item_propagate_W[i]
            )
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)
            tmp_item_embedding = tmp_item_propagation[item]

            tmp_user_neighbour = torch.mm(tmp_relation_matrix, self.item_embedding) / (
                self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8
            )
            tmp_user_item_neighbour_p = torch.mm(tmp_relation_matrix, tmp_item_propagation) / (
                self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8
            )

            if i == 0:
                user_feature = user_behavior_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehavior_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behavior_W[i])
                tuser_behavior_projection = tbehavior_item_projection[user].unsqueeze(1).expand(-1, item.shape[1], -1)
                score2 = torch.sum(tuser_behavior_projection * tmp_item_embedding, dim=2)
            else:
                user_feature += user_behavior_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehavior_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behavior_W[i])
                tuser_behavior_projection = tbehavior_item_projection[user].unsqueeze(1).expand(-1, item.shape[1], -1)
                score2 += torch.sum(tuser_behavior_projection * tmp_item_embedding, dim=2)

        # relation 개수로 나누어 평균
        score2 = score2 / len(self.relation_dict)

        # 사용자와 아이템 feature 추출
        item_feature = torch.mm(train_matrix.t(), self.user_embedding)
        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)
        user_feature = torch.cat((self.user_embedding, user_feature), dim=1)
        item_feature = torch.cat((self.item_embedding, item_feature), dim=1)

        # 메시지 드랍아웃 적용
        user_feature = self.message_drop(user_feature)
        item_feature = self.message_drop(item_feature)

        tmp_user_feature = user_feature[user].unsqueeze(1).expand(-1, item.shape[1], -1)
        tmp_item_feature = item_feature[item]
        score1 = torch.sum(tmp_user_feature * tmp_item_feature, dim=2)
        scores = score1 + self.lamb * score2

        l2_loss = self.regularize(tmp_user_feature, tmp_item_feature)
        return scores, l2_loss

    def evaluate(self, user):
        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behavior_degree, weight)
        user_behavior_weight = self.user_behavior_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)

        for i, key in enumerate(self.relation_dict):
            tmp_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8),
                self.item_propagate_W[i]
            )
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)
            tmp_user_neighbour = torch.mm(self.relation_dict[key], self.item_embedding) / (
                self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8
            )
            tmp_user_item_neighbour_p = torch.mm(self.relation_dict[key], tmp_item_propagation) / (
                self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8
            )

            if i == 0:
                user_feature = user_behavior_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehavior_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behavior_W[i])
                score2 = torch.mm(tbehavior_item_projection[user], tmp_item_propagation.t())
            else:
                user_feature += user_behavior_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehavior_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behavior_W[i])
                score2 += torch.mm(tbehavior_item_projection[user], tmp_item_propagation.t())

        score2 = score2 / len(self.relation_dict)
        item_feature = torch.mm(self.train_matrix.t(), self.user_embedding)
        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)
        user_feature = torch.cat((self.user_embedding, user_feature), dim=1)
        item_feature = torch.cat((self.item_embedding, item_feature), dim=1)

        tmp_user_feature = user_feature[user]
        score1 = torch.mm(tmp_user_feature, item_feature.t())
        scores = score1 + self.lamb * score2
        return scores


