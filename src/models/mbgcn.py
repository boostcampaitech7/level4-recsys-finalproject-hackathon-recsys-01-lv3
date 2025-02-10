import os
import argparse

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelBase(nn.Module):
    """
    Base model class for collaborative filtering models.

    Args:
        args (argparse.Namespace): Configuration parameters.
        trainset: Training dataset object.
        device (torch.device): Device on which to run the model.
    """
    
    def __init__(self, args: argparse.Namespace, trainset, device: torch.device):
        super().__init__()
        self.embed_size = args.embedding_size
        self.L2_norm = args.L2_norm
        self.device = device
        self.num_users = trainset.num_users
        self.num_items = trainset.num_items

        if args.create_embeddings:
            self.item_embedding = nn.Parameter(
                torch.empty(self.num_items, self.embed_size, device=self.device)
            )
            nn.init.xavier_normal_(self.item_embedding)
            self.user_embedding = nn.Parameter(
                torch.empty(self.num_users, self.embed_size, device=self.device)
            )
            nn.init.xavier_normal_(self.user_embedding)
        else:
            load_path = os.path.join(args.pretrain_path, 'model.pkl')
            load_data = torch.load(load_path, map_location='cpu')
            if not args.pretrain_frozen:
                self.item_embedding = nn.Parameter(
                    F.normalize(load_data['item_embedding']).to(self.device)
                )
                self.user_embedding = nn.Parameter(
                    F.normalize(load_data['user_embedding']).to(self.device)
                )
            else:
                self.item_embedding = F.normalize(
                    load_data['item_embedding']).to(self.device)
                self.user_embedding = F.normalize(
                    load_data['user_embedding']).to(self.device)

    def propagate(self, *args, **kwargs):
        """
        Compute embeddings for users and items for prediction.

        Returns:
            tuple: (user_embedding, item_embedding)
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """
        Compute predicted scores from the provided embeddings.

        Returns:
            torch.Tensor: Score tensor.
        """
        raise NotImplementedError

    def regularize(self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the L2 regularization term for the given embeddings.

        Args:
            user_embeddings (torch.Tensor): User embeddings.
            item_embeddings (torch.Tensor): Item embeddings.

        Returns:
            torch.Tensor: Regularization loss.
        """
        return self.L2_norm * ((user_embeddings ** 2).sum() + (item_embeddings ** 2).sum())

    def forward(self, users, items):
        """
        Compute predictions and regularization loss for given user and item indices.

        Args:
            users: Indices of users.
            items: Indices of items.

        Returns:
            tuple: (predicted scores, L2 loss)
        """
        users_feature, item_feature = self.propagate()
        item_embeddings = item_feature[items]
        user_embeddings = users_feature[users].unsqueeze(1).expand(-1, items.shape[1], -1)
        pred = self.predict(user_embeddings, item_embeddings)
        L2_loss = self.regularize(user_embeddings, item_embeddings)
        return pred, L2_loss

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on the full set of items.

        Returns:
            torch.Tensor: Score tensor for evaluation.
        """
        raise NotImplementedError

class MF(ModelBase):
    """
    Matrix Factorization (MF) model.
    
    Args:
            args (argparse.Namespace): Configuration parameters.
            trainset: Training dataset object.
            device (torch.device): Device on which to run the model.
    """
    def __init__(self, args: argparse.Namespace, trainset, device: torch.device):
        super().__init__(args, trainset, device)

    def propagate(self, task: str = 'train'):
        """
        Return raw user and item embeddings.

        Args:
            task (str, optional): Task identifier. Defaults to 'train'.

        Returns:
            tuple: (user_embedding, item_embedding)
        """
        return self.user_embedding, self.item_embedding

    def predict(self, user_embedding, item_embedding):
        """
        Compute dot-product between user and item embeddings.

        Args:
            user_embedding (torch.Tensor): User embeddings.
            item_embedding (torch.Tensor): Item embeddings.

        Returns:
            torch.Tensor: Dot-product scores.
        """
        return torch.sum(user_embedding * item_embedding, dim=2)

    def evaluate(self, users):
        """
        Compute scores for given users against all items.

        Args:
            users: Indices of users.

        Returns:
            torch.Tensor: Score matrix.
        """
        users_feature, item_feature = self.propagate()
        user_feature = users_feature[users]
        scores = torch.mm(user_feature, item_feature.t())
        return scores

class MBGCN(ModelBase):
    """
    MBGCN model with graph-based message passing.
    
    Args:
            args (argparse.Namespace): Configuration parameters.
            trainset: Training dataset object with graph information.
            device (torch.device): Device on which to run the model.
    """
    def __init__(self, args: argparse.Namespace, trainset, device: torch.device):
        super().__init__(args, trainset, device)
        self.relation_dict = trainset.relation_dict  
        self.mgnn_weight = args.mgnn_weight  
        self.item_graph = trainset.item_graph  
        self.train_matrix = trainset.train_matrix.to(self.device)
        self.relation = trainset.relation
        self.lamb = args.lamb
        self.item_graph_degree = trainset.item_graph_degree
        self.user_behavior_degree = trainset.user_behavior_degree.to(self.device)

        self.message_drop = nn.Dropout(p=args.message_dropout)
        self.train_node_drop = nn.Dropout(p=args.node_dropout)
        self.node_drop = nn.ModuleList(
            [nn.Dropout(p=args.node_dropout) for _ in self.relation_dict]
        )

        self._to_gpu()
        self._param_init()

    def _to_gpu(self):
        """
        Move all graph-related tensors to the specified device.
        """
        for key in self.relation_dict:
            self.relation_dict[key] = self.relation_dict[key].to(self.device)
        for key in self.item_graph:
            self.item_graph[key] = self.item_graph[key].to(self.device)
        for key in self.item_graph_degree:
            self.item_graph_degree[key] = self.item_graph_degree[key].to(self.device)

    def _decode_weight(self):
        """
        Decode and normalize the message passing weights.
        """
        weight = torch.softmax(self.mgnn_weight, dim=0).unsqueeze(-1)
        total_weight = torch.mm(self.user_behavior_degree, weight)
        self.user_behavior_weight = self.user_behavior_degree.float() / (total_weight + 1e-8)

    def _param_init(self):
        """
        Initialize model parameters for message propagation.
        """
        self.mgnn_weight = nn.Parameter(
            torch.tensor(self.mgnn_weight, dtype=torch.float32, device=self.device)
        )
        self.item_behavior_W = nn.ParameterList([
            nn.Parameter(
                torch.empty(self.embed_size * 2, self.embed_size * 2, device=self.device)
            )
            for _ in self.mgnn_weight
        ])
        for param in self.item_behavior_W:
            nn.init.xavier_normal_(param)
        self.item_propagate_W = nn.ParameterList([
            nn.Parameter(
                torch.empty(self.embed_size, self.embed_size, device=self.device)
            )
            for _ in self.mgnn_weight
        ])
        for param in self.item_propagate_W:
            nn.init.xavier_normal_(param)
        self.W = nn.Parameter(
            torch.empty(self.embed_size, self.embed_size, device=self.device)
        )
        nn.init.xavier_normal_(self.W)

    def forward(self, user, item):
        """
        Compute predictions and L2 regularization loss for a batch of users and items.

        Args:
            user: Indices of users.
            item: Indices of items.

        Returns:
            tuple: (scores, L2_loss)
        """
        indices = self.train_matrix._indices()
        values = self.train_matrix._values()
        values = self.train_node_drop(values)
        train_matrix = torch.sparse_coo_tensor(
            indices,
            values,
            size=self.train_matrix.shape,
            dtype=torch.float32,
            device=self.device,
        )

        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behavior_degree, weight)
        user_behavior_weight = (
            self.user_behavior_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)
        )

        for i, key in enumerate(self.relation_dict):
            rel = self.relation_dict[key]
            indices = rel._indices()
            values = rel._values()
            values = self.node_drop[i](values)
            tmp_relation_matrix = torch.sparse_coo_tensor(
                indices, values, size=rel.shape, device=self.device
            )

            tmp_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) 
                / (self.item_graph_degree[key] + 1e-8),
                self.item_propagate_W[i],
            )
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)
            tmp_item_embedding = tmp_item_propagation[item]

            tmp_user_neighbour = torch.mm(
                tmp_relation_matrix, self.item_embedding
            ) / (self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8)
            tmp_user_item_neighbour_p = torch.mm(
                tmp_relation_matrix, tmp_item_propagation
            ) / (self.user_behavior_degree[:, i].unsqueeze(-1) + 1e-8)

            if i == 0:
                user_feature = user_behavior_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehavior_item_projection = torch.mm(
                    tmp_user_item_neighbour_p, self.item_behavior_W[i]
                )
                tuser_behavior_projection = tbehavior_item_projection[user].unsqueeze(1).expand(
                    -1, item.shape[1], -1
                )
                score2 = torch.sum(tuser_behavior_projection * tmp_item_embedding, dim=2)
            else:
                user_feature += user_behavior_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehavior_item_projection = torch.mm(
                    tmp_user_item_neighbour_p, self.item_behavior_W[i]
                )
                tuser_behavior_projection = tbehavior_item_projection[user].unsqueeze(1).expand(
                    -1, item.shape[1], -1
                )
                score2 += torch.sum(tuser_behavior_projection * tmp_item_embedding, dim=2)

        score2 = score2 / len(self.relation_dict)

        item_feature = torch.mm(train_matrix.t(), self.user_embedding)
        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)
        user_feature = torch.cat((self.user_embedding, user_feature), dim=1)
        item_feature = torch.cat((self.item_embedding, item_feature), dim=1)


        user_feature = self.message_drop(user_feature)
        item_feature = self.message_drop(item_feature)

        tmp_user_feature = user_feature[user].unsqueeze(1).expand(-1, item.shape[1], -1)
        tmp_item_feature = item_feature[item]
        score1 = torch.sum(tmp_user_feature * tmp_item_feature, dim=2)
        scores = score1 + self.lamb * score2

        l2_loss = self.regularize(tmp_user_feature, tmp_item_feature)
        return scores, l2_loss

    def evaluate(self, user):
        """
        Evaluate the model for given users across all items.

        Args:
            user: Indices of users.

        Returns:
            torch.Tensor: Score matrix for evaluation.
        """
        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behavior_degree, weight)
        user_behavior_weight = (
            self.user_behavior_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)
        )

        for i, key in enumerate(self.relation_dict):
            tmp_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding)
                / (self.item_graph_degree[key] + 1e-8),
                self.item_propagate_W[i],
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

