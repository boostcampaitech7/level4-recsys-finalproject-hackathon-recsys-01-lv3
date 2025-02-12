import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    Dataset for training interactions.

    Args:
        data_path (str): Path to the dataset directory.
        dataset_name (str): Name of the dataset (e.g. "Tmall").
        relations (str, optional): Comma-separated string of relations. Defaults to "buy,cart,click,collect".
        neg_sample_size (int, optional): Number of negative samples per positive sample. Defaults to 1.
        debug_sample (bool, optional): Whether to print debug information. Defaults to False.
    """
    def __init__(
        self, 
        data_path: str,
        dataset_name: str,
        relations: str="buy,cart,click,collect",
        neg_sample_size: int = 1,   
        debug_sample: bool = False  
    ):
        super().__init__()
        self.path = data_path
        self.name = dataset_name
        self.neg_sample_size = neg_sample_size
        self.debug_sample = debug_sample
        self._decode_relation(relations)
        self._load_size()
        self._create_relation_matrix()
        self._calculate_user_behavior()
        self._generate_ground_truth()
        self._generate_train_matrix()
        self._load_item_graph()
        self._build_user_positive() 
        self._build_item_sampling_distribution()
        
    def _decode_relation(self, relations_str: str):
        """
        Decode the relation string into a list.

        Args:
            relations_str (str): Comma-separated relations.
        """
        self.relation = relations_str.split(",")
    
    def _load_size(self):
        """
        Load dataset size from 'data_size.txt'. Expected file format: "user_num item_num".
        """
        size_file = os.path.join(self.path, self.name, "data_size.txt")
        with open(size_file, 'r') as f:
            line = f.readline().strip()
            user_num_str, item_num_str = line.split()
            self.num_users = int(user_num_str)
            self.num_items = int(item_num_str)
            
    def _load_item_graph(self):
        """
        For each relation, load the item graph from 'item/item_{relation}.pth'
        and compute the row-sum (degree) for each item.
        """
        self.item_graph = {}
        self.item_graph_degree = {}
        for r in self.relation:
            pth_path = os.path.join(self.path, self.name, 'item', 'item_' + r + '.pth')
            if not os.path.exists(pth_path):
                self.item_graph[r] = torch.zeros(self.num_items, self.num_items, dtype=torch.int32)
                self.item_graph_degree[r] = torch.zeros(self.num_items, 1, dtype=torch.float32)
                continue
            t = torch.load(pth_path, weights_only=True) 
            self.item_graph[r] = t
            deg = t.sum(dim=1).float().unsqueeze(-1)
            self.item_graph_degree[r] = deg

    def _create_relation_matrix(self):
        """
        For each relation (e.g. from 'buy.txt'), parse the file and build a sparse COO tensor.
        The tensor is stored in self.relation_dict.
        """
        self.relation_dict = {}
        for r in self.relation:
            txt_file = os.path.join(self.path, self.name, r+'.txt')
            if not os.path.exists(txt_file):
                print(f"[WARNING] {txt_file} not found => create empty. ")
                mat_empty = torch.sparse_coo_tensor(
                    torch.zeros((2,0),dtype=torch.long),
                    torch.zeros((0,),dtype=torch.float),
                    size=(self.num_users,self.num_items)
                )
                self.relation_dict[r] = mat_empty
                continue
            index = []
            with open(txt_file,'r') as f:
                lines = f.readlines()
                for row in lines:
                    user_str, item_str = row.strip().split()
                    user, item = int(user_str), int(item_str)
                    index.append([user,item])
            if len(index)==0:
                self.relation_dict[r] = torch.sparse_coo_tensor(
                    torch.zeros((2,0),dtype=torch.long),
                    torch.zeros((0,),dtype=torch.float),
                    size=(self.num_users,self.num_items)
                )
                continue
            index_tensor = torch.LongTensor(index)  
            lens = index_tensor.size(0)
            ones = torch.ones(lens, dtype=torch.float)
            sp_tensor = torch.sparse_coo_tensor(
                index_tensor.t(), # (2,N)
                ones,
                size=(self.num_users, self.num_items)
            )
            self.relation_dict[r] = sp_tensor.coalesce()
       
    def _calculate_user_behavior(self):
        """
        Calculate the behavior degree for users and items.
        For each relation, compute the number of interactions per user and per item.
        """
        user_behavior = None
        item_behavior = None
        for i, r in enumerate(self.relation):
            sp_r = self.relation_dict[r] 
            dense_r = sp_r.to_dense()
            user_sum = dense_r.sum(dim=1, keepdim=True) 
            item_sum = dense_r.t().sum(dim=1, keepdim=True)
            if i==0:
                user_behavior = user_sum
                item_behavior = item_sum
            else:
                user_behavior = torch.cat([user_behavior, user_sum], dim=1)
                item_behavior = torch.cat((item_behavior,item_sum), dim=1)
        self.user_behavior_degree = user_behavior
        self.item_behavior_degree = item_behavior
        
    def _generate_ground_truth(self):
        """
        Generate the ground truth interaction matrix from 'train.txt' using CSR format.
        Also stores the check-ins as an array of (user, item) pairs.
        """
        txt_file = os.path.join(self.path, self.name, "train.txt")
        row_data = []
        col_data = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for row in lines:
                u_str, i_str = row.strip().split()
                u, i = int(u_str), int(i_str)
                row_data.append(u)
                col_data.append(i)
        row_data = np.array(row_data, dtype=np.int32)
        col_data = np.array(col_data, dtype=np.int32)
        values = np.ones(len(row_data), dtype=float)
        self.ground_truth = sp.csr_matrix(
            (values, (row_data, col_data)),
            shape=(self.num_users, self.num_items)
        )
        self.checkins = np.column_stack([row_data, col_data])
    
    def _generate_train_matrix(self):
        """
        Generate a training matrix (sparse COO tensor) from 'train.txt'.
        """
        txt_file = os.path.join(self.path, self.name, "train.txt")
        index = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for row in lines:
                u_str, i_str = row.strip().split()
                u, i = int(u_str), int(i_str)
                index.append([u, i])
        if len(index) == 0:
            self.train_matrix = torch.sparse_coo_tensor(
                torch.zeros((2,0), dtype=torch.long),
                torch.zeors((0,), dtype=torch.float),
                size=(self.num_users, self.num_items)
            )
            return
        index_tensor = torch.LongTensor(index)
        lens = index_tensor.size(0)
        ones = torch.ones(lens, dtype=torch.float)
        sp_tensor = torch.sparse_coo_tensor(
            index_tensor.t(),
            ones,
            size=(self.num_users, self.num_items)
        )
        self.train_matrix = sp_tensor.coalesce()
    
    def _build_user_positive(self):
        """
        Build a dictionary mapping each user to the set of items they have interacted with.
        """
        self.user_positive = {}
        gt = self.ground_truth
        for user in range(self.num_users):
            start = gt.indptr[user]
            end = gt.indptr[user+1]
            self.user_positive[user] = set(gt.indices[start:end])
        if self.debug_sample:
            print(f"DEBUG: Built user_positive mapping for {len(self.user_positive)} users.")
        
    def _build_item_sampling_distribution(self):
        """
        Build the negative sampling probability distribution based on item frequency.
        Uses frequency^0.75 for reweighting.
        """
        freq = self.ground_truth.getnnz(axis=0).astype(np.float32) 
        freq_pow = np.power(freq, 0.75)
        total = freq_pow.sum()
        self.neg_prob = freq_pow / total 
        if self.debug_sample:
            print(
                f"[DEBUG] Negative sampling probability: min={self.neg_prob.min():.6f}, "
                f"max={self.neg_prob.max():.6f}"
            )

    def _sample_negative(self, user):
        """
        Sample a negative item for the given user using weighted sampling.
        
        Args:
            user (int): User index.
        
        Returns:
            int: Negative item index.
        """
        max_trials = 1000
        trial = 0
        while trial < max_trials:
            neg_item = int(np.random.choice(self.num_items, p=self.neg_prob))
            if neg_item not in self.user_positive.get(user, set()):
                return neg_item
            trial += 1
        print(f"DEBUG: Failed to sample negative for user {user} after {max_trials} trials. Returning last candidate {neg_item}")
        return neg_item

    
    def __getitem__(self, idx):
        """
        Get the training sample for the given index.
        
        Returns:
            tuple: (torch.Tensor of user index, torch.Tensor of [positive item, negative items])
        """
        user, pos_item = self.checkins[idx]
        neg_items = []
        for _ in range(self.neg_sample_size):
            neg = self._sample_negative(user)
            neg_items.append(neg)
        return torch.tensor([user]), torch.tensor( [pos_item] + neg_items )
    
    def __len__(self):
        """
        Return the number of training samples.
        
        Returns:
            int: Number of samples.
        """
        return len(self.checkins)
    

class TestDataset(Dataset):
    """
    Dataset for testing or validation.

    Args:
        data_path (str): Path to the dataset directory.
        dataset_name (str): Name of the dataset.
        trainset (TrainDataset): Instance of the training dataset (to access ground truth).
        task (str, optional): Task type ("test" or "validation"). Defaults to "test".
    """
    
    def __init__(
        self,
        data_path: str,
        dataset_name: str, 
        trainset: TrainDataset, 
        task="test"
    ):
        super().__init__()
        self.path = data_path
        self.name = dataset_name
        self.train_mask = trainset.ground_truth
        self.num_users, self.num_items = trainset.num_users, trainset.num_items
        self.task = task
        self._read_testset()
        
    def _read_testset(self):
        """
        Read the test (or validation) file and build the ground truth CSR matrix.
        """
        txt_file = os.path.join(self.path, self.name, f"{self.task}.txt")
        row=[]
        col=[]
        if not os.path.exists(txt_file):
            print(f"[WARNING] {txt_file} not found => create empty {self.task} set.")
        else:
            with open(txt_file,'r') as f:
                lines=f.readlines()
                for line in lines:
                    u_str,i_str=line.strip().split()
                    u=int(u_str); i=int(i_str)
                    row.append(u); col.append(i)
        row_np = np.array(row,dtype=np.int32)
        col_np = np.array(col,dtype=np.int32)
        vals = np.ones(len(row),dtype=float)
        self.checkins = np.column_stack([row_np,col_np])
        self.ground_truth = sp.csr_matrix(
            (vals,(row_np,col_np)),
            shape=(self.num_users,self.num_items)
        )
    
    def __getitem__(self, idx):
        """
        Get the test sample for a given user index.
        
        Returns:
            tuple: (user index, ground truth tensor, train mask tensor)
        """
        row_gt = self.ground_truth.getrow(idx).toarray().squeeze()
        row_tr = self.train_mask.getrow(idx).toarray().squeeze().astype(np.float32)
        return idx, torch.from_numpy(row_gt), torch.from_numpy(row_tr)
    
    def __len__(self):
        """
        Return the number of users.
        
        Returns:
            int: Number of users.
        """
        return self.num_users
    
class TotalTrainDataset(Dataset):
    """
    Dataset for training on the complete set of interactions.

    Uses "total.txt" for overall interactions and r+"_total.txt" for each behavior,
    along with item graphs from "item/item_{r}_total.pth".

    Args:
        data_path (str): Path to the dataset directory.
        dataset_name (str): Name of the dataset.
        relations (str, optional): Comma-separated relations. Defaults to "buy,cart,click,collect".
        neg_sample_size (int, optional): Number of negative samples per positive sample. Defaults to 1.
        debug_sample (bool, optional): Whether to print debug information. Defaults to False.
    """
    def __init__(
        self, 
        data_path: str,
        dataset_name: str,
        relations: str="buy,cart,click,collect",
        neg_sample_size: int = 1,
        debug_sample: bool = False
    ):
        super().__init__()
        self.path = data_path
        self.name = dataset_name
        self.neg_sample_size = neg_sample_size
        self.debug_sample = debug_sample
        self._decode_relation(relations)
        self._load_size()
        self._create_relation_matrix()
        self._calculate_user_behavior()
        self._generate_ground_truth()
        self._generate_train_matrix()
        self._load_item_graph()
        self._build_user_positive()
        self._build_item_sampling_distribution()

    def _decode_relation(self, relations_str: str):
        """
        Decode the relation string into a list.

        Args:
            relations_str (str): Comma-separated relations.
        """
        self.relation = relations_str.split(",")

    def _load_size(self):
        """
        Load dataset size from 'data_size_total.txt'.
        """
        size_file = os.path.join(self.path, self.name, "data_size_total.txt")
        with open(size_file, 'r') as f:
            line = f.readline().strip()
            user_num_str, item_num_str = line.split()
            self.num_users = int(user_num_str)
            self.num_items = int(item_num_str)

    def _create_relation_matrix(self):
        """
        For each relation, read r+"_total.txt" and build a sparse COO tensor.
        The tensors are stored in self.relation_dict.
        """
        self.relation_dict = {}
        for r in self.relation:
            txt_file = os.path.join(self.path, self.name, r + "_total.txt")
            if not os.path.exists(txt_file):
                print(f"[WARNING] {txt_file} not found => create empty matrix.")
                mat_empty = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0,), dtype=torch.float),
                    size=(self.num_users, self.num_items)
                )
                self.relation_dict[r] = mat_empty
                continue
            index = []
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for row in lines:
                    parts = row.strip().split()
                    if len(parts) < 2:
                        continue
                    user_str, item_str = parts[:2]
                    user, item = int(user_str), int(item_str)
                    index.append([user, item])
            if len(index) == 0:
                self.relation_dict[r] = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0,), dtype=torch.float),
                    size=(self.num_users, self.num_items)
                )
                continue
            index_tensor = torch.LongTensor(index)
            lens = index_tensor.size(0)
            ones = torch.ones(lens, dtype=torch.float)
            sp_tensor = torch.sparse_coo_tensor(
                index_tensor.t(),
                ones,
                size=(self.num_users, self.num_items)
            )
            self.relation_dict[r] = sp_tensor.coalesce()

    def _calculate_user_behavior(self):
        """
        Calculate the behavior degree for users and items.
        The result is stored in self.user_behavior_degree and self.item_behavior_degree.
        """
        user_behavior = None
        item_behavior = None
        for i, r in enumerate(self.relation):
            sp_r = self.relation_dict[r]
            dense_r = sp_r.to_dense()
            user_sum = dense_r.sum(dim=1, keepdim=True)
            item_sum = dense_r.t().sum(dim=1, keepdim=True)
            if i == 0:
                user_behavior = user_sum
                item_behavior = item_sum
            else:
                user_behavior = torch.cat([user_behavior, user_sum], dim=1)
                item_behavior = torch.cat([item_behavior, item_sum], dim=1)
        self.user_behavior_degree = user_behavior
        self.item_behavior_degree = item_behavior

    def _generate_ground_truth(self):
        """
        Generate the ground truth interaction matrix from 'total.txt'
        using CSR format and store check-ins.
        """
        txt_file = os.path.join(self.path, self.name, "total.txt")
        row_data = []
        col_data = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for row in lines:
                parts = row.strip().split()
                if len(parts) < 2:
                    continue
                u_str, i_str = parts[:2]
                user, item = int(u_str), int(i_str)
                row_data.append(user)
                col_data.append(item)
        row_data = np.array(row_data, dtype=np.int32)
        col_data = np.array(col_data, dtype=np.int32)
        values = np.ones(len(row_data), dtype=float)
        self.ground_truth = sp.csr_matrix(
            (values, (row_data, col_data)),
            shape=(self.num_users, self.num_items)
        )
        self.checkins = np.column_stack([row_data, col_data])

    def _generate_train_matrix(self):
        """
        Generate the training matrix from 'total.txt' as a sparse COO tensor.
        """
        txt_file = os.path.join(self.path, self.name, "total.txt")
        index = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for row in lines:
                parts = row.strip().split()
                if len(parts) < 2:
                    continue
                u_str, i_str = parts[:2]
                user, item = int(u_str), int(i_str)
                index.append([user, item])
        if len(index) == 0:
            self.train_matrix = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float),
                size=(self.num_users, self.num_items)
            )
            return
        index_tensor = torch.LongTensor(index)
        lens = index_tensor.size(0)
        ones = torch.ones(lens, dtype=torch.float)
        sp_tensor = torch.sparse_coo_tensor(
            index_tensor.t(),
            ones,
            size=(self.num_users, self.num_items)
        )
        self.train_matrix = sp_tensor.coalesce()

    def _load_item_graph(self):
        """
        For each relation, load the item graph from 'item/item_{r}_total.pth'
        and compute the degree of each item.
        """
        self.item_graph = {}
        self.item_graph_degree = {}
        for r in self.relation:
            pth_path = os.path.join(self.path, self.name, "item", f"item_{r}_total.pth")
            if not os.path.exists(pth_path):
                print(f"[WARNING] {pth_path} not found => create zero matrix.")
                self.item_graph[r] = torch.zeros(self.num_items, self.num_items, dtype=torch.int32)
                self.item_graph_degree[r] = torch.zeros(self.num_items, 1, dtype=torch.float32)
                continue
            t = torch.load(pth_path, map_location="cpu", weights_only=True)
            self.item_graph[r] = t
            deg = t.sum(dim=1).float().unsqueeze(-1)
            self.item_graph_degree[r] = deg

    def _build_user_positive(self):
        """
        Build a dictionary mapping each user to the set of items they have interacted with.
        """
        self.user_positive = {}
        gt = self.ground_truth
        for user in range(self.num_users):
            start = gt.indptr[user]
            end = gt.indptr[user+1]
            self.user_positive[user] = set(gt.indices[start:end])
        if self.debug_sample:
            print(f"[DEBUG] Built user_positive for {len(self.user_positive)} users.")

    def _build_item_sampling_distribution(self):
        """
        Build the negative sampling probability distribution based on item frequencies.
        Uses frequency^0.75 for reweighting.
        """
        freq = self.ground_truth.getnnz(axis=0).astype(np.float32)
        freq_pow = np.power(freq, 0.75)
        total = freq_pow.sum()
        self.neg_prob = freq_pow / total
        if self.debug_sample:
            print(f"[DEBUG] Negative sampling: min={self.neg_prob.min():.6f}, max={self.neg_prob.max():.6f}")

    def _sample_negative(self, user):
        """
        Sample a negative item for the given user, ensuring it is not in the user's positive set.

        Args:
            user (int): User index.

        Returns:
            int: Negative item index.
        """
        max_trials = 1000
        trial = 0
        while trial < max_trials:
            neg_item = int(np.random.choice(self.num_items, p=self.neg_prob))
            if neg_item not in self.user_positive.get(user, set()):
                return neg_item
            trial += 1
        print(f"[DEBUG] Negative sampling failed for user {user} after {max_trials} trials. Returning {neg_item}")
        return neg_item

    def __getitem__(self, idx):
        """
        Get the training sample at the given index.

        Returns:
            tuple: (torch.Tensor containing user index, torch.Tensor containing [positive item, negative items])
        """
        user, pos_item = self.checkins[idx]
        neg_items = []
        for _ in range(self.neg_sample_size):
            neg = self._sample_negative(user)
            neg_items.append(neg)
        return torch.tensor([user]), torch.tensor([pos_item] + neg_items)

    def __len__(self):
        """
        Return the number of training samples.

        Returns:
            int: Number of samples.
        """
        return len(self.checkins)