import os
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(
        self, 
        data_path: str,
        dataset_name: str,
        relations: str="buy,cart,click,collect",
        sample_epoch: int=0
    ):
        """
        Args:
            - data_path: "src/data/MBGCN/Tmall/"
            - dataset_name: e.g. "Tmall"
            - relations: comma-separated relations
            - sample_epoch: index of sample-file to read
        """
        super().__init__()
        self.path = data_path
        self.name = dataset_name
        self._decode_relation(relations)
        self._load_size()
        self._create_relation_matrix()
        self._calculate_user_behavior()
        self._generate_ground_truth()
        self._generate_train_matrix()
        self._load_item_graph()
        self.cnt = sample_epoch
        self._read_train_data(self.cnt)
        
    def _decode_relation(self, relations_str: str):
        # e.g. "buy,cart,click,collect" => ["buy", "cart", "click", "collect"]
        self.relation = relations_str.split(",")
    
    def _load_size(self):
        """
        data_size.txt => user_num, item_num
        """
        size_file = os.path.join(self.path, self.name, "data_size.txt")
        with open(size_file, 'r') as f:
            line = f.readline().strip()
            user_num_str, item_num_str = line.split()
            self.num_users = int(user_num_str)
            self.num_items = int(item_num_str)
            
    def _load_item_graph(self):
        """
        For each realation, load_item{relation}.pth -> (num_items, num_items)
        => store in self.item_graph
        => store degree as sum of row
        """
        self.item_graph = {}
        self.item_graph_degree = {}
        for r in self.relation:
            # item_buy.pth, item_cart.pth, item_click.pth, item_collect.pth
            pth_path = os.path.join(self.path, self.name, 'item', 'item_' + r + '.pth')
            if not os.path.exists(pth_path):
                self.item_graph[r] = torch.zeros(self.num_items, self.num_items, dtype=torch.int32)
                self.item_graph_degree[r] = torch.zeros(self.num_items, 1, dtype=torch.float32)
                continue
            t = torch.load(pth_path, weights_only=True) # shape=(num_items, num_items), dtype=inr32
            self.item_graph[r] = t
            deg = t.sum(dim=1).float().unsqueeze(-1)
            self.item_graph_degree[r] = deg

    def _create_relation_matrix(self):
        """
        For each relation e.g. buy.txt => parse lines => build sparse
        => store in self.relation_dict[relation] = sparse_coo_tensor
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
                # empty
                self.relation_dict[r] = torch.sparse_coo_tensor(
                    torch.zeros((2,0),dtype=torch.long),
                    torch.zeros((0,),dtype=torch.float),
                    size=(self.num_users,self.num_items)
                )
                continue
            index_tensor = torch.LongTensor(index)  # shape=(N,2)
            lens = index_tensor.size(0)
            ones = torch.ones(lens, dtype=torch.float)
            # shape => ( user, item ), values => 1
            # we want shape=(user_num, item_num)
            sp_tensor = torch.sparse_coo_tensor(
                index_tensor.t(), # (2,N)
                ones,
                size=(self.num_users, self.num_items)
            )
            self.relation_dict[r] = sp_tensor.coalesce()
        # for b in self.relation:
        #     mat = self.relation_dict[b]
        #     print(f"Behavior '{b}': nonzero count = {mat._nnz()}")
        # for b in self.relation:
        #     I = self.relation_dict[b]
        #     print(f"Behavior '{b}' ii_mats: mean={I.mean().item():.4f}, std={I.std().item():.4f}")
    
    def _calculate_user_behavior(self):
        """
        For each user, how many items in each relation => user_behavior_degree
        For each item, how many users in each relation => item_behavior_degree
        => stored in shape=(user_num, #relation), (item_num, #relation)
        => uses to_dense() => watch out for memory if large
        """
        user_behavior = None
        item_behavior = None
        for i, r in enumerate(self.relation):
            sp_r = self.relation_dict[r] # shape=(num_users, num_items)
            dense_r = sp_r.to_dense()
            user_sum = dense_r.sum(dim=1, keepdim=True) # (num_users, 1)
            item_sum = dense_r.t().sum(dim=1, keepdim=True) # (num_items, 1)
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
        Use train.txt => ground_truth matrix => CSR
        => self.ground_truth => sp.csr_matrix(...)
        => self.checkins => array of (user, item)
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
        # shape=(num_users, num_items)
        self.ground_truth = sp.csr_matrix(
            (values, (row_data, col_data)),
            shape=(self.num_users, self.num_items)
        )
        self.checkins = np.column_stack([row_data, col_data])
    
    def _generate_train_matrix(self):
        """
        A big matrix for GCN => sum of all training edges?
        Actually old code uses only train.txt => sp matrix
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
        index_tensor = torch.LongTensor(index) # (N,2)
        lens = index_tensor.size(0)
        ones = torch.ones(lens, dtype=torch.float)
        sp_tensor = torch.sparse_coo_tensor(
            index_tensor.t(),
            ones,
            size=(self.num_users, self.num_items)
        )
        self.train_matrix = sp_tensor.coalesce()
        
    def _read_train_data(self, i):
        """
        read sample_file/sampe_{i}.txt => user, pos_item, neg_item
        store in self.train_tmp => shape=(N, 3)
        """
        sample_dir = os.path.join(self.path, self.name, 'sample_file')
        sample_file = os.path.join(sample_dir, f'sample_{i}.txt')
        if not os.path.exists(sample_file):
            print(f"[WARNING] {sample_file} not found => maybe no negative sampling? empty.")
            self.train_tmp = torch.zeros((0,3), dtype=torch.long)
            return
        tmp_array = []
        with open(sample_file, 'r') as f:
            lines = f.readlines()
            for row in lines:
                user_str, pid_str, nid_str = row.strip().split()
                user, pid, nid = int(user_str), int(pid_str), int(nid_str)
                tmp_array.append([user, pid, nid])
        if len(tmp_array)==0:
            self.train_tmp = torch.zeros((0,3), dtype=torch.long)
        else:
            self.train_tmp = torch.LongTensor(tmp_array)
        print(f"[INFO] Read Epoch {i} Train Data => shape={self.train_tmp.shape}")

    def newit(self):
        """
        move to next sample_x.txt
        """
        self.cnts += 1
        self._read_train_data(self.cnt)
    
    def __getitem__(self, idx):
        """
        => return user, (pos_item, neg_item)
        train_tmp[idx, :] => (user, pos, neg)
        user => shape=(1,)
        items => shape=(2,) => (pos,neg)
        """
        row = self.train_tmp[idx]   # shape=(3,)
        user = row[0].unsqueeze(-1) # shape=(1,)
        pin = row[1:]               # shape=(2,)
        return user, pin
    
    def __len__(self):
        """
        => len(self.checkins)
        (or len of train_tmp if we want #pos?)
        Old code does => len(self.checkins)
        but checkins is #train positives from train.txt
        """
        return len(self.checkins)
    
class TestDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_name: str, 
        trainset: TrainDataset, 
        task="test"
    ):
        """
        uses trainset.ground_truth => (num_users, num_items)
        self.ground_truth => (num_users, num_items) for valid or test
        """
        super().__init__()
        self.path = data_path
        self.name = dataset_name
        self.train_mask = trainset.ground_truth
        self.num_users, self.num_items = trainset.num_users, trainset.num_items
        self.task = task
        self._read_testset()
        
    def _read_testset(self):
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
        row_gt = self.ground_truth.getrow(idx).toarray().squeeze() # (num_items,)
        row_tr = self.train_mask.getrow(idx).toarray().squeeze().astype(np.float32)   # (num_items,)
        # 디버깅: 각 row의 shape과 최댓값/최솟값을 출력해볼 수 있음
        # print(f"DEBUG: __getitem__ idx {idx}: row_gt shape: {row_gt.shape}, min: {row_gt.min()}, max: {row_gt.max()}")
        # print(f"DEBUG: __getitem__ idx {idx}: row_tr shape: {row_tr.shape}, min: {row_tr.min()}, max: {row_tr.max()}")
        
        return idx, torch.from_numpy(row_gt), torch.from_numpy(row_tr)
    
    def __len__(self):
        return self.num_users