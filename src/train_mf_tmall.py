# train_mf_tmall.py

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.mf import BPRMF
from src.data.preparation import load_parquet_data  # or load_tmall_data, etc.
from src.data.preparation import build_user_item_matrices, BPRDataset
from src.utils.metrics import Recall, NDCG

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="./pretrained_mf")

    parser.add_argument("--root_dir", type=str, default="src/data/MBGCN/Tmall", help="Tmall dataset root")
    return parser.parse_args()

def bpr_loss(pos_scores, neg_scores):
    """
    NaN 방지 => logsigmoid(diff) 전에 clamp
    """
    if len(neg_scores.shape)==2:
        diff = pos_scores.unsqueeze(1) - neg_scores
    else:
        diff = pos_scores - neg_scores

    # clamp
    diff = torch.clamp(diff, -15, 15)
    return -torch.mean(F.logsigmoid(diff))

def compute_metrics(model, loader, device, ui_mats_train, k_values=[10, 20, 40]):
    """
    전체 배치에 대한 Recall@k 및 NDCG@k 계산
    Args:
        model: 학습된 모델
        loader: DataLoader 객체 (검증 데이터)
        device: 장치 (CPU 또는 GPU)
        ui_mats_train: 학습 데이터의 사용자-아이템 매트릭스 (이미 본 아이템 제외에 필요)
        k_values: 평가할 k 값의 리스트
    Returns:
        metrics_dict: 각 k에 대한 Recall과 NDCG을 포함하는 딕셔너리
    """
    model.eval()

    # 각 k에 대한 메트릭 인스턴스 생성
    recall_metrics = {k: Recall(topk=k) for k in k_values}
    ndcg_metrics = {k: NDCG(topk=k) for k in k_values}
    max_k = max(k_values)

    with torch.no_grad():
        for batch in loader:
            user_ids = batch["user"].squeeze().to(device)          # shape=(batch_size,)
            pos_item_ids = batch["pos_item"].squeeze().to(device)  # shape=(batch_size,)

            batch_size = user_ids.size(0)

            # 모든 아이템에 대한 예측 점수 계산
            user_emb = model.user_emb[user_ids]  # (batch_size, emb_dim)
            item_emb = model.item_emb            # (num_items, emb_dim)
            scores = torch.matmul(user_emb, item_emb.t())  # (batch_size, num_items)

            # 이미 본 아이템 제외
            if ui_mats_train is not None:
                # ui_mats_train는 scipy.sparse.csr_matrix 형태
                # 각 사용자별로 이미 본 아이템의 인덱스를 가져와서 -inf로 설정
                # 이를 벡터화하여 처리
                rows = user_ids.cpu().numpy()
                for i, u in enumerate(rows):
                    items_u = ui_mats_train[u].indices
                    scores[i, items_u] = float('-inf')

            # 상위 max_k 아이템 추출
            _, topk_items = torch.topk(scores, max_k, dim=1)  # (batch_size, max_k)

            for k in k_values:
                topk = topk_items[:, :k]  # (batch_size, k)

                # Recall@k 계산
                hits = (topk == pos_item_ids.unsqueeze(1)).any(dim=1).float()  # (batch_size,)
                recall_metrics[k].update(hits)

                # NDCG@k 계산
                # 각 사용자에 대해 정답 아이템의 첫 번째 히트 위치를 찾고, NDCG 계산
                # 정답 아이템이 topk에 없으면 NDCG는 0
                relevant = (topk == pos_item_ids.unsqueeze(1))  # (batch_size, k)

                # 찾은 정답 아이템의 인덱스 (ranks)
                ranks = torch.full((batch_size,), -1, dtype=torch.float, device=device)  # 초기화

                hits_tensor = relevant.nonzero(as_tuple=False)  # (num_hits, 2)
                for hit in hits_tensor:
                    user_idx, rank = hit.tolist()
                    if ranks[user_idx] == -1:
                        ranks[user_idx] = rank + 1  # rank는 0-based

                # DCG 계산
                dcg = torch.zeros(batch_size, device=device)
                valid = ranks > 0
                dcg[valid] = 1.0 / torch.log2(ranks[valid] + 1)

                # NDCG는 DCG / IDCG = DCG / 1.0
                ndcg = dcg

                ndcg_metrics[k].update(ndcg)

    # Compute average metrics
    metrics_dict = {}
    for k in k_values:
        metrics_dict[f"Recall@{k}"] = recall_metrics[k].compute()
        metrics_dict[f"NDCG@{k}"] = ndcg_metrics[k].compute()

    return metrics_dict

def compute_bpr_loss(model, loader, device, l2_reg):
    """
    전체 배치에 대한 BPR Loss(평균) 계산 (Validation에서 EarlyStopping용)
    """
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            user_ids = batch["user"].squeeze()
            pos_item_ids = batch["pos_item"].squeeze()
            neg_item_ids = batch["neg_item"].squeeze()

            pos_scores, neg_scores = model(user_ids, pos_item_ids, neg_item_ids)
            loss_bpr = bpr_loss(pos_scores, neg_scores)

            # L2
            user_e = model.user_emb[user_ids]
            pos_e  = model.item_emb[pos_item_ids]

            if len(neg_item_ids.shape)==2:
                neg_item_ids = neg_item_ids.view(-1)
            neg_e  = model.item_emb[neg_item_ids]
            reg = l2_reg*(user_e.norm(2).pow(2)+pos_e.norm(2).pow(2)+neg_e.norm(2).pow(2))/ loader.batch_size

            loss = loss_bpr + reg
            total_loss += loss.item()
            count += 1
    return total_loss / count

def train_mf_tmall(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load dataset => train.txt, validation.txt
    #    (여기선 Polars or custom code)
    #    예시: build_user_item_matrices() -> ui_mats_train, ui_mats_valid ...
    #    or load_tmall_data() if we splitted them already
    # --- 간단히 build 2 user-item mats for buy: train & valid
    #  (실제로는 train.txt, validation.txt 를 merge해서 remap해야 하지만,
    #   여기서는 "논문에서 Tmall은 user/item id가 일관"이라고 가정)
    #   => 아래 로직은 src/data/preparation.py에 따라 조정
    #   => or we do a small function "load_tmall_train_valid"

    # 예) load train
    train_path = os.path.join(args.root_dir, "train.txt")
    valid_path = os.path.join(args.root_dir, "validation.txt")

    # user/item 수, re-map => build_user_item_matrices(???)
    # 가정: build_user_item_matrices는
    #       buy.txt가 아닌 "train.txt"만 로딩 => ui_mats["buy"]
    #       validation.txt 는 별도 build_user_item_matrices => ui_mats_valid["buy"]
    # 실제 구현은 아래 처럼:
    train_df = []
    with open(train_path,"r") as f:
        for line in f:
            u,i=line.strip().split()
            train_df.append((int(u),int(i)))
    valid_df = []
    with open(valid_path,"r") as f:
        for line in f:
            u,i=line.strip().split()
            valid_df.append((int(u),int(i)))

    # user/item max id
    max_u = max( max([r[0] for r in train_df]), max([r[0] for r in valid_df]) )
    max_i = max( max([r[1] for r in train_df]), max([r[1] for r in valid_df]) )
    num_users = max_u+1
    num_items = max_i+1

    # build CSR for train
    import scipy.sparse as sp
    row = [r[0] for r in train_df]
    col = [r[1] for r in train_df]
    data = [1]*len(row)
    ui_mat_train = sp.csr_matrix((data,(row,col)), shape=(num_users,num_items))

    row_v = [r[0] for r in valid_df]
    col_v = [r[1] for r in valid_df]
    data_v = [1]*len(row_v)
    ui_mat_valid = sp.csr_matrix((data_v,(row_v,col_v)), shape=(num_users,num_items))

    # 2) Dataset
    from src.data.preparation import BPRDataset
    train_dataset = BPRDataset(ui_mat_train, num_items, neg_size=1)
    valid_dataset = BPRDataset(ui_mat_valid, num_items, neg_size=1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train #User={num_users}, #Item={num_items}, #Pos={len(train_df)}")
    print(f"Valid #Pos={len(valid_df)}")

    # 3) init BPRMF
    from src.models.mf import BPRMF
    model = BPRMF(num_users, num_items, args.embedding_size, device=str(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_recall = -float('inf')  # Recall은 최대화
    patience_count = 0
    best_state = None

    # 4) train loop => epoch=400, early_stop=40
    for ep in range(1, args.epoch+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            user_ids = batch["user"].squeeze()
            pos_item_ids = batch["pos_item"].squeeze()
            neg_item_ids = batch["neg_item"].squeeze()

            # forward
            pos_scores, neg_scores = model(user_ids, pos_item_ids, neg_item_ids)

            # bpr loss
            # clamp
            diff = (pos_scores.unsqueeze(1)-neg_scores) if len(neg_scores.shape)==2 else (pos_scores-neg_scores)
            diff = torch.clamp(diff, -15, 15)
            loss_bpr = -torch.mean(F.logsigmoid(diff))

            # l2 reg
            user_e = model.user_emb[user_ids]
            pos_e  = model.item_emb[pos_item_ids]
            if len(neg_item_ids.shape)==2:
                neg_item_ids = neg_item_ids.view(-1)
            neg_e  = model.item_emb[neg_item_ids]
            reg = args.l2_reg*(user_e.norm(2).pow(2) + pos_e.norm(2).pow(2)+neg_e.norm(2).pow(2))/train_loader.batch_size

            loss = loss_bpr+reg

            optimizer.zero_grad()
            loss.backward()

            # gradient clipping => further avoid NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss/len(train_loader)

        # Validation
        valid_loss = compute_bpr_loss(model, valid_loader, device, args.l2_reg)
        # Recall@10,20,40 and NDCG@10,20,40 계산
        metrics = compute_metrics(model, valid_loader, device, ui_mat_train, k_values=[10, 20, 40])
        valid_recall = metrics["Recall@40"]
        
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"Epoch[{ep}/{args.epoch}] \ntrain_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, \n{metrics_str}")

        # Early stopping based on Recall@40
        if valid_recall > best_recall:
            best_recall = valid_recall
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"EarlyStopping at epoch={ep}, best_valid_recall@40={best_recall:.4f}")
                break

        # NaN check
        if torch.isnan(loss):
            print("!!! Loss is NaN => break training.")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # save emb
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    user_np = model.get_user_emb_matrix()
    item_np = model.get_item_emb_matrix()
    np.save(os.path.join(args.save_path,"user_embed.npy"), user_np)
    np.save(os.path.join(args.save_path,"item_embed.npy"), item_np)
    print(f"Saved MF embedding => {args.save_path}")

def main():
    args = parse_args()
    train_mf_tmall(args)

if __name__=="__main__":
    main()
