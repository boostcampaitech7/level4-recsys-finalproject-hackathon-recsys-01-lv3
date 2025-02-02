# main.py

import argparse
from src.train_manager import TrainManager

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train Recommendation Model (MF or MBGCN)")
    
#     # 모델 선택 및 기본 하이퍼파라미터
#     parser.add_argument("--model", type=str, choices=["MF", "MBGCN"], default="MBGCN",
#                         help="Model to train: MF or MBGCN")
#     parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
#     parser.add_argument("--l2_reg", type=float, default=1e-4, help="L2 regularization coefficient")
#     parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
#     parser.add_argument("--epoch", type=int, default=400, help="Maximum number of epochs")
#     parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
#     parser.add_argument("--embedding_size", type=int, default=64, help="Embedding dimension")
    
#     # MBGCN 전용 하이퍼파라미터
#     parser.add_argument("--num_layers", type=int, default=2, help="Number of GCN layers (for MBGCN)")
#     parser.add_argument("--lamb", type=float, default=0.5, help="Weight for user-based CF in final score (for MBGCN)")
#     parser.add_argument("--node_dropout", type=float, default=0.2, help="Node dropout rate (for MBGCN)")
#     parser.add_argument("--message_dropout", type=float, default=0.2, help="Message dropout rate (for MBGCN)")
#     parser.add_argument("--alpha_mode", type=str, choices=["global", "per_user"], default="global",
#                         help="Alpha mode for behavior weighting (for MBGCN)")
#     parser.add_argument("--alpha_learning", action="store_true", default=True,
#                         help="Whether to learn alpha parameters (for MBGCN); default is True")
#     parser.add_argument("--item_cf_mode", type=str, default="bigmat",
#                         help="Item CF mode for MBGCN (default: bigmat)")
#     parser.add_argument("--item_alpha", action="store_true",
#                         help="Whether to use item_alpha in MBGCN")
    
#     # 사전 학습된 임베딩 관련 (MBGCN에서 MF pretrain 임베딩을 초기값으로 사용)
#     parser.add_argument("--use_pretrain", action="store_true",
#                         help="Use pre-trained MF embeddings for MBGCN initialization")
#     parser.add_argument("--pretrain_user_emb", type=str, default="",
#                         help="Path to pre-trained user embedding (npy file)")
#     parser.add_argument("--pretrain_item_emb", type=str, default="",
#                         help="Path to pre-trained item embedding (npy file)")
    
#     # 데이터 경로 관련
#     parser.add_argument("--data_path", type=str, default="Hackathon/src/data/MBGCN",
#                         help="Root directory of dataset")
#     parser.add_argument("--dataset_name", type=str, default="Tmall",
#                         help="Dataset name folder (e.g., Tmall)")
#     parser.add_argument("--relations", type=str, default="buy,cart,click,collect",
#                         help="Comma-separated relations")
#     parser.add_argument("--save_path", type=str, default="Hackathon/src/data/MBGCN/pretrained_model",
#                         help="Path to save pretrained embeddings")
#     parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu",
#                         help="Device to use for training")
#     return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description="Train Recommendation Model (MF or MBGCN)")
    
    # 모델 선택 및 기본 하이퍼파라미터 (공통)
    parser.add_argument("--model", type=str, choices=["MF", "MBGCN"], default="MBGCN",
                        help="Model to train: MF or MBGCN")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l2_reg", type=float, default=1e-4, help="L2 regularization coefficient")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--epoch", type=int, default=400, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument("--embedding_size", type=int, default=64, help="Embedding dimension")
    
    # MBGCN 전용 하이퍼파라미터
    parser.add_argument("--num_layers", type=int, default=2, help="Number of propagation layers (for MBGCN)")
    parser.add_argument("--lamb", type=float, default=0.5, help="Weight for user-based CF in final score (for MBGCN)")
    parser.add_argument("--node_dropout", type=float, default=0.2, help="Node dropout rate (for MBGCN)")
    parser.add_argument("--message_dropout", type=float, default=0.2, help="Message dropout rate (for MBGCN)")
    
    # 사전 학습된 임베딩 관련 (MBGCN에서 MF pretrain 임베딩을 초기값으로 사용)
    parser.add_argument("--use_pretrain", action="store_true",
                        help="Use pre-trained MF embeddings for MBGCN initialization")
    parser.add_argument("--pretrain_user_emb", type=str, default="",
                        help="Path to pre-trained user embedding (npy file)")
    parser.add_argument("--pretrain_item_emb", type=str, default="",
                        help="Path to pre-trained item embedding (npy file)")
    
    # 데이터 경로 관련
    parser.add_argument("--data_path", type=str, default="Hackathon/src/data/MBGCN",
                        help="Root directory of dataset")
    parser.add_argument("--dataset_name", type=str, default="Tmall",
                        help="Dataset name folder (e.g., Tmall)")
    parser.add_argument("--relations", type=str, default="buy,cart,click,collect",
                        help="Comma-separated relations")
    parser.add_argument("--save_path", type=str, default="Hackathon/src/data/MBGCN/pretrained_model",
                        help="Path to save pretrained embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = TrainManager(args)
    trainer.train()

if __name__ == "__main__":
    main()
