# main.py

import argparse
import torch
from src.train_manager import TrainManager

def parse_args():
    parser = argparse.ArgumentParser(description="Train Recommendation Model (MF or MBGCN)")
    
    # 공통 인자
    parser.add_argument("--model", type=str, choices=["MF", "MBGCN"], default="MBGCN",
                        help="Model to train: MF or MBGCN")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l2_reg", type=float, default=1e-4, help="L2 regularization coefficient")
    parser.add_argument("--train_batch_size", type=int, default=4096, help="Training Batch size")
    parser.add_argument("--valid_batch_size", type=int, default=4096, help="Validation Batch size")
    parser.add_argument("--epoch", type=int, default=400, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument("--embedding_size", type=int, default=64, help="Embedding dimension")
    
    # MBGCN 전용 인자
    # parser.add_argument("--num_layers", type=int, default=2, help="Number of propagation layers (for MBGCN)")
    parser.add_argument("--lamb", type=float, default=0.5, help="Weight for user-based CF in final score (for MBGCN)")
    parser.add_argument("--node_dropout", type=float, default=0.2, help="Node dropout rate (for MBGCN)")
    parser.add_argument("--message_dropout", type=float, default=0.2, help="Message dropout rate (for MBGCN)")
    
    # 사전 학습 임베딩 관련 인자
    parser.add_argument("--use_pretrain", action="store_true",
                        help="Use pre-trained MF embeddings for MBGCN initialization")
    parser.add_argument("--pretrain_path", type=str, default="",
                        help="Path to pre-trained model directory containing model.pkl")
    parser.add_argument("--pretrain_frozen", action="store_true", default=False,
                        help="Whether pre-trained embeddings are frozen")
    parser.add_argument("--create_embeddings", action="store_true", default=True,
                        help="Whether to create new embeddings (if not, load pre-trained)")
    
    # 데이터 관련 인자
    parser.add_argument("--data_path", type=str, default="Hackathon/src/data/MBGCN",
                        help="Root directory of dataset")
    parser.add_argument("--dataset_name", type=str, default="data/final",
                        help="Dataset name folder (e.g., Tmall)")
    parser.add_argument("--relations", type=str, default="buy,cart,click,collect",
                        help="Comma-separated relations")
    parser.add_argument("--save_path", type=str, default="Hackathon/src/data/MBGCN/pretrained_model",
                        help="Path to save pretrained embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    
    args = parser.parse_args()
    
    # train_manager.py와 모델 내부에서 사용하는 L2 정규화 인자 이름 통일 (L2_norm)
    args.L2_norm = args.l2_reg
    
    # pretrain 사용 시 새 임베딩을 만들지 않고 로드하도록 처리
    if args.use_pretrain:
        args.create_embeddings = False
    
    return args

def main():
    args = parse_args()
    trainer = TrainManager(args)
    trainer.train()

if __name__ == "__main__":
    main()
