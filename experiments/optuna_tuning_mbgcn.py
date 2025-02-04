# optuna_tuning_mbgcn.py

import optuna
import torch
import argparse
from src.train_manager import TrainManager

def objective(trial: optuna.Trial):
    args = argparse.Namespace()
    
    # MBGCN 모델 고정
    args.model = "MBGCN"
    args.lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    args.l2_reg = trial.suggest_loguniform("l2_reg", 1e-6, 1e-2)
    args.batch_size = 2048
    args.epoch = 100     # 실험 속도를 위해 낮게 설정 (필요에 따라 조정)
    args.patience = 10
    args.embedding_size = trial.suggest_categorical("embedding_size", [32, 64, 128])
    
    # MBGCN 전용 하이퍼파라미터
    args.lamb = trial.suggest_uniform("lamb", 0.1, 1.0)
    args.node_dropout = trial.suggest_uniform("node_dropout", 0.0, 0.5)
    args.message_dropout = trial.suggest_uniform("message_dropout", 0.0, 0.5)
    
    # 사전 학습 임베딩 관련 (MBGCN는 MF 사전학습 모델을 사용)
    # pretrain_path는 실험 환경에 맞게 후보 중 선택하거나 고정값을 넣을 수 있습니다.
    args.pretrain_path = trial.suggest_categorical("pretrain_path", [
        "Hackathon/src/data/MBGCN/data/smartphones/pretrained_mf_runs/MF_lr1e-4_L21e-5_dim64",
        "Hackathon/src/data/MBGCN/data/smartphones/pretrained_mf_runs/MF_lr1e-3_L21e-3_dim64"
    ])
    args.pretrain_frozen = trial.suggest_categorical("pretrain_frozen", [False, True])
    args.create_embeddings = False
    
    # 데이터셋 관련 (예시로 smartphones 데이터셋 사용)
    args.data_path = "src/data/MBGCN"
    args.dataset_name = "data/smartphones"
    args.relations = "purchase,cart,view"
    
    args.save_path = "./optuna_saved_models/MBGCN"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 내부 코드에서 L2 정규화 인자로 사용할 이름 통일
    args.L2_norm = args.l2_reg

    trainer = TrainManager(args)
    best_recall = trainer.train()  # 최종 평가 지표(예: Recall@40) 반환한다고 가정
    return best_recall

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    print("===== 최적의 MBGCN 하이퍼파라미터 =====")
    best_trial = study.best_trial
    print(f"최고 평가 지표: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
