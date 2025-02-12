# optuna_tuning_mf.py
import os
import optuna
import torch
import argparse
from src.train_manager import TrainManager

def objective(trial: optuna.Trial):
    args = argparse.Namespace()
    
    # MF 모델 고정
    args.model = "MF"
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
    args.batch_size = 4096
    args.epoch = 100     # 실험 속도를 위해 낮게 설정 (필요에 따라 조정)
    args.patience = 10
    args.embedding_size = trial.suggest_categorical("embedding_size", [32, 64, 128])
    
    # MF 전용 인자
    args.use_pretrain = False
    args.pretrain_path = ""
    args.pretrain_frozen = False
    args.create_embeddings = True
    
    # 데이터셋 관련 (예시로 Tmall 사용)
    args.data_path = "src/data/MBGCN"
    args.dataset_name = "data/smartphones"
    args.relations = "purchase,cart,view"
    
    args.save_path = "./optuna_saved_models/MF"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 내부 코드에서 L2 정규화 인자로 사용할 이름 통일
    args.L2_norm = args.l2_reg

    trainer = TrainManager(args)
    best_recall = trainer.train()  # 최종 평가 지표(예: Recall@40) 반환한다고 가정
    return best_recall

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    print("===== 최적의 MF 하이퍼파라미터 =====")
    best_trial = study.best_trial
    print(f"최고 평가 지표: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
