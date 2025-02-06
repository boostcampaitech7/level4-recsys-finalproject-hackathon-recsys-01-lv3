import os
import sys
project_root = os.path.join(os.path.expanduser("~"), "Hackathon")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agent.DQNTD3Agent import HRLAgent
from src.data.preprocess_for_rl.environment import DynamicPricingEnv
from src.data.preprocess_for_rl.cal_elasticity import ElasticityCalculator
from src.data.preprocess_for_rl.data_preprocess import PreprocessRL
import yaml
import polars as pl


DATA_PATH = "/data/ephemeral/home/Hackathon/data/final_purchase_df.parquet"
CONFIG = "/data/ephemeral/home/Hackathon/config/rl.yaml"
REC_PATH = "/data/ephemeral/home/Hackathon/scripts/eda/item_topk.pkl"

def main():
    # 설정 파일 로드
    with open(CONFIG, "r") as file:
        config = yaml.safe_load(file)
    print("설정 파일 불러오기 완료.")

    # 데이터 불러오기
    df = pl.read_parquet(DATA_PATH)
    preprocess = PreprocessRL(df, REC_PATH)

    print("데이터 불러오기 완료.")

    # 데이터 전처리
    elasticity_calculator = ElasticityCalculator(df)
    elasticity_df = elasticity_calculator.run()
    print("가격 탄력성 계산 완료.")
    train_df = preprocess.make_train_df()
    print("학습 데이터 전처리 완료.")
    recsys_df = preprocess.load_recsys()
    print("추천시스템 결과 불러오기 완료.")
    true_user_df = preprocess.true_user_df()
    print("실제 구매 유저 데이터 전처리 완료.")

    # 환경 불러오기
    env = DynamicPricingEnv(
        df=train_df,  # DataFrame containing state information
        item_user_scores=recsys_df,  # Precomputed recommendation scores
        true_users_by_product=true_user_df,  # Mapping of product IDs to true users
        elasticity_df=elasticity_df,  # Price elasticity data
        tau=1  # Optional parameter for reward scaling
    )
    print("강화학습 MDP 환경 초기화 및 불러오기 완료.")

    # Agent 초기화 및 불러오기
    agent = HRLAgent(env, config)
    print("HRL : DQN-TD3 학습 Agent 초기화 및 불러오기 완료.")

    print("학습 시작")
    # 학습
    agent.train(num_epochs=100)

# 실행
if __name__ == "__main__":
    main()



