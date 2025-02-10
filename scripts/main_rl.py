import os
import sys
import yaml
import argparse
import polars as pl
from src.agent.DQNTD3Agent import HRLAgent
from src.data.preprocess_for_rl.environment import DynamicPricingEnv
from src.data.preprocess_for_rl.cal_elasticity import ElasticityCalculator
from src.data.preprocess_for_rl.data_preprocess import PreprocessRL

def main(args):
    """
    Main function to execute the reinforcement learning pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    project_root = os.path.join(os.path.expanduser("~"), "Hackathon")
    if project_root not in sys.path:
        sys.path.append(project_root)

    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    print("Configuration file loaded successfully.")

    df = pl.read_parquet(args.data_path)
    preprocess = PreprocessRL(df, args.rec_path)
    print("Data loaded successfully.")

    elasticity_calculator = ElasticityCalculator(df)
    elasticity_df = elasticity_calculator.run()
    print("Price elasticity calculation completed.")

    train_df = preprocess.make_train_df(random=True)
    print("Training data preprocessing completed.")

    recsys_df = preprocess.load_recsys()
    print("Recommendation system data loaded successfully.")

    env = DynamicPricingEnv(
        df=train_df,
        item_user_scores=recsys_df,
        elasticity_df=elasticity_df,
        raw_df=df
    )
    print("Reinforcement learning environment initialized successfully.")

    agent = HRLAgent(env, config)
    print("HRL Agent (DQN-TD3) initialized successfully.")

    print("Starting training...")
    agent.train(num_episodes=args.num_episodes, warm_up=args.warm_up)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the reinforcement learning pipeline.")
    
    parser.add_argument("--data-path", type=str, required=True, 
                        help="Path to the input data file (Parquet format).")
    
    parser.add_argument("--config-path", type=str, required=True, 
                        help="Path to the YAML configuration file.")
    
    parser.add_argument("--rec-path", type=str, required=True, 
                        help="Path to the recommendation system data file (Pickle format).")
    
    parser.add_argument("--num-episodes", type=int, default=200, 
                        help="Number of episodes for training. Default is 200.")
    
    parser.add_argument("--warm-up", type=int, default=20, 
                        help="Number of warm-up episodes. Default is 20.")
    
    args = parser.parse_args()
    
    main(args)
