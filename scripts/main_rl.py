from src.pipelines import train_agent
from src.agent import SACAgent, DDPGAgent
from src.data.preprocess_for_rl.environment import DynamicPricingEnv

# Initialize the environment (DynamicPricingEnv is assumed to be defined elsewhere)
env = DynamicPricingEnv(df=df,
                        interaction_matrix_csr=interaction_matrix_csr,
                        item_similarity=item_similarity,
                        item_user_scores=item_user_scores,
                        true_users_by_product=true_users_by_product)

# Initialize common parameters
state_size = env.reset().shape[0]
action_size = 2

# Train SAC Agent
print("Initializing and training SAC Agent...")
sac_agent = SACAgent(state_size=state_size, action_size=action_size)
sac_rewards = train_agent(env=env, agent=sac_agent, num_episodes=50, batch_size=128)

# Train DDPG Agent
print("Initializing and training DDPG Agent...")
ddpg_agent = DDPGAgent(state_size=state_size, action_size=action_size)
ddpg_rewards = train_agent(env=env, agent=ddpg_agent, num_episodes=50, batch_size=128)

# Save or analyze results
print("SAC Rewards:", sac_rewards)
print("DDPG Rewards:", ddpg_rewards)
