def train_agent(env, agent, num_episodes=50, batch_size=128):
    """
    Generalized training function for reinforcement learning agents.
    Supports both SACAgent and DDPGAgent.

    :param env: The environment instance.
    :param agent: The RL agent (SACAgent or DDPGAgent).
    :param num_episodes: Number of training episodes.
    :param batch_size: Batch size for replay buffer.
    :return: List of total rewards per episode.
    """
    episode_rewards = []

    for episode in range(num_episodes):
        state_continuous = env.reset()
        total_reward = 0

        while True:
            # Get actions from the agent
            price_action, top_k_action = agent.act(state_continuous)

            # Step through the environment
            next_state_continuous, reward, done = env.step((price_action, top_k_action))

            # Store experience in replay buffer
            agent.remember(
                state_continuous,
                [price_action, top_k_action],
                reward,
                next_state_continuous if not done else None,
                done,
            )

            # Update state and accumulate reward
            state_continuous = next_state_continuous if not done else None
            total_reward += reward

            if done:
                break

        # Train the agent using replay buffer
        agent.replay(batch_size)

        # Log total reward for the episode
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    return episode_rewards
