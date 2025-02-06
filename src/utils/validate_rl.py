def validate_agent(rl_df, agent, env):
    """
    검증 코드: 랜덤하게 5개의 상태를 샘플링하여 행동 예측 및 결과 출력.
    저장된 모델을 불러와서 사용.
    
    :param rl_df: DataFrame containing state information (Polars or Pandas).
    :param agent: Trained HRL Agent (initialized but not yet loaded with weights).
    :param env: Environment object.
    """
    # 저장된 모델 로드 (weights_only=True 설정)
    agent.high_policy.load_state_dict(torch.load('best_high_policy.pth', weights_only=True))
    agent.low_actor.load_state_dict(torch.load('best_low_actor.pth', weights_only=True))
    agent.low_critic_1.load_state_dict(torch.load('best_low_critic_1.pth', weights_only=True))
    agent.low_critic_2.load_state_dict(torch.load('best_low_critic_2.pth', weights_only=True))

    # 모델을 평가 모드로 설정
    agent.high_policy.eval()
    agent.low_actor.eval()
    agent.low_critic_1.eval()
    agent.low_critic_2.eval()

    # 랜덤하게 5개의 상태를 샘플링 (Polars -> Dict)
    sampled_states = rl_df.sample(n=5).to_dicts()  # Polars DataFrame의 각 행을 딕셔너리로 변환
    
    print(f"{'State (Current Price)':<25}{'Action 1 (Price Adj)':<20}{'Action 2 (Top_K)':<20}{'Changed Price':<20}{'Predicted Reward':<20}")
    print("=" * 110)
    
    for state_row in sampled_states:
        # 현재 상태 가져오기 (현재 가격 포함)
        current_price = state_row['price']
        state = np.array([state_row[col] for col in env.df.columns], dtype=np.float32)  # 'price' 포함

        # 상위 정책: 최적 가격 조정(Action 1) 선택
        price_action_idx = agent.select_high_action(state)
        price_adjustment = env.price_action_space[price_action_idx]
        changed_price = max(0, current_price * (1 + price_adjustment))
        price_adjustment2 = -0.3

        state_low = np.concatenate([state, [price_adjustment]])
        state_low2 = np.concatenate([state, [price_adjustment2]])

        # 하위 정책: 최적 추천 사용자 수(Action 2) 선택
        top_k_action = agent.select_low_action(state_low)
        top_k_action2 = 35000

        # 행동에 대한 예상 보상 계산
        predicted_reward = agent.low_critic_1(
            torch.FloatTensor(state_low).unsqueeze(0).to(agent.device),
            torch.FloatTensor([top_k_action / 100000.0]).unsqueeze(0).to(agent.device)
        ).item()

        predicted_reward2 = agent.low_critic_1(
            torch.FloatTensor(state_low2).unsqueeze(0).to(agent.device),
            torch.FloatTensor([top_k_action2 / 100000.0]).unsqueeze(0).to(agent.device)
        ).item()

        # 결과 출력
        print(f"{current_price:<25.2f}{price_adjustment:<20.2f}{top_k_action:<20d}{changed_price:<20.2f}{predicted_reward:<20.2f} Optimal")
        print(f"{current_price:<25.2f}{price_adjustment2:<20.2f}{top_k_action2:<20d}{changed_price:<20.2f}{predicted_reward2:<20.2f} Compare")