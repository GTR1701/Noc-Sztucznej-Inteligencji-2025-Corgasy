from uno.env import UnoEnv
from uno.agents import QLearningAgent, RandomAgent, FeatureBasedQLearningAgent  # Dodać import
from uno.tournament import Tournament
import joblib
import pickle  # Dla lepszej obsługi custom objects


if __name__ == "__main__":

    # Wybór agenta do trenowania
    agent_type = "feature_based"  # "old_qlearning" lub "feature_based"
    train = True  # Set to False to skip training and only test
    
    agent = FeatureBasedQLearningAgent(
        learning_rate=0.15,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=5000,
        discount_factor=0.9
    )
    model_filename = "feature_based_agent.pkl"

    
    opponent_agent = RandomAgent()

    if train:
        print(f"Training {agent_type} agent...")
        agent.train_mode()  # Włącz tryb treningu
        
        env = UnoEnv(render_mode=None)

        num_episodes = 8000
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False

            cumulative_reward = 0
            while not done:
                current_player_id = info["player_id"]
                legal_actions = info["legal_actions"]

                if current_player_id == 0:
                    agent_to_use = agent
                else:
                    agent_to_use = opponent_agent

                action = agent_to_use.select_action(obs, legal_actions)

                next_obs, reward, done, truncated, info = env.step(action)

                if current_player_id == 0:
                    agent.learn(obs, action, reward, next_obs, done, episode)
                cumulative_reward += reward

                obs = next_obs
                
            # Progress tracking
            if episode % 500 == 0:
                print(f"Episode {episode + 1}, Reward: {cumulative_reward}")
                if hasattr(agent, 'get_stats'):
                    stats = agent.get_stats()
                    print(f"  Q-table size: {stats['q_table_size']}, Epsilon: {stats['epsilon']:.3f}")
                    
        print(f"Training completed over {num_episodes} episodes.")
        
        # Save model with appropriate method
        if hasattr(agent, 'save_model'):
            agent.save_model(model_filename)  # Nasz custom save
        else:
            joblib.dump(agent, model_filename)  # Fallback dla starych agentów

    # Testing the trained agent
    print(f"Loading and testing {agent_type} agent...")
    
    if agent_type == "feature_based":
        agent = FeatureBasedQLearningAgent()  # Create new instance
        if hasattr(agent, 'load_model'):
            agent.load_model(model_filename)
        agent.eval_mode()  # Switch to evaluation mode
    else:
        agent = joblib.load(model_filename)
        if hasattr(agent, 'eval_mode'):
            agent.eval_mode()
    
    opponent_agent = RandomAgent()
    env = UnoEnv(render_mode=None)

    tournament = Tournament(env, [agent, opponent_agent])
    num_test_games = 1000  # Reduced for faster testing
    
    print(f"Running tournament with {num_test_games} games...")
    payoffs = tournament.run_tournament(num_test_games)
    
    # Dodatkowa analiza wyników
    import numpy as np
    payoffs_array = np.array(payoffs)
    agent_wins = np.sum(payoffs_array[:, 0] > 0)
    win_percentage = agent_wins / num_test_games * 100
    
    print(f"\n=== Final Results ===")
    print(f"Agent wins: {agent_wins}/{num_test_games} ({win_percentage:.1f}%)")
    print(f"Average reward: {np.mean(payoffs_array[:, 0]):.3f}")
    
    if hasattr(agent, 'get_stats'):
        final_stats = agent.get_stats()
        print(f"Final Q-table size: {final_stats['q_table_size']}")
