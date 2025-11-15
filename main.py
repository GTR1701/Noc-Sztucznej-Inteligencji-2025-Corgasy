from uno.env import UnoEnv
from uno.agents import QLearningAgent, RandomAgent
from uno.tournament import Tournament
import joblib


if __name__ == "__main__":


    # example usage: training and testing a Q-learning agent
    train = False  # Set to False to skip training and only test
    agent = QLearningAgent()  # qlearning agent, generally dosent perform well on games with this many states, so build your own :D
    opponent_agent = RandomAgent()

    if train:
        env = UnoEnv(render_mode=None)

        num_episodes = 5000
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
                    agent_to_use = (
                        opponent_agent  
                    )

                action = agent_to_use.select_action(obs, legal_actions)

                next_obs, reward, done, truncated, info = env.step(action)

                if current_player_id == 0:
                    agent.learn(obs, action, reward, next_obs, done, episode)
                cumulative_reward += reward

                obs = next_obs
            print(f"Episode {episode + 1}, Cumulative Reward: {cumulative_reward}")
        print(f"Training completed over {num_episodes} episodes.")
        joblib.dump(agent, "q_learning_agent.pkl")


    # Testing the trained agent
    opponent_agent = RandomAgent()
    agent = joblib.load("q_learning_agent.pkl")
    env = UnoEnv(render_mode=None)


    tournament = Tournament(env, [agent, opponent_agent])
    num_test_games = 10000
    tournament.run_tournament(num_test_games)
