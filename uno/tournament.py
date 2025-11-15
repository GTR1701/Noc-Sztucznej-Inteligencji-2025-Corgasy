from uno.env import UnoEnv
from pprint import pprint
import numpy as np
from typing import List
from uno.agents import BaseAgent


class Tournament:
    """
    Manages a single game of Uno between multiple agents.
    """

    def __init__(self, env: UnoEnv, agents: List[BaseAgent]):
        self.env = env

        assert all(
            hasattr(agent, "select_action") for agent in agents
        ), "All agents must implement 'select_action' method."
        
        self.agents = dict(enumerate(agents))

        if len(self.agents) != self.env.num_players:
            raise ValueError(
                f"Agents list must have length equal to the number of players: {self.env.num_players}"
            )

    def run_game(self):
        """
        Resets the environment and runs one full game until 'done'.
        """
        print("--- Starting New Game ---")

        try:
            obs, info = self.env.reset()
            done = False

            while not done:
                current_player_id = info["player_id"]
                legal_actions = info["legal_actions"]

                agent = self.agents[current_player_id]

                action = agent.select_action(obs, legal_actions)

                obs, reward, done, truncated, info = self.env.step(action)

        except Exception as e:
            print(f"An error occurred during the game: {e}")
            self.env.close()
            return

        # --- Game Over ---
        print("\n--- Game Over! ---")
        payoffs = self.env.env.get_payoffs()
        pprint(f"Final Payoffs: {payoffs}")
        self.env.close()
        return payoffs

    def run_tournament(self, num_games: int):
        """
        Runs a tournament of multiple games.
        """
        payoffs_list = []
        for game_idx in range(num_games):
            print(f"=== Game {game_idx + 1} of {num_games} ===")
            payoffs = self.run_game()
            payoffs_list.append(payoffs)
        pprint(f"=== Tournament Over: {num_games} Games Played ===")

        payoffs_array = np.array(payoffs_list) / 2 + 0.5  # Normalize payoffs to [0, 1]
        avg_payoffs = np.mean(payoffs_array, axis=0)

        for player_id, avg_payoff in enumerate(avg_payoffs):
            print(f"Player {player_id} Percentage of won games: { 100* avg_payoff:.1f}")

        return payoffs_list
