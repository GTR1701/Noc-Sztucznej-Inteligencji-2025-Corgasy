import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state, legal_actions):
        """
        Args:
            state: Raw state dict from RLCard containing 'obs', 'legal_actions', 'raw_obs', etc.
            legal_actions: List of legal action indices
        """
        pass


# =================================
# --- YOUR CUSTOM AGENT IMPLEMENTATION ---
# =================================
class SubmissionAgent(BaseAgent):
    def select_action(self, state, legal_actions):
        # TODO implement your agent's logic here
        # You can access:
        # - state['obs']: (4, 4, 15) tensor with game state
        # - state['raw_obs']['hand']: list of card names in your hand
        # - state['raw_obs']['target']: the card on the table
        # - state['raw_obs']['num_cards']: number of cards each player has
        
        return legal_actions[0]


# =================================


class RandomAgent(BaseAgent):
    def select_action(self, state, legal_actions):
        return np.random.choice(legal_actions)


class QLearningAgent:
    def __init__(
        self,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1000,
        learning_rate=0.1,
        discount_factor=0.9,
    ):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = self.epsilon_start
        self.epsilon_decay = epsilon_decay

    def _encode_state(self, state):
        """Convert raw state to a hashable representation for Q-table"""
        if 'obs' not in state:
            return tuple([0] * 108)
        
        # Flatten the observation tensor
        obs_flat = state['obs'].flatten()
        # Convert to tuple for hashing
        return tuple(obs_flat.astype(int).tolist())

    def select_action(self, state, legal_actions):
        state_key = self._encode_state(state)
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)  # Explore
        else:
            # Exploit: choose action with highest Q-value
            q_values = [self.q_table.get((state_key, a), 0) for a in legal_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(legal_actions, q_values) if q == max_q]
            return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done, episode):
        state_key = self._encode_state(state)
        next_state_key = self._encode_state(next_state)
        
        # Update Q-values based on the experience
        current_q = self.q_table.get((state_key, action), 0)
        max_next_q = (
            max([self.q_table.get((next_state_key, a), 0) for a in range(0, 108)])
            if not done
            else 0
        )
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[(state_key, action)] = new_q

        self.update_epsilon(episode)

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-0.5 * episode / self.epsilon_decay)

    def eval_mode(self):
        self.epsilon = 0.0

    def train_mode(self):
        self.epsilon = self.epsilon_start