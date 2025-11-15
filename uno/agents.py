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

# =================================
# --- SUPPORTING CLASSES FOR FEATURE EXTRACTION ---
# =================================

class UnoActionMapper:
    """Maps RLCard card names to action indices based on the action table"""
    
    def __init__(self):
        self.card_to_action = {}
        
        # Red cards (0-14)
        for i in range(10):
            self.card_to_action[f'r-{i}'] = i
        self.card_to_action['r-skip'] = 10
        self.card_to_action['r-reverse'] = 11  
        self.card_to_action['r-draw_2'] = 12
        self.card_to_action['r-wild'] = 13
        self.card_to_action['r-wild_draw_4'] = 14
        
        # Green cards (15-29)  
        for i in range(10):
            self.card_to_action[f'g-{i}'] = 15 + i
        self.card_to_action['g-skip'] = 25
        self.card_to_action['g-reverse'] = 26
        self.card_to_action['g-draw_2'] = 27
        self.card_to_action['g-wild'] = 28
        self.card_to_action['g-wild_draw_4'] = 29
        
        # Blue cards (30-44)
        for i in range(10):
            self.card_to_action[f'b-{i}'] = 30 + i  
        self.card_to_action['b-skip'] = 40
        self.card_to_action['b-reverse'] = 41
        self.card_to_action['b-draw_2'] = 42
        self.card_to_action['b-wild'] = 43
        self.card_to_action['b-wild_draw_4'] = 44
        
        # Yellow cards (45-59)
        for i in range(10):
            self.card_to_action[f'y-{i}'] = 45 + i
        self.card_to_action['y-skip'] = 55  
        self.card_to_action['y-reverse'] = 56
        self.card_to_action['y-draw_2'] = 57
        self.card_to_action['y-wild'] = 58
        self.card_to_action['y-wild_draw_4'] = 59
        
        # Action 60 = draw (no card mapping)
    
    def get_action_for_card(self, card_name):
        """Get action index for card name"""
        return self.card_to_action.get(card_name, None)
    
    def get_playable_cards_from_hand(self, hand, legal_actions):
        """Get which cards from hand are actually playable"""
        playable = []
        for card in hand:
            action_idx = self.get_action_for_card(card)
            if action_idx is not None and action_idx in legal_actions:
                playable.append(card)
        return playable


class SimpleCardMapper:
    """Simple card parser for RLCard format"""
    
    def __init__(self):
        self.colors = {'r': 'red', 'g': 'green', 'b': 'blue', 'y': 'yellow'}
        self.color_indices = {'red': 0, 'green': 1, 'blue': 2, 'yellow': 3}
    
    def parse_card(self, card_name):
        """Parse card name like 'r-1', 'b-reverse', 'g-wild'"""
        if not card_name or '-' not in card_name:
            return (None, 'unknown', None)
            
        parts = card_name.lower().split('-', 1)
        if len(parts) != 2:
            return (None, 'unknown', None)
            
        color_char, card_part = parts
        color = self.colors.get(color_char)
        
        if card_part.isdigit():
            return (color, 'number', int(card_part))
        elif card_part in ['skip', 'reverse']:
            return (color, card_part, None)
        elif card_part == 'draw_2':
            return (color, 'draw', 2)
        elif card_part in ['wild', 'wild_draw_4']:
            return (None, 'wild', 4 if 'draw_4' in card_part else None)
        else:
            return (None, 'unknown', None)
    
    def get_color_index(self, color):
        return self.color_indices.get(color, -1)


class ImprovedFeatureExtractor:
    """Improved feature extraction with proper card-to-action mapping"""
    
    def __init__(self):
        self.card_mapper = SimpleCardMapper()
        self.action_mapper = UnoActionMapper()
    
    def extract_features(self, state):
        """Extract 20 strategic features from game state"""
        if not state or 'raw_obs' not in state:
            return tuple([0] * 20)
        
        raw_obs = state['raw_obs']
        hand = raw_obs.get('hand', [])
        target = raw_obs.get('target', '')
        num_cards = raw_obs.get('num_cards', [7, 7])
        legal_actions = state.get('legal_actions', [])
        
        features = []
        
        # 1. Hand composition (8 features)
        color_counts = {'red': 0, 'green': 0, 'blue': 0, 'yellow': 0}
        action_count = 0
        number_count = 0
        wild_count = 0
        
        for card in hand:
            color, card_type, value = self.card_mapper.parse_card(card)
            if color and color in color_counts:
                color_counts[color] += 1
            if card_type == 'number':
                number_count += 1
            elif card_type in ['skip', 'reverse', 'draw']:
                action_count += 1
            elif card_type == 'wild':
                wild_count += 1
        
        # Color distribution (4)
        features.extend([min(color_counts[c], 3) for c in ['red', 'green', 'blue', 'yellow']])
        # Card type counts (4)
        features.extend([min(number_count, 5), min(action_count, 3), 
                        min(wild_count, 2), min(len(hand), 7)])
        
        # 2. Playability analysis (6 features) - KEY IMPROVEMENT
        playable_cards = self.action_mapper.get_playable_cards_from_hand(hand, legal_actions)
        
        playable_colors = {'red': 0, 'green': 0, 'blue': 0, 'yellow': 0}
        playable_numbers = 0
        playable_actions = 0
        
        for card in playable_cards:
            color, card_type, value = self.card_mapper.parse_card(card)
            if color and color in playable_colors:
                playable_colors[color] += 1
            if card_type == 'number':
                playable_numbers += 1
            elif card_type in ['skip', 'reverse', 'draw']:
                playable_actions += 1
        
        features.extend([min(len(playable_cards), 5), min(playable_numbers, 3)])
        features.extend([min(playable_actions, 2), 1 if 60 in legal_actions else 0])
        
        # Dominant playable color
        if playable_colors:
            max_color = max(playable_colors.values())
            features.extend([1 if playable_colors[c] == max_color and max_color > 0 else 0 
                           for c in ['red', 'green']])  # Just 2 colors to save space
        else:
            features.extend([0, 0])
        
        # 3. Target analysis (3 features)
        target_color, target_type, target_value = self.card_mapper.parse_card(target)
        
        # Target color matches hand
        hand_colors = set(self.card_mapper.parse_card(card)[0] for card in hand)
        features.append(1 if target_color in hand_colors else 0)
        
        # Target type
        features.append(1 if target_type == 'number' else 0)
        features.append(1 if target_type in ['skip', 'reverse', 'draw'] else 0)
        
        # 4. Game state (3 features)
        if len(num_cards) >= 2:
            opponent_cards = num_cards[1:]
            features.append(min(min(opponent_cards), 7))  # Min opponent cards
            features.append(1 if any(c <= 2 for c in opponent_cards) else 0)  # Opponent close to win
            
            # Game phase
            total_cards = sum(num_cards)
            features.append(1 if total_cards <= 15 else (2 if total_cards <= 25 else 3))
        else:
            features.extend([7, 0, 3])
        
        return tuple(features[:20])  # Exactly 20 features


# =================================
# --- IMPROVED FEATURE-BASED Q-LEARNING AGENT ---
# =================================

class FeatureBasedQLearningAgent(BaseAgent):
    """
    IMPROVED Q-Learning agent with proper feature extraction and reward shaping
    """
    
    def __init__(self, 
                 learning_rate=0.15,
                 discount_factor=0.9,
                 epsilon_start=1.0,
                 epsilon_end=0.02,
                 epsilon_decay=5000):
        
        # Core Q-learning parameters
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.episode_count = 0
        
        # Feature extraction
        self.feature_extractor = ImprovedFeatureExtractor()
        
        # Training mode
        self.training_mode = True
        
        # Statistics and debugging
        self.q_table_size_history = []
        self.update_count = 0
        self.reward_history = []
        
    def select_action(self, state, legal_actions): # type: ignore
        """Select action using improved Q-learning with strategic bonuses"""
        try:
            # Edge case handling
            if not legal_actions:
                return 60  # Draw card
            if len(legal_actions) == 1:
                return legal_actions[0]
            
            # Get state representation
            state_key = self._get_state_key(state)
            
            # Epsilon-greedy exploration
            if np.random.rand() < self.epsilon and self.training_mode:
                return np.random.choice(legal_actions)
            
            # Get Q-values with strategic bonuses
            best_action = self._get_best_action(state, state_key, legal_actions)
            
            return best_action
            
        except Exception as e:
            # Robust fallback
            if self.training_mode:
                print(f"Error in action selection: {e}")
            return np.random.choice(legal_actions)
    
    def _get_best_action(self, state, state_key, legal_actions):
        """Get best action with Q-values and strategic bonuses"""
        raw_obs = state.get('raw_obs', {})
        hand = raw_obs.get('hand', [])
        num_cards = raw_obs.get('num_cards', [7, 7])
        
        action_values = []
        
        for action in legal_actions:
            # Base Q-value
            q_val = self.q_table.get((state_key, action), 0.0)
            
            # Strategic bonus
            bonus = self._calculate_strategic_bonus(action, hand, num_cards, legal_actions)
            
            total_value = q_val + bonus
            action_values.append((action, total_value))
        
        # Select best action
        best_action = max(action_values, key=lambda x: x[1])[0]
        return best_action
    
    def _calculate_strategic_bonus(self, action, hand, num_cards, legal_actions):
        """Calculate strategic bonus for action"""
        bonus = 0.0
        
        # Avoid drawing if possible
        if action == 60:
            bonus -= 0.02
        
        # Prefer action cards when opponent is close to winning
        if len(num_cards) > 1:
            min_opponent = min(num_cards[1:])
            if min_opponent <= 2:
                # Check if this is an action card (skip, reverse, draw_2)
                if action in [10,11,12, 25,26,27, 40,41,42, 55,56,57]:
                    bonus += 0.05
        
        # Small bonus for reducing hand size
        if action != 60:
            bonus += 0.01
        
        # Bonus for wild cards when hand is large
        if len(hand) > 5:
            # Wild cards: 13,14, 28,29, 43,44, 58,59
            if action in [13,14, 28,29, 43,44, 58,59]:
                bonus += 0.03
        
        return bonus
    
    def _get_state_key(self, state):
        """Extract features as state key"""
        return self.feature_extractor.extract_features(state)
    
    def learn(self, state, action, reward, next_state, done, episode):
        """Improved Q-learning update with reward shaping"""
        if not self.training_mode:
            return
        
        self.update_count += 1
        
        # Get state representations
        state_key = self._get_state_key(state)
        
        # Reward shaping for better learning
        shaped_reward = self._shape_reward(state, action, reward, next_state, done)
        self.reward_history.append(shaped_reward)
        
        # Current Q-value
        current_q = self.q_table.get((state_key, action), 0.0)
        
        # Next state max Q-value
        if done:
            max_next_q = 0.0
        else:
            next_state_key = self._get_state_key(next_state)
            next_legal = next_state.get('legal_actions', [])
            
            if next_legal:
                next_q_values = [self.q_table.get((next_state_key, a), 0.0) for a in next_legal]
                max_next_q = max(next_q_values) if next_q_values else 0.0
            else:
                max_next_q = 0.0
        
        # Q-learning update
        target_q = shaped_reward + self.gamma * max_next_q
        new_q = current_q + self.lr * (target_q - current_q)
        
        # Update Q-table
        self.q_table[(state_key, action)] = new_q
        
        # Update exploration
        self.update_epsilon(episode)
        
        # Debug info
        if episode % 1000 == 0 and self.update_count % 100 == 0:
            avg_reward = np.mean(self.reward_history[-100:]) if self.reward_history else 0
            print(f"Episode {episode}: Q-table: {len(self.q_table)}, Avg reward: {avg_reward:.3f}, Epsilon: {self.epsilon:.3f}")
        
        # Track Q-table growth
        if episode % 500 == 0:
            self.q_table_size_history.append(len(self.q_table))
    
    def _shape_reward(self, state, action, reward, next_state, done):
        """Shape rewards for better learning"""
        shaped_reward = reward
        
        if not done:  # Intermediate rewards
            raw_obs = state.get('raw_obs', {})
            next_raw_obs = next_state.get('raw_obs', {})
            
            hand_size = len(raw_obs.get('hand', []))
            next_hand_size = len(next_raw_obs.get('hand', []))
            
            # Reward for reducing hand size
            if next_hand_size < hand_size:
                shaped_reward += 0.1
            
            # Small penalty for drawing
            if action == 60:
                shaped_reward -= 0.05
            
            # Bonus for playing action cards when opponent is close
            num_cards = raw_obs.get('num_cards', [7, 7])
            if len(num_cards) > 1:
                min_opponent = min(num_cards[1:])
                if min_opponent <= 2:
                    if action in [10,11,12, 25,26,27, 40,41,42, 55,56,57]:
                        shaped_reward += 0.05
        
        return shaped_reward
    
    def update_epsilon(self, episode):
        """Epsilon decay schedule"""
        self.episode_count = episode
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-episode / self.epsilon_decay)
    
    def eval_mode(self):
        """Switch to evaluation mode"""
        self.training_mode = False
        self.epsilon = 0.0
    
    def train_mode(self):
        """Switch to training mode"""
        self.training_mode = True
        self.epsilon = max(self.epsilon, 0.1)
    
    def get_stats(self):
        """Get comprehensive training statistics"""
        avg_reward = np.mean(self.reward_history[-1000:]) if len(self.reward_history) > 100 else 0
        
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'avg_recent_reward': avg_reward,
            'q_table_growth': self.q_table_size_history,
            'non_zero_q_values': sum(1 for q in self.q_table.values() if abs(q) > 0.001)
        }
    
    def save_model(self, filepath):
        """Save Q-table and parameters"""
        import pickle
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'lr': self.lr,
            'gamma': self.gamma
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load Q-table and parameters"""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.epsilon = model_data['epsilon'] 
        self.episode_count = model_data['episode_count']
        self.lr = model_data.get('lr', self.lr)
        self.gamma = model_data.get('gamma', self.gamma)
        print(f"Model loaded from {filepath}")
        print(f"Q-table size: {len(self.q_table)}")