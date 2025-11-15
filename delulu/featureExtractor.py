from delulu.cardMapper import CardMapper
from delulu.actionMapper import UnoActionMapper

class UnoFeatureExtractor:
    """FIXED: Feature extraction that correctly maps cards to actions"""
    
    def __init__(self):
        self.card_mapper = CardMapper()  # Keep for parsing
        self.action_mapper = UnoActionMapper()  # NEW: For action mapping
    
    def extract_features(self, state):
        """Extract features that correctly relate to available actions"""
        if not state or 'raw_obs' not in state:
            return tuple([0] * 30)  # Reduced feature set
        
        raw_obs = state['raw_obs']
        hand = raw_obs.get('hand', [])
        target = raw_obs.get('target', '')
        num_cards = raw_obs.get('num_cards', [7, 7])
        legal_actions = state.get('legal_actions', [])
        
        features = []
        
        # 1. Hand composition (10 features)
        color_counts = {'red': 0, 'green': 0, 'blue': 0, 'yellow': 0}
        action_counts = {'skip': 0, 'reverse': 0, 'draw': 0, 'wild': 0}
        number_count = 0
        
        for card in hand:
            color, card_type, value = self.card_mapper.parse_card(card)
            if color:
                color_counts[color] += 1
            if card_type == 'number':
                number_count += 1
            elif card_type in action_counts:
                action_counts[card_type] += 1
        
        features.extend([min(color_counts[c], 5) for c in ['red', 'green', 'blue', 'yellow']])
        features.extend([min(action_counts[c], 3) for c in ['skip', 'reverse', 'draw', 'wild']])
        features.append(min(number_count, 7))
        features.append(min(len(hand), 10))
        
        # 2. Playability - KEY FIX! (8 features)
        playable_cards = self.action_mapper.get_playable_cards_from_hand(hand, legal_actions)
        
        playable_colors = {'red': 0, 'green': 0, 'blue': 0, 'yellow': 0}
        playable_actions = 0
        playable_numbers = 0
        
        for card in playable_cards:
            color, card_type, value = self.card_mapper.parse_card(card)
            if color:
                playable_colors[color] += 1
            if card_type == 'number':
                playable_numbers += 1
            elif card_type in ['skip', 'reverse', 'draw']:
                playable_actions += 1
        
        features.extend([min(playable_colors[c], 3) for c in ['red', 'green', 'blue', 'yellow']])
        features.append(min(playable_numbers, 5))
        features.append(min(playable_actions, 3))
        features.append(1 if 60 in legal_actions else 0)  # Can draw
        features.append(min(len(playable_cards), 7))
        
        # 3. Game state (5 features)
        target_color, target_type, target_value = self.card_mapper.parse_card(target)
        
        # Target color (4 features - one hot)
        target_color_onehot = [0, 0, 0, 0]
        if target_color:
            color_idx = self.card_mapper.get_color_index(target_color)
            if 0 <= color_idx < 4:
                target_color_onehot[color_idx] = 1
        features.extend(target_color_onehot)
        
        # Target is action card
        features.append(1 if target_type in ['skip', 'reverse', 'draw'] else 0)
        
        # 4. Opponent info (7 features)
        if len(num_cards) >= 2:
            opponent_cards = num_cards[1:]
            features.append(min(opponent_cards[0], 10))  # First opponent
            features.append(min(min(opponent_cards), 10))  # Min opponent
            features.append(min(sum(opponent_cards), 20))  # Total opponent
            features.append(1 if any(c == 1 for c in opponent_cards) else 0)  # Uno warning
            
            # Game phase
            total_cards = sum(num_cards)
            if total_cards <= 15:
                features.extend([0, 0, 1])  # Late
            elif total_cards <= 25:
                features.extend([0, 1, 0])  # Mid  
            else:
                features.extend([1, 0, 0])  # Early
        else:
            features.extend([7, 7, 14, 0, 1, 0, 0])  # Default values
        
        return tuple(features[:30])  # Exactly 30 features