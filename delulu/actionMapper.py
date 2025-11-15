class UnoActionMapper:
    """Maps RLCard card names to action indices"""
    
    def __init__(self):
        # Zgodnie z tabelą z obrazu
        self.card_to_action = {}
        
        # Red cards (0-14)
        for i in range(10):
            self.card_to_action[f'r-{i}'] = i
        self.card_to_action['r-skip'] = 10
        self.card_to_action['r-reverse'] = 11  
        self.card_to_action['r-draw_2'] = 12
        self.card_to_action['r-wild'] = 13  # Wild pokazywane jako czerwone
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
        
        # Draw action
        # Action 60 = draw (nie ma karty w ręce)
    
    def get_action_for_card(self, card_name):
        """Get action index for card name"""
        return self.card_to_action.get(card_name, None)
    
    def get_playable_cards_from_hand(self, hand, legal_actions):
        """Get which cards from hand are actually playable"""
        playable = []
        for card in hand:
            action_idx = self.get_action_for_card(card)
            if action_idx and action_idx in legal_actions:
                playable.append(card)
        return playable