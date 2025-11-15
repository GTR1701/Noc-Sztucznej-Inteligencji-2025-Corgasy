class CardMapper:
    """Maps RLCard Uno card names to properties"""
    
    def __init__(self):
        self.colors = {'r': 'red', 'g': 'green', 'b': 'blue', 'y': 'yellow'}
        self.color_indices = {'red': 0, 'green': 1, 'blue': 2, 'yellow': 3}
        
        # Card type mappings
        self.card_types = {
            'number': 0, 'skip': 1, 'reverse': 2, 'draw': 3, 'wild': 4, 'wild_draw': 5
        }
    
    def parse_card(self, card_name):
        """
        Parse RLCard Uno card name formats:
        - Number cards: 'r-1', 'y-4', 'b-0' 
        - Action cards: 'b-reverse', 'g-skip', 'r-draw_2'
        - Wild cards: 'r-wild', 'wild_draw_4'
        Returns: (color, card_type, number/value)
        """
        if not card_name:
            return (None, 'unknown', None)
            
        card_name = str(card_name).lower()
        
        # Handle wild_draw_4 (no color prefix)
        if card_name == 'wild_draw_4':
            return (None, 'wild_draw', 4)
        
        # Split by dash
        if '-' in card_name:
            parts = card_name.split('-', 1)  # Split only on first dash
            if len(parts) != 2:
                return (None, 'unknown', None)
                
            color_char, card_part = parts
            color = self.colors.get(color_char)
            
            # Number cards: 'r-1', 'y-4', 'b-0'
            if card_part.isdigit():
                number = int(card_part)
                return (color, 'number', number)
            
            # Action cards: 'b-reverse', 'g-skip' 
            elif card_part in ['skip', 'reverse']:
                return (color, card_part, None)
            
            # Draw 2 cards: 'r-draw_2'
            elif card_part == 'draw_2':
                return (color, 'draw', 2)
                
            # Wild cards with color: 'r-wild'
            elif card_part == 'wild':
                return (None, 'wild', None)  # Wild cards have no effective color
                
        # Fallback for unknown formats
        return (None, 'unknown', None)
    
    def get_color_index(self, color):
        """Get numeric color index"""
        return self.color_indices.get(color, -1)
    
    def get_type_index(self, card_type):
        """Get numeric card type index"""
        return self.card_types.get(card_type, -1)