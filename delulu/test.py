from cardMapper import CardMapper
from featureExtractor import UnoFeatureExtractor
from env import UnoEnv

mapper = CardMapper()

env = UnoEnv()
state, info = env.reset()


print("\n=== TESTING CARDMAPPER ===")
hand_cards = ['r-wild', 'b-7', 'y-4', 'b-reverse', 'g-7', 'r-1', 'b-8']
target_card = 'y-1'

print(f"Target: {target_card} -> {mapper.parse_card(target_card)}")

for card in hand_cards:
    parsed = mapper.parse_card(card)
    print(f"Card: {card} -> {parsed}")

# Test feature extraction
extractor = UnoFeatureExtractor()
features = extractor.extract_features(state)

print(f"\n=== FEATURE EXTRACTION ===")
print(f"Features length: {len(features)}")
print(f"First 20 features: {features[:20]}")
print(f"Non-zero features: {[(i, f) for i, f in enumerate(features) if f != 0]}")

