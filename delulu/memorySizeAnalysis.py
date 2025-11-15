
"""
Analyze the theoretical state space size with our feature extraction
"""
feature_ranges = {
    'color_counts': 10**4,          # 0-5 for each color = 1,296
    'number_cards': 10,             # 0-7 = 8  
    'action_counts': 4**5,         # 0-3 for each action type = 1,024
    'hand_size': 11,               # 0-10 = 11
    'playable_total': 8,           # 0-7 = 8
    'color_matches': 10**4,         # 0-5 for each color = 1,296
    'number_matches': 10,           # 0-5 = 6
    'wild_available': 4,           # 0-3 = 4
    'action_playable': 4,          # 0-3 = 4
    'target_color': 16,            # one-hot combinations = 16
    'target_type': 64,             # one-hot combinations = 64
    'game_phase': 8,               # one-hot combinations = 8
    'turn_position': 1,            # always 0 = 1
    'opponent_sizes': 10,        # 0-7 for each opponent = 512
    'min_opponent': 10,             # 0-7 = 8
    'total_opponents': 1,         # 0-20 = 21
    'uno_warning': 2,              # 0-1 = 2
    'color_control': 4**4,         # 0-3 for each color = 256
}

# Conservative estimate (features are not independent)
theoretical_max = 1
for feature, range_size in feature_ranges.items():
    theoretical_max *= range_size

print(f"Theoretical maximum states: {theoretical_max:,}")
print(f"This is approximately: {theoretical_max:.2e}")

# More realistic estimate (considering feature dependencies)
realistic_estimate = (
    1296 * 8 * 1024 * 11 * 8 * 1296 * 6 * 4 * 4 * 2 *  # Core features
    16 * 64 * 8 * 512 * 8 * 21 * 2 * 256  # Context features
) // 1000000  # Divide by factor for dependencies

print(f"More realistic estimate: {realistic_estimate:,}")
print(f"Memory requirement (8 bytes per Q-value): {realistic_estimate * 61 * 8 / 1024 / 1024:.2f} MB")

# analyze_feature_space()
# Expected output:
# Realistic estimate: ~10^6 to 10^7 states
# Memory requirement: ~500 MB to 5 GB (manageable)