from delulu.featureExtractor import UnoFeatureExtractor

def test_feature_extraction():
    """Test feature extraction with sample game states"""
    
    # Mock state dla testowania
    test_state = {
        'raw_obs': {
            'hand': ['r-5', 'b-3', 'skip-g', 'wild', 'y-7'],
            'target': 'r-8', 
            'num_cards': [5, 3, 6, 2]  # me, opp1, opp2, opp3
        },
        'legal_actions': [5, 23, 45, 52]  # Przykładowe legalne akcje
    }
    
    extractor = UnoFeatureExtractor()
    
    # Test extraction
    import time
    start_time = time.time()
    
    features = extractor.extract_features(test_state)
    
    extraction_time = time.time() - start_time
    
    print(f"Feature extraction time: {extraction_time*1000:.2f} ms")
    print(f"Features length: {len(features)}")
    print(f"Features: {features[:10]}...")  # Pokazuj pierwsze 10
    
    # Test czy features są hashable
    try:
        state_hash = hash(features)
        print(f"State successfully hashed: {state_hash}")
    except Exception as e:
        print(f"Hashing failed: {e}")
    
    # Test multiple extractions dla performance
    num_tests = 1000
    start_time = time.time()
    
    for _ in range(num_tests):
        features = extractor.extract_features(test_state)
    
    avg_time = (time.time() - start_time) / num_tests
    print(f"Average extraction time over {num_tests} runs: {avg_time*1000:.4f} ms")
    
    # Sprawdź czy mieszczemy się w 3s limicie
    max_extractions_per_3s = 3.0 / avg_time
    print(f"Max extractions per 3 seconds: {max_extractions_per_3s:.0f}")

# test_feature_extraction()