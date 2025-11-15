from env import UnoEnv

env = UnoEnv()
state, info = env.reset()

print("=== ACTUAL RLCARD STATE FORMAT ===")
print(f"State keys: {list(state.keys())}")
print(f"Raw obs keys: {list(state.get('raw_obs', {}).keys())}")
print(f"Hand: {state.get('raw_obs', {}).get('hand', 'NOT_FOUND')}")
print(f"Target: {state.get('raw_obs', {}).get('target', 'NOT_FOUND')}")
print(f"Num cards: {state.get('raw_obs', {}).get('num_cards', 'NOT_FOUND')}")
print("===============================")
