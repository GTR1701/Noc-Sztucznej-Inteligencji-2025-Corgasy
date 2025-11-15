import gymnasium as gym
import rlcard
import numpy as np


class UnoEnv(gym.Env):

    def __init__(self, render_mode=None):

        assert render_mode in (
            None,
            "human",
        ), "render_mode must be either None or 'human'"

        self.env = rlcard.make("uno", config={"allow_step_back": False})
        self.action_space = gym.spaces.Discrete(self.env.num_actions)
        self.render_mode = render_mode
        self.num_players = self.env.num_players

        # Update observation space to Dict to be more accurate
        # Agents can now handle the raw state themselves
        self.observation_space = gym.spaces.Dict({})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, player_id = self.env.reset()

        if self.render_mode == "human":
            self._render_state(state, player_id, None, "GAME START")

        # Convert OrderedDict to regular dict if present
        if 'legal_actions' in state and hasattr(state['legal_actions'], 'keys'):
            state['legal_actions'] = dict(state['legal_actions'])

        info = self._get_info(player_id)
        return state, info  # Return raw state

    def step(self, action):
        prev_player_id = self.env.get_player_id()

        # --- Action Validation ---
        legal_actions = list(self.env._get_legal_actions().keys())
        if action not in legal_actions:
            if 60 in legal_actions:  # 60 is 'draw'
                action = 60
            else:
                action = np.random.choice(legal_actions)

        # --- Perform the Step ---
        next_state, next_player_id = self.env.step(action)

        # Convert OrderedDict to regular dict if present
        if 'legal_actions' in next_state and hasattr(next_state['legal_actions'], 'keys'):
            next_state['legal_actions'] = dict(next_state['legal_actions'])

        if self.render_mode == "human":
            self._render_state(
                next_state, next_player_id, action, f"Player {prev_player_id} played"
            )

        done = self.env.is_over()

        reward = 0
        if done:
            payoffs = self.env.get_payoffs()
            reward = payoffs[0]  # Return Player 0's payoff

        info = self._get_info(next_player_id, done)

        return next_state, reward, done, False, info  # Return raw state

    def _get_info(self, player_id, done=False):
        info = {"player_id": player_id, "legal_actions": []}
        if not done:
            info["legal_actions"] = list(self.env._get_legal_actions().keys())
        return info

    def get_current_player(self):
        return self.env.get_player_id()

    #
    # --- RENDERING METHODS ---
    #

    def _render_state(self, state, player_id, action, message):
        print("\n" + "┏" + "━" * 78 + "┓")
        print(f"┃ {message:^76} ┃")
        print("┗" + "━" * 78 + "┛")

        if action is not None:
            card_str = self._render_card(action)
            print(f"\n   Last played:")
            for line in card_str:
                print(f"   {line}")

        if "obs" in state and "target" in state["obs"]:
            print(f"\n   Card on table:")
            card_str = self._render_card(state["obs"]["target"])
            for line in card_str:
                print(f"   {line}")

        if "obs" in state and "hand" in state["obs"]:
            hand = state["obs"]["hand"]
            print(f"\n   Player {player_id}'s hand ({len(hand)} cards):")
            self._render_hand(hand)

        legal_actions = list(self.env._get_legal_actions().keys())
        print(
            f"\n   Legal moves for Player {player_id} ({len(legal_actions)} options):"
        )
        self._render_hand(legal_actions)
        print("\n" + "─" * 80)

    def _render_card(self, action):
        color, text = self._get_card_info(action)
        colors_map = {
            "RED": "\033[91m",
            "GREEN": "\033[92m",
            "BLUE": "\033[94m",
            "YELLOW": "\033[93m",
            "WILD": "\033[95m",
            "RESET": "\033[0m",
        }
        c = colors_map.get(color, "")
        r = colors_map["RESET"]
        return [
            f"{c}┌─────────┐{r}",
            f"{c}│ {text:^7} │{r}",
            f"{c}│         │{r}",
            f"{c}│    {color[0]}    │{r}",
            f"{c}│         │{r}",
            f"{c}│ {text:^7} │{r}",
            f"{c}└─────────┘{r}",
        ]

    def _render_hand(self, cards):
        if not cards:
            print("     (empty)")
            return

        cards_per_row = 8
        for i in range(0, len(cards), cards_per_row):
            row_cards = cards[i : i + cards_per_row]
            card_renders = [self._render_card(card) for card in row_cards]

            for line_idx in range(7):
                line = "   "
                for card_lines in card_renders:
                    line += card_lines[line_idx] + " "
                print(line)
            print()

    def _get_card_info(self, action):
        colors = ["RED", "GREEN", "BLUE", "YELLOW"]

        if action < 40:  # Number cards
            return colors[action // 10], str(action % 10)
        elif action < 52:  # Action cards
            action_type = (action - 40) // 4
            color_idx = (action - 40) % 4
            if action_type == 0:
                return colors[color_idx], "SKIP"
            elif action_type == 1:
                return colors[color_idx], "REVERSE"
            elif action_type == 2:
                return colors[color_idx], "DRAW 2"
        elif action < 56:  # Wild cards
            return "WILD", "WILD"
        elif action < 60:  # Wild Draw 4
            return "WILD", "DRAW 4"
        elif action == 60:  # Draw action
            return "WILD", "DRAW"
        return "WILD", f"#{action}"

    def _action_to_card(self, action):
        color, text = self._get_card_info(action)
        return text if color == "WILD" else f"{color[0]}{text}"