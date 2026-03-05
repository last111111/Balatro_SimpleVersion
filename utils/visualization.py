# utils/visualization.py
# -*- coding: utf-8 -*-
"""游戏可视化工具"""


class GameVisualizer:

    SUITS = {'H': '\u2665', 'D': '\u2666', 'C': '\u2663', 'S': '\u2660'}
    SUIT_COLORS = {'H': '\033[91m', 'D': '\033[91m', 'C': '\033[90m', 'S': '\033[90m'}
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BG_DARK = '\033[48;5;236m'

    @staticmethod
    def card_to_str(card):
        rank, suit = card
        rank_str = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}.get(rank, str(rank))
        suit_symbol = GameVisualizer.SUITS[suit]
        color = GameVisualizer.SUIT_COLORS[suit]
        return f"{color}{GameVisualizer.BOLD} {rank_str}{suit_symbol} {GameVisualizer.RESET}"

    @staticmethod
    def cards_to_str(cards):
        if not cards:
            return "[ empty ]"
        return "[ " + "  ".join(GameVisualizer.card_to_str(card) for card in cards) + " ]"

    @staticmethod
    def print_separator(char="═", length=90):
        print(char * length)

    @staticmethod
    def print_header(text):
        GameVisualizer.print_separator()
        padded = f"  {text}  "
        print(f"{GameVisualizer.BOLD}{GameVisualizer.CYAN}{padded:═^90}{GameVisualizer.RESET}")
        GameVisualizer.print_separator()

    @staticmethod
    def print_game_state(env, step_num, total_reward):
        print(f"\n{'─'*90}")
        print(f"  {GameVisualizer.YELLOW}{GameVisualizer.BOLD}[ Step {step_num} ]{GameVisualizer.RESET}")
        print()
        print(f"  🃏  Hand:          {GameVisualizer.cards_to_str(env.hand)}")
        print(f"  ▶   Plays left:    {GameVisualizer.GREEN}{GameVisualizer.BOLD}{env.play_count} / {env.max_play}{GameVisualizer.RESET}")
        print(f"  ✖   Discards left: {GameVisualizer.BLUE}{GameVisualizer.BOLD}{env.discard_count} / {env.max_discard}{GameVisualizer.RESET}")
        print(f"  ★   Score:         {GameVisualizer.BOLD}{GameVisualizer.YELLOW}{total_reward:.1f}{GameVisualizer.RESET}")

        if hasattr(env, 'jokers') and env.jokers:
            from utils.chatgpt_reward import JOKER_DESCRIPTIONS
            joker_names = []
            for j in env.jokers:
                jid = j.joker_type
                desc = JOKER_DESCRIPTIONS.get(jid, f"#{jid}")
                name = desc.split(" - ")[0]
                joker_names.append(name)
            print(f"  🎭  Jokers:        {GameVisualizer.MAGENTA}{GameVisualizer.BOLD}{', '.join(joker_names)}{GameVisualizer.RESET}")

    @staticmethod
    def print_action(action_type, selected_cards, reward, hand_name=None):
        if action_type == 1:
            label = f"{GameVisualizer.GREEN}{GameVisualizer.BOLD}▶ PLAY{GameVisualizer.RESET}"
        else:
            label = f"{GameVisualizer.BLUE}{GameVisualizer.BOLD}✖ DISCARD{GameVisualizer.RESET}"

        print(f"\n  {label}:  {GameVisualizer.cards_to_str(selected_cards)}")
        if hand_name and action_type == 1:
            print(f"         Hand type:  {GameVisualizer.GREEN}{GameVisualizer.BOLD}{hand_name}{GameVisualizer.RESET}")
        sign = "+" if reward >= 0 else ""
        print(f"         Reward:     {GameVisualizer.YELLOW}{GameVisualizer.BOLD}{sign}{reward:.1f}{GameVisualizer.RESET}")
