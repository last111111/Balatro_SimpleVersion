# game_gui.py
# -*- coding: utf-8 -*-
"""Balatro 简化版游戏 GUI — 人工游玩 / AI 游玩"""

import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
import numpy as np
import torch
import os
import glob

from envs.joint_env import JointEnv
from envs.joker_env import NUM_JOKER_TYPES, MAX_HELD, NUM_ROUNDS, NUM_OFFERED
from utils.chatgpt_reward import JOKER_DESCRIPTIONS

# ── 常量 ──────────────────────────────────────────────

SUIT_MAP = {'H': 0, 'D': 1, 'C': 2, 'S': 3}
SUIT_SYMBOL = {'H': '\u2665', 'D': '\u2666', 'C': '\u2663', 'S': '\u2660'}
RANK_STR = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}

BG = '#FFFFFF'
FG = '#000000'
CARD_BG = '#FFFFFF'
CARD_SEL = '#BBBBBB'
BTN_BG = '#E0E0E0'
HEADER_BG = '#F0F0F0'

FONT_TITLE = ('Courier', 22, 'bold')
FONT_HEADER = ('Courier', 13, 'bold')
FONT_BODY = ('Courier', 11)
FONT_CARD = ('Courier', 14, 'bold')
FONT_SMALL = ('Courier', 10)
FONT_BTN = ('Courier', 12, 'bold')


def card_str(card):
    rank, suit = card
    r = RANK_STR.get(rank, str(rank))
    return f"{r}{SUIT_SYMBOL[suit]}"


def card_index(card):
    rank, suit = card
    return (rank - 1) * 4 + SUIT_MAP[suit]


def sort_hand(hand):
    return sorted(hand, key=lambda c: (c[0], SUIT_MAP[c[1]]))


def parse_joker(jid):
    desc = JOKER_DESCRIPTIONS.get(jid, f"Unknown #{jid}")
    parts = desc.split(' - ', 1)
    name = parts[0]
    effect = parts[1] if len(parts) > 1 else ''
    return name, effect


# ── 主应用 ────────────────────────────────────────────

class BalatroGUI:

    def __init__(self, root):
        self.root = root
        self.root.title('Balatro - Simplified Version')
        self.root.geometry('960x720')
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # 游戏状态
        self.mode = None          # 'human' or 'ai'
        self.joint_env = None
        self.current_round = 0
        self.round_scores = []
        self.action_log = []
        self.step_count = 0

        # 打牌界面的选中状态
        self.selected_cards = set()
        self.card_buttons = []

        # 小丑牌替换状态
        self._pending_offer_idx = None

        # AI 模式
        self.card_model = None
        self.joker_model = None
        self.card_wrapper = None
        self.ai_speed = 800
        self.joker_obs = None

        self._show_mode_select()

    # ── 工具方法 ──────────────────────────────────────

    def _clear(self):
        for w in self.root.winfo_children():
            w.destroy()

    def _make_header(self, parent, text):
        f = tk.Frame(parent, bg=HEADER_BG, bd=1, relief=tk.GROOVE)
        f.pack(fill=tk.X, padx=5, pady=(5, 0))
        tk.Label(f, text=text, font=FONT_HEADER, bg=HEADER_BG, fg=FG).pack(pady=4)
        return f

    # ── 界面 1: 模式选择 ─────────────────────────────

    def _show_mode_select(self):
        self._clear()
        self.mode = None

        frame = tk.Frame(self.root, bg=BG)
        frame.pack(expand=True)

        tk.Label(frame, text='BALATRO', font=FONT_TITLE, bg=BG, fg=FG).pack(pady=(40, 5))
        tk.Label(frame, text='Simplified Version', font=FONT_HEADER, bg=BG, fg='#666666').pack(pady=(0, 30))

        btn_frame = tk.Frame(frame, bg=BG)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text='  Human Play  ', font=FONT_BTN,
                  bg=BTN_BG, fg=FG, relief=tk.RAISED, bd=2,
                  command=self._start_human).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text='   AI Play    ', font=FONT_BTN,
                  bg=BTN_BG, fg=FG, relief=tk.RAISED, bd=2,
                  command=self._start_ai_dialog).pack(side=tk.LEFT, padx=20)

        # AI checkpoint 选择区
        ai_frame = tk.LabelFrame(frame, text='AI Settings', font=FONT_BODY,
                                 bg=BG, fg=FG, bd=1, padx=10, pady=10)
        ai_frame.pack(pady=30, padx=40, fill=tk.X)

        # Card agent checkpoint
        row_card = tk.Frame(ai_frame, bg=BG)
        row_card.pack(fill=tk.X, pady=2)
        tk.Label(row_card, text='Card Agent: ', font=FONT_BODY, bg=BG, fg=FG).pack(side=tk.LEFT)
        self.card_ckpt_var = tk.StringVar(value=self._find_latest_card_ckpt())
        tk.Entry(row_card, textvariable=self.card_ckpt_var, font=FONT_SMALL, width=45).pack(side=tk.LEFT, padx=5)
        tk.Button(row_card, text='Browse', font=FONT_SMALL, bg=BTN_BG,
                  command=lambda: self._browse_ckpt_to(self.card_ckpt_var)).pack(side=tk.LEFT)

        # Joker agent checkpoint
        row_joker = tk.Frame(ai_frame, bg=BG)
        row_joker.pack(fill=tk.X, pady=2)
        tk.Label(row_joker, text='Joker Agent:', font=FONT_BODY, bg=BG, fg=FG).pack(side=tk.LEFT)
        self.joker_ckpt_var = tk.StringVar(value=self._find_latest_joker_ckpt())
        tk.Entry(row_joker, textvariable=self.joker_ckpt_var, font=FONT_SMALL, width=45).pack(side=tk.LEFT, padx=5)
        tk.Button(row_joker, text='Browse', font=FONT_SMALL, bg=BTN_BG,
                  command=lambda: self._browse_ckpt_to(self.joker_ckpt_var)).pack(side=tk.LEFT)

        tk.Label(ai_frame, text='(Leave Joker Agent empty to skip joker selection)',
                 font=FONT_SMALL, bg=BG, fg='#666666').pack(anchor=tk.W, pady=(0, 2))

        row2 = tk.Frame(ai_frame, bg=BG)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text='Speed:', font=FONT_BODY, bg=BG, fg=FG).pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=800)
        for text, val in [('Fast', 200), ('Normal', 800), ('Slow', 2000)]:
            tk.Radiobutton(row2, text=text, variable=self.speed_var, value=val,
                           font=FONT_SMALL, bg=BG, fg=FG).pack(side=tk.LEFT, padx=8)

    def _find_latest_card_ckpt(self):
        """查找最新的 card agent checkpoint"""
        for d in ['outputs/joint/checkpoints', 'outputs/card/checkpoints', 'outputs/checkpoints']:
            files = glob.glob(os.path.join(d, '*_final.pt'))
            if not files:
                files = glob.glob(os.path.join(d, '*.pt'))
            if files:
                return max(files, key=os.path.getmtime)
        # 也检查根目录的 ppo_balatro.pt
        if os.path.isfile('ppo_balatro.pt'):
            return 'ppo_balatro.pt'
        return ''

    def _find_latest_joker_ckpt(self):
        """查找最新的 joker agent checkpoint"""
        for d in ['outputs/joker/checkpoints']:
            files = glob.glob(os.path.join(d, '*_final.pt'))
            if not files:
                files = glob.glob(os.path.join(d, '*.pt'))
            if files:
                return max(files, key=os.path.getmtime)
        return ''

    def _browse_ckpt_to(self, var):
        path = filedialog.askopenfilename(
            title='Select Checkpoint',
            filetypes=[('PyTorch', '*.pt'), ('All', '*.*')],
            initialdir='outputs/'
        )
        if path:
            var.set(path)

    # ── 人工模式启动 ─────────────────────────────────

    def _start_human(self):
        self.mode = 'human'
        self.joint_env = JointEnv(max_hand_size=8, max_play=5, shaping_beta=0.0)
        self.round_scores = []
        self.current_round = 0
        self.joker_obs = self.joint_env.reset()
        self._show_joker_select()

    # ── AI 模式启动 ──────────────────────────────────

    def _start_ai_dialog(self):
        card_path = self.card_ckpt_var.get().strip()
        joker_path = self.joker_ckpt_var.get().strip()

        if not card_path or not os.path.isfile(card_path):
            messagebox.showerror('Error', 'Please select a valid Card Agent checkpoint.')
            return

        self.ai_speed = self.speed_var.get()
        self._start_ai(card_path, joker_path if joker_path and os.path.isfile(joker_path) else None)

    def _start_ai(self, card_ckpt_path, joker_ckpt_path=None):
        self.mode = 'ai'
        from models.card_agent import ActorCritic
        from models.joker_agent import JokerSelectNet
        from evaluation.eval_joint import _CardAgentWrapper

        card_ckpt = torch.load(card_ckpt_path, map_location='cpu')
        card_config = card_ckpt.get('config', {})

        # Joint checkpoint 兼容: card_state_dict + joker_state_dict 在同一文件
        if 'card_state_dict' in card_ckpt:
            obs_dim = card_config.get('card_obs_dim', 226)
            max_hand = card_config.get('max_hand_size', 8)
            card_sd = card_ckpt['card_state_dict']
            # 如果没有单独的 joker checkpoint，从 joint 里取
            if joker_ckpt_path is None and 'joker_state_dict' in card_ckpt:
                joker_ckpt_path = '__from_joint__'
        elif 'state_dict' in card_ckpt:
            obs_dim = card_config.get('obs_dim', 226)
            max_hand = card_config.get('max_hand_size', 8)
            card_sd = card_ckpt['state_dict']
        else:
            messagebox.showerror('Error', 'Card checkpoint has no state_dict or card_state_dict.')
            return

        self.joint_env = JointEnv(max_hand_size=max_hand, max_play=card_config.get('max_play', 5),
                                  shaping_beta=0.0)

        # 加载 Card Agent
        self.card_model = ActorCritic(obs_dim, max_hand)
        self.card_model.load_state_dict(card_sd)
        self.card_model.eval()
        self.card_wrapper = _CardAgentWrapper(self.card_model, torch.device('cpu'))

        # 加载 Joker Agent
        self.joker_model = None
        if joker_ckpt_path == '__from_joint__':
            joker_obs_dim = card_config.get('joker_obs_dim', 41)
            self.joker_model = JokerSelectNet(obs_dim=joker_obs_dim)
            self.joker_model.load_state_dict(card_ckpt['joker_state_dict'])
            self.joker_model.eval()
        elif joker_ckpt_path is not None:
            joker_ckpt = torch.load(joker_ckpt_path, map_location='cpu')
            joker_config = joker_ckpt.get('config', {})
            joker_obs_dim = joker_config.get('obs_dim', 41)
            self.joker_model = JokerSelectNet(obs_dim=joker_obs_dim)
            if 'joker_state_dict' in joker_ckpt:
                self.joker_model.load_state_dict(joker_ckpt['joker_state_dict'])
            elif 'state_dict' in joker_ckpt:
                self.joker_model.load_state_dict(joker_ckpt['state_dict'])
            else:
                messagebox.showerror('Error', 'Joker checkpoint has no recognized state_dict.')
                return
            self.joker_model.eval()

        self.round_scores = []
        self.current_round = 0
        self.joker_obs = self.joint_env.reset()
        self._show_joker_select()

    # ── 界面 2: 小丑牌选择 ───────────────────────────

    def _show_joker_select(self):
        self._clear()
        self._pending_offer_idx = None

        env = self.joint_env.joker_env
        total_prev = sum(self.round_scores)

        # 顶栏
        top = tk.Frame(self.root, bg=HEADER_BG, bd=1, relief=tk.GROOVE)
        top.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(top, text=f'Round {self.current_round + 1}/{NUM_ROUNDS}',
                 font=FONT_HEADER, bg=HEADER_BG, fg=FG).pack(side=tk.LEFT, padx=10)
        tk.Label(top, text=f'Total Score: {total_prev:.0f}',
                 font=FONT_HEADER, bg=HEADER_BG, fg=FG).pack(side=tk.RIGHT, padx=10)

        main = tk.Frame(self.root, bg=BG)
        main.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # 已持有
        held_frame = tk.LabelFrame(main, text=f'Your Jokers ({len(env.held)}/{MAX_HELD})',
                                   font=FONT_HEADER, bg=BG, fg=FG, padx=10, pady=5)
        held_frame.pack(fill=tk.X, pady=5)

        self._held_labels = []
        for i in range(MAX_HELD):
            f = tk.Frame(held_frame, bg=BG, bd=1, relief=tk.RIDGE, padx=5, pady=3)
            f.pack(fill=tk.X, pady=1)
            if i < len(env.held):
                jid = env.held[i]
                name, effect = parse_joker(jid)
                txt = f'[{i+1}] {name} — {effect}'
                lbl = tk.Label(f, text=txt, font=FONT_BODY, bg=BG, fg=FG, anchor=tk.W)
                lbl.pack(fill=tk.X)
                if len(env.held) == MAX_HELD:
                    lbl.bind('<Button-1>', lambda e, idx=i: self._on_held_click(idx))
                    lbl.configure(cursor='hand2')
            else:
                tk.Label(f, text=f'[{i+1}] (empty)', font=FONT_BODY,
                         bg=BG, fg='#999999', anchor=tk.W).pack(fill=tk.X)
            self._held_labels.append(f)

        # 提供的小丑牌
        offer_frame = tk.LabelFrame(main, text='Offered This Round',
                                    font=FONT_HEADER, bg=BG, fg=FG, padx=10, pady=5)
        offer_frame.pack(fill=tk.X, pady=5)

        for i, jid in enumerate(env.offered):
            row = tk.Frame(offer_frame, bg=BG, padx=5, pady=2)
            row.pack(fill=tk.X)
            name, effect = parse_joker(jid)
            tk.Label(row, text=f'  {name} — {effect}',
                     font=FONT_BODY, bg=BG, fg=FG, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

            if self.mode == 'human':
                if len(env.held) < MAX_HELD:
                    tk.Button(row, text='Pick', font=FONT_SMALL, bg=BTN_BG, width=6,
                              command=lambda idx=i: self._handle_joker_pick(idx)).pack(side=tk.RIGHT)
                else:
                    tk.Button(row, text='Select', font=FONT_SMALL, bg=BTN_BG, width=6,
                              command=lambda idx=i: self._on_offer_click(idx)).pack(side=tk.RIGHT)

        # 按钮
        btn_frame = tk.Frame(main, bg=BG)
        btn_frame.pack(pady=15)

        if self.mode == 'human':
            tk.Button(btn_frame, text='  Skip This Round  ', font=FONT_BTN,
                      bg=BTN_BG, fg=FG, command=self._handle_joker_skip).pack()
            if len(env.held) == MAX_HELD:
                tk.Label(main, text='(Click "Select" on an offered joker, then click a held joker to replace)',
                         font=FONT_SMALL, bg=BG, fg='#666666').pack()

        # 替换提示
        self._replace_hint = tk.Label(main, text='', font=FONT_BODY, bg=BG, fg='#CC0000')
        self._replace_hint.pack()

        # AI 模式自动操作
        if self.mode == 'ai':
            self.root.after(self.ai_speed, self._ai_joker_step)

    def _handle_joker_pick(self, offered_idx):
        action = offered_idx + 1
        self._execute_joker(action)

    def _handle_joker_skip(self):
        self._execute_joker(0)

    def _on_offer_click(self, offered_idx):
        """选中要拿的 offered joker，等待点击 held slot 替换"""
        self._pending_offer_idx = offered_idx
        name, _ = parse_joker(self.joint_env.joker_env.offered[offered_idx])
        self._replace_hint.config(text=f'Now click a held joker to replace with "{name}"')

    def _on_held_click(self, held_idx):
        """点击已持有的 joker slot 进行替换"""
        if self._pending_offer_idx is None:
            return
        offer_idx = self._pending_offer_idx
        action = 5 + offer_idx * MAX_HELD + held_idx
        self._execute_joker(action)

    def _execute_joker(self, action):
        self.joker_obs, joker_done, info = self.joint_env.joker_step(action)
        self._start_card_play()

    # ── 界面 3: 打牌 ────────────────────────────────

    def _start_card_play(self):
        held_ids = list(self.joint_env.joker_env.held)
        self.joint_env.card_env.reset_with_jokers(held_ids)
        self.selected_cards = set()
        self.action_log = []
        self.step_count = 0
        self._show_card_play()

    def _show_card_play(self):
        self._clear()
        env = self.joint_env.card_env

        # 顶栏
        top = tk.Frame(self.root, bg=HEADER_BG, bd=1, relief=tk.GROOVE)
        top.pack(fill=tk.X, padx=5, pady=(5, 0))
        info_text = (f'Round {self.current_round + 1}/{NUM_ROUNDS}  |  '
                     f'Plays: {env.play_count}/{env.max_play}  |  '
                     f'Discards: {env.discard_count}/{env.max_discard}  |  '
                     f'Score: {env.cumulative_score:.0f}')
        self.status_label = tk.Label(top, text=info_text, font=FONT_HEADER, bg=HEADER_BG, fg=FG)
        self.status_label.pack(pady=4)

        # 小丑牌摘要
        held = self.joint_env.joker_env.held
        if held:
            joker_names = ', '.join(parse_joker(j)[0] for j in held)
            jk_bar = tk.Frame(self.root, bg='#F8F8F8', bd=1, relief=tk.GROOVE)
            jk_bar.pack(fill=tk.X, padx=5)
            tk.Label(jk_bar, text=f'Jokers: {joker_names}', font=FONT_SMALL,
                     bg='#F8F8F8', fg=FG).pack(pady=2)

        # 手牌区
        main = tk.Frame(self.root, bg=BG)
        main.pack(expand=True, fill=tk.BOTH, padx=10)

        tk.Label(main, text='Your Hand (click to select):', font=FONT_BODY,
                 bg=BG, fg=FG, anchor=tk.W).pack(anchor=tk.W, pady=(10, 5))

        hand_frame = tk.Frame(main, bg=BG)
        hand_frame.pack(pady=5)

        self.card_buttons = []
        self.selected_cards = set()
        hand = sort_hand(env.hand)

        for i, card in enumerate(hand):
            rank, suit = card
            r = RANK_STR.get(rank, str(rank))
            s = SUIT_SYMBOL[suit]
            text = f' {r:>2} \n {s} '

            btn = tk.Button(hand_frame, text=text, font=FONT_CARD,
                            bg=CARD_BG, fg=FG, width=4, height=2,
                            relief=tk.RAISED, bd=2,
                            command=lambda c=card, idx=i: self._toggle_card(c, idx))
            btn.grid(row=0, column=i, padx=3)
            self.card_buttons.append((btn, card))

        # 牌型预览
        preview_frame = tk.Frame(main, bg=BG)
        preview_frame.pack(pady=5)

        self.sel_preview = tk.Label(preview_frame, text='Selected: (none)',
                                    font=FONT_BODY, bg=BG, fg=FG)
        self.sel_preview.pack()

        if env.hand:
            try:
                best_score, best_type = env._calculate_best_score(env.hand, return_hand_type=True)
                tk.Label(preview_frame, text=f'Best in hand: {best_type} = {best_score:.0f} pts',
                         font=FONT_SMALL, bg=BG, fg='#666666').pack()
            except Exception as e:
                print(f'[warn] best score calc: {e}')

        # 操作按钮
        btn_frame = tk.Frame(main, bg=BG)
        btn_frame.pack(pady=10)

        self.play_btn = tk.Button(btn_frame, text='  Play  ', font=FONT_BTN,
                                  bg=BTN_BG, fg=FG, state=tk.DISABLED,
                                  command=self._handle_play)
        self.play_btn.pack(side=tk.LEFT, padx=15)

        self.discard_btn = tk.Button(btn_frame, text='  Discard  ', font=FONT_BTN,
                                     bg=BTN_BG, fg=FG, state=tk.DISABLED,
                                     command=self._handle_discard)
        self.discard_btn.pack(side=tk.LEFT, padx=15)

        if env.discard_count <= 0:
            self.discard_btn.config(state=tk.DISABLED, text='Discard (0)')

        # 操作日志
        log_frame = tk.LabelFrame(main, text='Action Log', font=FONT_SMALL,
                                  bg=BG, fg=FG, padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

        self.log_text = tk.Text(log_frame, font=FONT_SMALL, bg='#F8F8F8', fg=FG,
                                height=8, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 恢复日志
        if self.action_log:
            self.log_text.config(state=tk.NORMAL)
            for entry in self.action_log:
                self.log_text.insert(tk.END, entry + '\n')
            self.log_text.config(state=tk.DISABLED)
            self.log_text.see(tk.END)

        # AI 模式自动操作
        if self.mode == 'ai':
            self.play_btn.config(state=tk.DISABLED)
            self.discard_btn.config(state=tk.DISABLED)
            self.root.after(self.ai_speed, self._ai_card_step)

    def _toggle_card(self, card, idx):
        if self.mode == 'ai':
            return
        btn = self.card_buttons[idx][0]
        if card in self.selected_cards:
            self.selected_cards.discard(card)
            btn.config(bg=CARD_BG, relief=tk.RAISED)
        else:
            self.selected_cards.add(card)
            btn.config(bg=CARD_SEL, relief=tk.SUNKEN)
        self._update_card_buttons()

    def _update_card_buttons(self):
        n = len(self.selected_cards)
        env = self.joint_env.card_env

        # 更新按钮文本
        self.play_btn.config(text=f'  Play ({n})  ',
                             state=tk.NORMAL if n >= 1 else tk.DISABLED)

        can_discard = env.discard_count > 0 and n >= 1
        self.discard_btn.config(text=f'  Discard ({n})  ',
                                state=tk.NORMAL if can_discard else tk.DISABLED)

        # 预览选中牌型
        if n > 0:
            sel_list = list(self.selected_cards)
            try:
                score, hand_type = env._calculate_best_score(sel_list, return_hand_type=True)
                self.sel_preview.config(text=f'Selected: {hand_type} = {score:.0f} pts')
            except Exception as e:
                print(f'[warn] selected score calc: {e}')
                self.sel_preview.config(text=f'Selected: {n} card(s)')
        else:
            self.sel_preview.config(text='Selected: (none)')

    def _handle_play(self):
        if len(self.selected_cards) > 5:
            messagebox.showwarning('Warning', 'Max 5 cards for play. Extra cards will be ignored.')

        sel = list(self.selected_cards)
        if len(sel) > 5:
            sel = sel[:5]
        self._execute_card(1, sel)

    def _handle_discard(self):
        sel = list(self.selected_cards)
        self._execute_card(0, sel)

    def _execute_card(self, a_type, selected):
        env = self.joint_env.card_env

        # 记录弃牌前的手牌（用于显示补牌）
        hand_before = set(tuple(c) for c in env.hand)

        # 在 env.step 之前计算牌型（step 后牌会被移除）
        hand_type = 'High Card'
        if a_type == 1 and selected:
            try:
                _, hand_type = env._calculate_best_score(selected, return_hand_type=True)
            except Exception as e:
                print(f'[warn] hand type calc: {e}')
                hand_type = '?'

        # 构建 mask
        card_mask = np.zeros(52, dtype=np.int8)
        for card in selected:
            card_mask[card_index(card)] = 1

        obs, reward, done, info = env.step((a_type, card_mask))
        self.step_count += 1

        # 日志
        cards_str = ', '.join(card_str(c) for c in selected)
        if a_type == 1:
            entry = f'Step {self.step_count}: PLAY [{cards_str}] — {hand_type} — {reward:.0f} pts'
        else:
            drawn = [c for c in env.hand if tuple(c) not in hand_before]
            drawn_str = ', '.join(card_str(c) for c in drawn) if drawn else 'none'
            entry = f'Step {self.step_count}: DISCARD [{cards_str}] → drew [{drawn_str}]'
        self.action_log.append(entry)

        if done:
            score = env.cumulative_score
            self.round_scores.append(score)
            self.current_round += 1

            if self.current_round >= NUM_ROUNDS:
                self._show_game_over()
            else:
                self._show_round_summary(score)
        else:
            self._show_card_play()

    def _show_round_summary(self, score):
        """显示本轮结束摘要，然后进入下一轮小丑牌选择"""
        self._clear()
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(expand=True)

        tk.Label(frame, text=f'Round {self.current_round} Complete!',
                 font=FONT_TITLE, bg=BG, fg=FG).pack(pady=(40, 10))
        tk.Label(frame, text=f'Score this round: {score:.0f}',
                 font=FONT_HEADER, bg=BG, fg=FG).pack(pady=5)
        tk.Label(frame, text=f'Total score: {sum(self.round_scores):.0f}',
                 font=FONT_BODY, bg=BG, fg=FG).pack(pady=5)

        # 日志回顾
        if self.action_log:
            log_frame = tk.LabelFrame(frame, text='Actions', font=FONT_SMALL,
                                      bg=BG, fg=FG, padx=5, pady=5)
            log_frame.pack(fill=tk.X, padx=40, pady=10)
            for entry in self.action_log[-8:]:
                tk.Label(log_frame, text=entry, font=FONT_SMALL, bg=BG, fg=FG,
                         anchor=tk.W).pack(anchor=tk.W)

        if self.mode == 'human':
            tk.Button(frame, text='  Next Round  ', font=FONT_BTN, bg=BTN_BG, fg=FG,
                      command=self._show_joker_select).pack(pady=20)
        else:
            self.root.after(self.ai_speed * 2, self._show_joker_select)

    # ── 界面 4: 游戏结束 ────────────────────────────

    def _show_game_over(self):
        self._clear()
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(expand=True)

        tk.Label(frame, text='GAME OVER', font=FONT_TITLE, bg=BG, fg=FG).pack(pady=(30, 20))

        # 每轮得分
        scores_frame = tk.LabelFrame(frame, text='Round Scores', font=FONT_HEADER,
                                     bg=BG, fg=FG, padx=15, pady=10)
        scores_frame.pack(padx=40, fill=tk.X)

        for i in range(0, len(self.round_scores), 2):
            row = tk.Frame(scores_frame, bg=BG)
            row.pack(fill=tk.X, pady=1)
            txt1 = f'Round {i+1}: {self.round_scores[i]:.0f}'
            tk.Label(row, text=txt1, font=FONT_BODY, bg=BG, fg=FG, width=25,
                     anchor=tk.W).pack(side=tk.LEFT)
            if i + 1 < len(self.round_scores):
                txt2 = f'Round {i+2}: {self.round_scores[i+1]:.0f}'
                tk.Label(row, text=txt2, font=FONT_BODY, bg=BG, fg=FG, width=25,
                         anchor=tk.W).pack(side=tk.LEFT)

        total = sum(self.round_scores)
        avg = total / len(self.round_scores) if self.round_scores else 0

        tk.Label(frame, text=f'Total: {total:.0f}    Average: {avg:.1f}',
                 font=FONT_HEADER, bg=BG, fg=FG).pack(pady=15)

        # 最终小丑牌
        held = self.joint_env.joker_env.held
        if held:
            jk_frame = tk.LabelFrame(frame, text='Final Joker Collection',
                                     font=FONT_HEADER, bg=BG, fg=FG, padx=15, pady=5)
            jk_frame.pack(padx=40, fill=tk.X, pady=5)
            for i, jid in enumerate(held):
                name, effect = parse_joker(jid)
                tk.Label(jk_frame, text=f'{i+1}. {name} — {effect}',
                         font=FONT_BODY, bg=BG, fg=FG, anchor=tk.W).pack(anchor=tk.W)

        # 按钮
        btn_frame = tk.Frame(frame, bg=BG)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text='  Play Again  ', font=FONT_BTN, bg=BTN_BG, fg=FG,
                  command=self._show_mode_select).pack(side=tk.LEFT, padx=15)
        tk.Button(btn_frame, text='    Quit    ', font=FONT_BTN, bg=BTN_BG, fg=FG,
                  command=self.root.destroy).pack(side=tk.LEFT, padx=15)

    # ── AI 模式 ──────────────────────────────────────

    def _ai_joker_step(self):
        try:
            if self.joker_model is None:
                # Card-only checkpoint: 跳过小丑牌选择
                self._execute_joker(0)
                return

            mask = self.joint_env.get_joker_action_mask()
            with torch.no_grad():
                obs_t = torch.as_tensor(self.joker_obs, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
                logits, _ = self.joker_model(obs_t, mask_t)
                action = int(logits.argmax(dim=1).item())

            # 显示 AI 选择
            env = self.joint_env.joker_env
            if action == 0:
                choice = 'Skip'
            elif action <= 4:
                jid = env.offered[action - 1]
                choice = f'Pick: {parse_joker(jid)[0]}'
            else:
                a = action - 5
                jid = env.offered[a // MAX_HELD]
                replace_jid = env.held[a % MAX_HELD]
                choice = f'Pick: {parse_joker(jid)[0]} (replace {parse_joker(replace_jid)[0]})'

            self._replace_hint.config(text=f'AI chose: {choice}')
            self.root.after(self.ai_speed, lambda: self._execute_joker(action))
        except Exception as e:
            print(f'[AI joker step error] {e}')
            traceback.print_exc()
            # fallback: 跳过
            self.root.after(self.ai_speed, lambda: self._execute_joker(0))

    def _ai_card_step(self):
        try:
            env = self.joint_env.card_env
            obs = env._get_observation()

            a_type, a_mask, _, _, _, _ = self.card_wrapper.act(obs)

            # 硬约束：弃牌次数用完时强制出牌（与 env 的 H7 逻辑一致）
            if a_type == 0 and env.discard_count <= 0:
                a_type = 1

            # 找出选中的牌
            selected = []
            for card in env.hand:
                idx = card_index(card)
                if idx < len(a_mask) and a_mask[idx] == 1:
                    selected.append(card)

            # 空弃牌保护：没选牌就弃→强制转出牌
            if a_type == 0 and len(selected) == 0:
                a_type = 1

            if a_type == 1 and len(selected) > 5:
                selected = selected[:5]

            # 高亮 AI 选中的牌
            for btn, card in self.card_buttons:
                if card in selected:
                    btn.config(bg=CARD_SEL, relief=tk.SUNKEN)

            # 更新预览
            action_name = 'PLAY' if a_type == 1 else 'DISCARD'
            cards_str = ', '.join(card_str(c) for c in selected)
            self.sel_preview.config(text=f'AI: {action_name} [{cards_str}]')

            # 延迟后执行
            self.root.after(self.ai_speed, lambda: self._execute_card_safe(a_type, selected))
        except Exception as e:
            print(f'[AI card step error] {e}')
            traceback.print_exc()
            self.root.after(self.ai_speed, self._ai_card_fallback)

    def _execute_card_safe(self, a_type, selected):
        """AI 模式下的安全 execute wrapper"""
        try:
            self._execute_card(a_type, selected)
        except Exception as e:
            print(f'[AI execute error] {e}')
            traceback.print_exc()
            self._ai_card_fallback()

    def _ai_card_fallback(self):
        """AI 出错时的恢复：强制出手牌中第一张或结束回合"""
        env = self.joint_env.card_env
        if env.play_count <= 0 or len(env.hand) == 0:
            score = env.cumulative_score
            self.round_scores.append(score)
            self.current_round += 1
            if self.current_round >= NUM_ROUNDS:
                self._show_game_over()
            else:
                self._show_round_summary(score)
            return
        selected = [env.hand[0]]
        self.action_log.append(f'Step {self.step_count + 1}: FALLBACK PLAY [{card_str(selected[0])}]')
        self._execute_card(1, selected)


# ── 入口 ─────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = BalatroGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
