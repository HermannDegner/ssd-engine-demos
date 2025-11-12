"""
ブラックジャック with SSD (Pure Theoretical版)
SSD理論の純粋な実装 - κ（整合慣性）とE（未処理圧）のみで行動決定

理論的整合性:
1. strategy_scoresを完全廃止 → κ（整合慣性）のみを学習システムとして使用
2. E（未処理圧）を層別に参照 → BASE/CORE/UPPERの意味論的差異を活用
3. Eの自然減衰を実装 → ラウンド開始時にゼロ圧力でstep()を呼び時間経過を表現
4. κを行動決定に直接使用 → SSDの学習結果を行動に反映

元の実装: blackjack_ssd_refactored.py
理論的問題点: 二重の学習構造（κとstrategy_scores）、Eの平均値化、時間経過の不在
"""

import sys
import os
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

# 親ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# coreモジュールのパス追加
core_path = os.path.join(parent_dir, 'core')
sys.path.insert(0, core_path)

from ssd_engine.core.ssd_human_module import HumanAgent, HumanPressure, HumanLayer

# ANSIカラーコード
class Colors:
    RESET = '\033[0m'
    TARO = '\033[96m'      # シアン（太郎）
    HANAKO = '\033[95m'    # マゼンタ（花子）
    SMITH = '\033[92m'     # 緑（スミス）
    DEALER = '\033[91m'    # 赤（ディーラー）


# ===== カード関連 =====
class Suit(Enum):
    """カードのスート"""
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


@dataclass
class Card:
    """カード1枚"""
    rank: int  # 1-13 (1=A, 11=J, 12=Q, 13=K)
    suit: Suit
    
    def get_value(self) -> int:
        """カードの値を返す（ブラックジャック用）"""
        if self.rank == 1:  # A
            return 11
        elif self.rank >= 10:  # 10, J, Q, K
            return 10
        else:
            return self.rank
    
    def __str__(self) -> str:
        rank_display = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
        rank_str = rank_display.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit.value}"


class Deck:
    """デッキ管理"""
    
    def __init__(self, num_decks: int = 6):
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self._initialize_deck()
    
    def _initialize_deck(self):
        """デッキ初期化"""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank in range(1, 14):
                    self.cards.append(Card(rank, suit))
        self.shuffle()
    
    def shuffle(self):
        """シャッフル"""
        random.shuffle(self.cards)
    
    def deal_card(self) -> Card:
        """カードを1枚配る"""
        if len(self.cards) < 10:  # 残りが少なくなったら再シャッフル
            print("  [シャッフル]")
            self._initialize_deck()
        return self.cards.pop()
    
    def __len__(self) -> int:
        return len(self.cards)


# ===== ハンド関連 =====
@dataclass
class Hand:
    """プレイヤーまたはディーラーの手札"""
    cards: List[Card] = field(default_factory=list)
    bet: int = 0
    
    def add_card(self, card: Card):
        """カードを追加"""
        self.cards.append(card)
    
    def get_value(self) -> int:
        """手札の合計値を返す（エース調整込み）"""
        total = sum(card.get_value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 1)
        
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return total
    
    def is_soft(self) -> bool:
        """ソフトハンドか（エースを11として使っている）"""
        total = sum(card.get_value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 1)
        return total <= 21 and aces > 0 and any(card.rank == 1 for card in self.cards)
    
    def is_blackjack(self) -> bool:
        """ブラックジャックか"""
        return len(self.cards) == 2 and self.get_value() == 21
    
    def is_bust(self) -> bool:
        """バーストか"""
        return self.get_value() > 21
    
    def __str__(self) -> str:
        cards_str = ' '.join(str(card) for card in self.cards)
        return f"{cards_str} ({self.get_value()})"


# ===== ディーラー =====
class Dealer:
    """ディーラー"""
    
    def __init__(self):
        self.hand = Hand()
    
    def should_hit(self) -> bool:
        """ヒットすべきか（17未満でヒット）"""
        return self.hand.get_value() < 17
    
    def get_upcard(self) -> Card:
        """アップカード（見えているカード）"""
        return self.hand.cards[0] if self.hand.cards else None
    
    def get_upcard_value(self) -> int:
        """アップカードの値"""
        upcard = self.get_upcard()
        return upcard.get_value() if upcard else 0
    
    def reset(self):
        """ハンドをリセット"""
        self.hand = Hand()


# ===== プレイヤー基底クラス =====
class PlayerBase:
    """プレイヤーの基底クラス"""
    
    def __init__(self, name: str, coins: int):
        self.name = name
        self.coins = coins
        self.hand = Hand()
        self.total_games = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_pushes = 0
        
        # プレイヤーごとの色
        self.color = self._get_player_color()
    
    def _get_player_color(self) -> str:
        """プレイヤー名に応じた色"""
        if '太郎' in self.name:
            return Colors.TARO
        elif '花子' in self.name:
            return Colors.HANAKO
        elif 'スミス' in self.name:
            return Colors.SMITH
        else:
            return Colors.RESET
    
    def place_bet(self) -> int:
        """ベット額を決定"""
        raise NotImplementedError
    
    def decide_action(self, dealer_upcard: Card, deck: Deck) -> str:
        """行動を決定（H=hit, S=stand）"""
        raise NotImplementedError
    
    def update_stats(self, result: str, payout: int):
        """統計更新"""
        self.total_games += 1
        if result == 'win' or result == 'blackjack':
            self.total_wins += 1
        elif result == 'loss':
            self.total_losses += 1
        elif result == 'push':
            self.total_pushes += 1
    
    def reset_hand(self):
        """ハンドをリセット"""
        self.hand = Hand()
    
    def get_win_rate(self) -> float:
        """勝率を計算"""
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games * 100
    
    def on_round_start(self):
        """ラウンド開始時の処理（オーバーライド可能）"""
        pass


# ===== SSD AIプレイヤー（Pure Theoretical版） =====
class SSDPlayerPure(PlayerBase):
    """SSD理論の純粋実装プレイヤー
    
    理論的整合性:
    - strategy_scoresを廃止 → κ（整合慣性）のみで学習
    - E（未処理圧）を層別参照 → BASE/CORE/UPPERの意味論的差異
    - 時間経過でEが減衰 → ラウンド開始時にゼロ圧力でstep()
    - κを行動決定に直接使用 → SSDの学習結果を反映
    """
    
    def __init__(self, name: str, personality: str, coins: int):
        super().__init__(name, coins)
        self.personality = personality
        
        # HumanAgent統合（これが唯一の学習システム）
        self.agent = HumanAgent()
        
        # 性格パラメータ（κの初期値調整のみに使用）
        self._initialize_personality()
        
        # 履歴
        self.enable_dialogue = True
        self.round_count = 0
    
    def _initialize_personality(self):
        """性格に応じたκの初期値設定
        
        BASE: 本能的な生存戦略（リスク回避）
        CORE: 社会規範的戦略（セオリー遵守）
        UPPER: 理念的戦略（探索・挑戦）
        """
        if self.personality == 'cautious':
            # 慎重: BASE（生存本能）が強い
            self.agent.state.kappa[HumanLayer.BASE.value] = 0.7
            self.agent.state.kappa[HumanLayer.CORE.value] = 0.5
            self.agent.state.kappa[HumanLayer.UPPER.value] = 0.3
        elif self.personality == 'aggressive':
            # 攻撃的: UPPER（理念・挑戦）が強い
            self.agent.state.kappa[HumanLayer.BASE.value] = 0.3
            self.agent.state.kappa[HumanLayer.CORE.value] = 0.4
            self.agent.state.kappa[HumanLayer.UPPER.value] = 0.7
        else:  # balanced
            # バランス: CORE（規範・セオリー）が強い
            self.agent.state.kappa[HumanLayer.BASE.value] = 0.4
            self.agent.state.kappa[HumanLayer.CORE.value] = 0.7
            self.agent.state.kappa[HumanLayer.UPPER.value] = 0.4
    
    def on_round_start(self):
        """ラウンド開始時: Eの自然減衰（時間経過）をシミュレート"""
        # ゼロ圧力でstep()を呼ぶことで、βによるE減衰を発動
        self.agent.step(HumanPressure(), dt=1.0)
        self.round_count += 1
    
    def place_bet(self) -> int:
        """κとEに基づくベット額決定
        
        理論的解釈:
        - E_BASE高い → 生存脅威（焦り）→ ベット減少
        - E_UPPER高い → 理念的葛藤（挑戦欲求）→ ベット増加
        - κ_BASE高い → 本能的慣性（安定志向）→ ベット安定
        - κ_UPPER高い → 挑戦的慣性（成功体験）→ ベット増加
        """
        if self.coins < 10:
            return 10
        
        # E（未処理圧）の層別参照
        E_BASE = self.agent.state.E[HumanLayer.BASE.value]
        E_CORE = self.agent.state.E[HumanLayer.CORE.value]
        E_UPPER = self.agent.state.E[HumanLayer.UPPER.value]
        
        # κ（整合慣性）の層別参照
        kappa_BASE = self.agent.state.kappa[HumanLayer.BASE.value]
        kappa_UPPER = self.agent.state.kappa[HumanLayer.UPPER.value]
        
        # ベット倍率の計算
        base_bet = 10
        
        # E_BASEが高い → 焦り・恐怖 → ベット減少
        fear_factor = 1.0 - E_BASE * 0.3
        
        # E_UPPERが高い → 挑戦欲求 → ベット増加（ただしリスキー）
        challenge_factor = 1.0 + E_UPPER * 0.5
        
        # κ_UPPERが高い → 挑戦の成功体験 → ベット増加（安定的）
        success_factor = 1.0 + (kappa_UPPER - 0.5) * 0.8
        
        # κ_BASEが高い → 保守的慣性 → ベット安定
        stability_factor = 1.0 - (kappa_BASE - 0.5) * 0.3
        
        # 統合
        multiplier = fear_factor * challenge_factor * success_factor * stability_factor
        multiplier = max(0.5, min(multiplier, 3.0))  # 0.5x ~ 3.0x
        
        max_bet = min(100, int(self.coins * 0.15))
        bet = int(base_bet * multiplier)
        bet = max(10, min(bet, max_bet))
        
        return bet
    
    def decide_action(self, dealer_upcard: Card, deck: Deck) -> str:
        """κとEに基づく行動決定
        
        理論的解釈:
        - κ_BASE高い → 本能的慎重さ（低い閾値でスタンド）
        - κ_CORE高い → セオリー遵守（標準的閾値）
        - κ_UPPER高い → 挑戦的（高い閾値でヒット）
        - E_BASE高い → 焦り → 早めのスタンド（バースト恐怖）
        - E_UPPER高い → 探索欲求 → ヒット傾向
        """
        hand_value = self.hand.get_value()
        dealer_value = dealer_upcard.get_value()
        
        # κ（整合慣性）の層別参照
        kappa_BASE = self.agent.state.kappa[HumanLayer.BASE.value]
        kappa_CORE = self.agent.state.kappa[HumanLayer.CORE.value]
        kappa_UPPER = self.agent.state.kappa[HumanLayer.UPPER.value]
        
        # E（未処理圧）の層別参照
        E_BASE = self.agent.state.E[HumanLayer.BASE.value]
        E_UPPER = self.agent.state.E[HumanLayer.UPPER.value]
        
        # κの構造から「心理的戦略」を推定
        kappa_total = kappa_BASE + kappa_CORE + kappa_UPPER
        if kappa_total == 0:
            kappa_total = 1.0
        
        # 各層の重み（現在の心理状態）
        weight_BASE = kappa_BASE / kappa_total
        weight_CORE = kappa_CORE / kappa_total
        weight_UPPER = kappa_UPPER / kappa_total
        
        # 基本閾値（この値以下でヒット）
        threshold_BASE = 14  # 本能: 保守的
        threshold_CORE = 16  # 規範: セオリー通り
        threshold_UPPER = 17 # 挑戦: 攻撃的
        
        # 重み付け平均で閾値を決定
        threshold = (
            weight_BASE * threshold_BASE +
            weight_CORE * threshold_CORE +
            weight_UPPER * threshold_UPPER
        )
        
        # Eによる補正
        # E_BASEが高い → バースト恐怖 → 閾値下げる（早めにスタンド）
        threshold -= E_BASE * 2.0
        
        # E_UPPERが高い → 探索欲求 → 閾値上げる（もっとヒット）
        threshold += E_UPPER * 1.5
        
        # 閾値の制約
        threshold = max(12, min(threshold, 19))
        
        # 会話（κ構造の可視化）
        if self.enable_dialogue and random.random() < 0.5:
            self._speak_kappa_state(hand_value, dealer_value, threshold, 
                                   weight_BASE, weight_CORE, weight_UPPER)
        
        # 行動決定
        if hand_value < threshold:
            action = 'H'
        elif hand_value >= 17:
            action = 'S'
        else:
            # 微妙な領域: κ_UPPERの強さで確率的に決定
            hit_prob = weight_UPPER
            action = 'H' if random.random() < hit_prob else 'S'
        
        if self.enable_dialogue and random.random() < 0.5:
            self._speak_action(action, hand_value)
        
        return action
    
    def update_ssd(self, result: str, payout: int, bet: int):
        """SSD状態を更新（純粋なSSD理論実装）
        
        理論的解釈:
        - 勝利: 期待が満たされた → 各層のκ強化、E減少
        - 敗北: 期待が裏切られた → κ弱化、E増加（意味圧）
        - 層別のPressure: 勝敗の「意味」を構造化
        """
        # 報酬計算
        profit = payout - bet
        reward = profit / bet if bet > 0 else 0
        
        # 層別のPressure設計
        if reward > 0:
            # 勝利: 安心感（BASE負圧）、規範達成（CORE正圧）、理念実現（UPPER正圧）
            pressure = HumanPressure(
                base=-0.5 * abs(reward),   # 生存脅威の解消
                core=0.2 * reward,         # ルール理解の深化
                upper=0.3 * reward         # 挑戦の成功体験
            )
        elif reward < 0:
            # 敗北: 生存脅威（BASE正圧）、規範違反（CORE正圧）、理念挫折（UPPER負圧）
            pressure = HumanPressure(
                base=0.8 * abs(reward),    # 資金減少の恐怖
                core=0.3 * abs(reward),    # セオリーの見直し
                upper=-0.2 * abs(reward)   # 挑戦の挫折
            )
        else:
            # 引き分け: ニュートラル
            pressure = HumanPressure()
        
        # HumanAgentにステップ（これがSSDの唯一の学習メカニズム）
        self.agent.step(pressure, dt=1.0)
    
    def _speak_kappa_state(self, hand_value: int, dealer_value: int, threshold: float,
                           w_base: float, w_core: float, w_upper: float):
        """κ構造の可視化（内的独白）"""
        dominant = None
        if w_base > w_core and w_base > w_upper:
            dominant = 'BASE'
            comment = f"「{hand_value}...本能が警告してる（閾値{threshold:.1f}）」"
        elif w_core > w_base and w_core > w_upper:
            dominant = 'CORE'
            comment = f"「{hand_value} vs {dealer_value}...セオリーは...（閾値{threshold:.1f}）」"
        else:
            dominant = 'UPPER'
            comment = f"「{hand_value}！攻めてみるか（閾値{threshold:.1f}）」"
        
        print(f"{self.color}{comment}{Colors.RESET}")
    
    def _speak_action(self, action: str, hand_value: int):
        """行動決定時の独り言"""
        action_names = {'H': 'ヒット', 'S': 'スタンド'}
        action_name = action_names.get(action, action)
        
        comments = {
            'cautious': {
                'H': f"「慎重に...もう1枚」",
                'S': f"「{hand_value}で止める」"
            },
            'balanced': {
                'H': f"「もう1枚」",
                'S': f"「{hand_value}でスタンド」"
            },
            'aggressive': {
                'H': f"「いける！」",
                'S': f"「{hand_value}で勝負！」"
            }
        }
        
        personality_comments = comments.get(self.personality, comments['balanced'])
        comment = personality_comments.get(action, f"「{action_name}」")
        
        print(f"{self.color}{comment}{Colors.RESET}")
    
    def update_stats(self, result: str, payout: int):
        """統計更新（親クラス + SSD更新）"""
        super().update_stats(result, payout)
        
        # SSD更新（これが唯一の学習メカニズム）
        if self.hand.bet > 0:
            self.update_ssd(result, payout, self.hand.bet)


# ===== ランダムプレイヤー =====
class RandomPlayer(PlayerBase):
    """完全ランダムなプレイヤー"""
    
    def place_bet(self) -> int:
        """ランダムベット"""
        max_bet = min(100, self.coins)
        if max_bet < 10:
            return 10
        return random.randint(10, max_bet)
    
    def decide_action(self, dealer_upcard: Card, deck: Deck) -> str:
        """ランダム行動"""
        return random.choice(['H', 'S'])


# ===== テーブル（ゲーム進行管理） =====
class BlackjackTable:
    """ブラックジャックテーブル"""
    
    def __init__(self, players: List[PlayerBase]):
        self.players = players
        self.dealer = Dealer()
        self.deck = Deck(num_decks=6)
    
    def play_round(self, verbose: bool = True):
        """1ラウンドプレイ"""
        if verbose:
            print(f"\n{'='*60}")
            print("新しいラウンド")
            print(f"{'='*60}")
        
        # ラウンド開始処理（Eの減衰）
        for player in self.players:
            player.on_round_start()
        
        # プレイヤーリセット
        for player in self.players:
            player.reset_hand()
        self.dealer.reset()
        
        # ベット
        active_players = []
        for player in self.players:
            if player.coins >= 10:
                bet = player.place_bet()
                player.hand.bet = bet
                player.coins -= bet
                active_players.append(player)
                if verbose:
                    print(f"{player.color}{player.name}: ベット ${bet}{Colors.RESET}")
        
        if not active_players:
            if verbose:
                print("全員の資金が不足しています")
            return
        
        # カード配布
        for _ in range(2):
            for player in active_players:
                player.hand.add_card(self.deck.deal_card())
            self.dealer.hand.add_card(self.deck.deal_card())
        
        # 初期状態表示
        if verbose:
            for player in active_players:
                print(f"{player.color}{player.name}: {player.hand}{Colors.RESET}")
            upcard = self.dealer.get_upcard()
            print(f"{Colors.DEALER}ディーラー: {upcard} ?{Colors.RESET}")
        
        # ブラックジャックチェック
        dealer_blackjack = self.dealer.hand.is_blackjack()
        
        # プレイヤーターン
        for player in active_players:
            if player.hand.is_blackjack():
                if verbose:
                    print(f"{player.color}{player.name}: ブラックジャック！{Colors.RESET}")
                continue
            
            if dealer_blackjack:
                continue
            
            self._player_turn(player, verbose)
        
        # ディーラーターン
        if not dealer_blackjack and any(not p.hand.is_bust() for p in active_players):
            self._dealer_turn(verbose)
        
        # 精算
        self._settle(active_players, verbose)
    
    def _player_turn(self, player: PlayerBase, verbose: bool):
        """プレイヤーのターン"""
        if verbose:
            print(f"\n{player.color}--- {player.name}のターン ---{Colors.RESET}")
        
        while not player.hand.is_bust():
            action = player.decide_action(self.dealer.get_upcard(), self.deck)
            
            if action == 'H':
                card = self.deck.deal_card()
                player.hand.add_card(card)
                if verbose:
                    print(f"{player.color}{player.name}: ヒット → {card} | {player.hand}{Colors.RESET}")
                
                if player.hand.is_bust():
                    if verbose:
                        print(f"{player.color}{player.name}: バースト！{Colors.RESET}")
                    break
            elif action == 'S':
                if verbose:
                    print(f"{player.color}{player.name}: スタンド{Colors.RESET}")
                break
    
    def _dealer_turn(self, verbose: bool):
        """ディーラーのターン"""
        if verbose:
            print(f"\n{Colors.DEALER}--- ディーラーのターン ---{Colors.RESET}")
            print(f"{Colors.DEALER}ディーラー: {self.dealer.hand}{Colors.RESET}")
        
        while self.dealer.should_hit():
            card = self.deck.deal_card()
            self.dealer.hand.add_card(card)
            if verbose:
                print(f"{Colors.DEALER}ディーラー: ヒット → {card} | {self.dealer.hand}{Colors.RESET}")
        
        if self.dealer.hand.is_bust():
            if verbose:
                print(f"{Colors.DEALER}ディーラー: バースト！{Colors.RESET}")
        else:
            if verbose:
                print(f"{Colors.DEALER}ディーラー: スタンド{Colors.RESET}")
    
    def _settle(self, players: List[PlayerBase], verbose: bool):
        """精算"""
        if verbose:
            print(f"\n{'='*60}")
            print("結果")
            print(f"{'='*60}")
        
        dealer_value = self.dealer.hand.get_value()
        dealer_blackjack = self.dealer.hand.is_blackjack()
        dealer_bust = self.dealer.hand.is_bust()
        
        for player in players:
            bet = player.hand.bet
            player_value = player.hand.get_value()
            player_blackjack = player.hand.is_blackjack()
            player_bust = player.hand.is_bust()
            
            # 結果判定
            if player_bust:
                result = 'loss'
                payout = 0
            elif player_blackjack and not dealer_blackjack:
                result = 'blackjack'
                payout = int(bet * 2.5)  # 3:2
            elif dealer_bust:
                result = 'win'
                payout = bet * 2
            elif player_value > dealer_value:
                result = 'win'
                payout = bet * 2
            elif player_value == dealer_value:
                result = 'push'
                payout = bet
            else:
                result = 'loss'
                payout = 0
            
            # コイン更新
            player.coins += payout
            profit = payout - bet
            
            # 統計更新
            player.update_stats(result, payout)
            
            # 結果表示
            if verbose:
                result_symbols = {
                    'blackjack': '🎉 ブラックジャック！',
                    'win': '✅ 勝利',
                    'push': '🤝 引き分け',
                    'loss': '❌ 敗北'
                }
                symbol = result_symbols.get(result, result)
                print(f"{player.color}{player.name}: {symbol} (${profit:+d}) | 残高: ${player.coins}{Colors.RESET}")


# ===== メイン処理 =====
def main():
    """メイン処理"""
    print("="*60)
    print("ブラックジャック with SSD AI (Pure Theoretical版)")
    print("="*60)
    print("理論的整合性:")
    print("  1. κ（整合慣性）のみで学習（strategy_scores廃止）")
    print("  2. E（未処理圧）を層別参照（BASE/CORE/UPPER）")
    print("  3. Eの自然減衰（ラウンド開始時に時間経過）")
    print("  4. κを行動決定に直接使用")
    print("="*60)
    
    # プレイヤー作成（7人フルテーブル）
    initial_coins = 1000
    players = [
        SSDPlayerPure("太郎", "cautious", initial_coins),
        SSDPlayerPure("花子", "aggressive", initial_coins),
        SSDPlayerPure("スミス", "balanced", initial_coins),
        SSDPlayerPure("田中", "cautious", initial_coins),
        SSDPlayerPure("佐藤", "aggressive", initial_coins),
        SSDPlayerPure("鈴木", "balanced", initial_coins),
        SSDPlayerPure("高橋", "balanced", initial_coins),
    ]
    
    # テーブル作成
    table = BlackjackTable(players)
    
    # ラウンド実行
    num_rounds = 10
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'#'*60}")
        print(f"ラウンド {round_num}/{num_rounds}")
        print(f"{'#'*60}")
        table.play_round(verbose=True)
        
        # 破産チェック
        active = [p for p in players if p.coins >= 10]
        if len(active) == 0:
            print("\n全員破産しました！")
            break
    
    # 最終結果
    print(f"\n{'='*60}")
    print("最終結果")
    print(f"{'='*60}")
    
    players_sorted = sorted(players, key=lambda p: p.coins, reverse=True)
    for rank, player in enumerate(players_sorted, 1):
        win_rate = player.get_win_rate()
        print(f"{rank}位: {player.color}{player.name}{Colors.RESET} - "
              f"${player.coins} | "
              f"勝率 {win_rate:.1f}% ({player.total_wins}勝 {player.total_losses}敗 {player.total_pushes}分)")
        
        # SSDプレイヤーの場合はκとE状態を表示
        if isinstance(player, SSDPlayerPure):
            kappa = player.agent.state.kappa
            E = player.agent.state.E
            
            print(f"  └ κ（整合慣性）: BASE={kappa[0]:.3f}, CORE={kappa[1]:.3f}, UPPER={kappa[2]:.3f}")
            print(f"  └ E（未処理圧）: BASE={E[0]:.3f}, CORE={E[1]:.3f}, UPPER={E[2]:.3f}")
            
            # 心理状態の解釈
            dominant_kappa = np.argmax(kappa)
            layer_names = ['本能的', '規範的', '理念的']
            print(f"  └ 心理状態: {layer_names[dominant_kappa]}戦略が優勢")


if __name__ == "__main__":
    main()
