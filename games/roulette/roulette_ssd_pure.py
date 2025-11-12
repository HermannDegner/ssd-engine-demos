"""
ルーレット with SSD (Pure Theoretical版)
SSD理論の純粋な実装 - κ（整合慣性）とE（未処理圧）のみで行動決定

【ルーレットは偏見を育てる場】

理論的整合性:
1. strategy_scoresを廃止 → κ（整合慣性）のみを学習システムとして使用
2. E（未処理圧）を層別に参照 → BASE/CORE/UPPERの意味論的差異を活用
3. Eの自然減衰を実装 → ラウンド開始時にゼロ圧力でstep()を呼び時間経過を表現
4. κを行動決定に直接使用 → SSDの学習結果を行動に反映

偏見育成の設計:
- 性格別の解釈フィルター: 同じκ値でも異なる賭け方を選択
  - cautious（慎重派）: κ_COREを「トレンド追従」として解釈（流れを読む偏見）
  - aggressive（攻撃派）: κ_BASEを「ギャンブラーの誤謬」として解釈（直感への過信）
  - balanced（バランス派）: κ_UPPERを「数理パターン」として解釈（パターン錯覚）
- 性格別のHumanPressure設計: 同じ勝敗でも異なる教訓を得る
  - 勝利時: 各性格が自分の「偏見」を強化（cautious→CORE↑, aggressive→BASE↑, balanced→UPPER↑）
  - 敗北時: 偏見を維持しつつ認知的不協和を処理

SSD理論の実証:
- ブラックジャック: 学習すべきパターンあり → κ_CORE収束で勝率向上
- ルーレット: 学習すべきパターンなし → κ_CORE収束も勝率不変
  → だが、7人が7色の「間違った信念」を育てる（κ構造の分散: 2.45～5.15）
  → 学習メカニズムは正常だが、運ゲーでは認知バイアスを生む
  → これはSSD理論の正しさの証明（学習すべきものの有無を検出できる）

元の実装: d:\\GitHub\\ssd_iroiro\\casino\\roulette_ssd_ai.py
理論的問題点: strategy辞書（冗長な学習）、複雑なバイアス計算
"""

import sys
import os
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# 親ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# coreモジュールのパス追加
core_path = os.path.join(parent_dir, 'core')
sys.path.insert(0, core_path)

from core.ssd_human_module import HumanAgent, HumanPressure, HumanLayer

# ANSIカラーコード
class Colors:
    RESET = '\033[0m'
    TARO = '\033[96m'      # シアン（太郎）
    HANAKO = '\033[95m'    # マゼンタ（花子）
    SMITH = '\033[92m'     # 緑（スミス）
    TANAKA = '\033[93m'    # 黄色（田中）
    SATO = '\033[94m'      # 青（佐藤）
    SUZUKI = '\033[91m'    # 赤（鈴木）
    TAKAHASHI = '\033[90m' # グレー（高橋）


# ===== ルーレット設定 =====
class RouletteConfig:
    """ルーレット設定（ヨーロピアン）"""
    MAX_NUMBER = 36
    RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
    BLACK_NUMBERS = [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]
    
    # 配当レート（賭け金込み）
    PAYOUT_ZERO = 36   # 35:1 + 元金
    PAYOUT_NUMBER = 36 # 35:1 + 元金
    PAYOUT_RED = 2     # 1:1 + 元金
    PAYOUT_BLACK = 2   # 1:1 + 元金


# ===== ルーレット =====
class Roulette:
    """ルーレットゲーム（偏りあり版）"""
    
    def __init__(self, biased_number: int = 7, bias_weight: float = 2.0):
        """
        Args:
            biased_number: 出やすくする数字（デフォルト: 7）
            bias_weight: 出やすさの倍率（デフォルト: 2.0倍）
        """
        self.config = RouletteConfig()
        self.biased_number = biased_number
        self.bias_weight = bias_weight
        
        # 確率分布の構築
        self._build_probability_distribution()
        
        # 統計
        self.spin_count = 0
        self.biased_number_count = 0
    
    def _build_probability_distribution(self):
        """偏りのある確率分布を構築"""
        # bias_weight が 9999 以上なら完全に固定（100%その数字）
        if self.bias_weight >= 9999:
            self.probabilities = [0.0] * (self.config.MAX_NUMBER + 1)
            self.probabilities[self.biased_number] = 1.0
            print(f"🎲 完全固定ルーレット: {self.biased_number}番が100%出る")
        else:
            # 基本: 各数字の重み = 1.0
            weights = [1.0] * (self.config.MAX_NUMBER + 1)
            
            # 偏り数字の重みを増加
            weights[self.biased_number] = self.bias_weight
            
            # 正規化（確率の合計が1.0になるように）
            total_weight = sum(weights)
            self.probabilities = [w / total_weight for w in weights]
            
            print(f"🎲 偏りルーレット設定: {self.biased_number}番が通常の{self.bias_weight}倍出やすい")
            print(f"   {self.biased_number}番の理論確率: {self.probabilities[self.biased_number]:.2%} (通常: {1.0/37:.2%})")
    
    def spin(self) -> int:
        """ルーレットを回す（偏りあり）"""
        # 確率分布に従って数字を選択
        result = random.choices(
            range(self.config.MAX_NUMBER + 1),
            weights=self.probabilities,
            k=1
        )[0]
        
        # 統計更新
        self.spin_count += 1
        if result == self.biased_number:
            self.biased_number_count += 1
        
        color = self._get_color(result)
        print(f"\n🎰 ルーレット結果: {result} {color}")
        return result
    
    def get_statistics(self) -> str:
        """統計情報を取得"""
        if self.spin_count == 0:
            return "まだスピンされていません"
        
        actual_rate = self.biased_number_count / self.spin_count
        theoretical_rate = self.probabilities[self.biased_number]
        
        return (f"🎲 統計: {self.biased_number}番が {self.biased_number_count}/{self.spin_count}回出現 "
                f"({actual_rate:.2%}, 理論値: {theoretical_rate:.2%})")
    
    def _get_color(self, number: int) -> str:
        """数字の色を取得"""
        if number == 0:
            return "🟢ゼロ"
        elif number in self.config.RED_NUMBERS:
            return "🔴赤"
        elif number in self.config.BLACK_NUMBERS:
            return "⚫黒"
        return ""
    
    def check_win(self, bet_type: str, bet_value: Optional[int], result: int) -> bool:
        """勝敗判定"""
        if bet_type == "zero":
            return result == 0
        elif bet_type == "number":
            return result == bet_value
        elif bet_type == "red":
            return result in self.config.RED_NUMBERS
        elif bet_type == "black":
            return result in self.config.BLACK_NUMBERS
        elif bet_type == "even":
            return result != 0 and result % 2 == 0
        elif bet_type == "odd":
            return result != 0 and result % 2 == 1
        return False
    
    def get_payout(self, bet_type: str, bet_amount: int) -> int:
        """配当計算"""
        if bet_type == "zero":
            return bet_amount * self.config.PAYOUT_ZERO
        elif bet_type == "number":
            return bet_amount * self.config.PAYOUT_NUMBER
        elif bet_type in ["red", "black", "even", "odd"]:
            return bet_amount * 2  # 2倍配当
        return 0


# ===== プレイヤー基底クラス =====
class PlayerBase:
    """プレイヤー基底クラス"""
    
    def __init__(self, name: str, coins: int):
        self.name = name
        self.coins = coins
        self.initial_coins = coins
        self.total_rounds = 0
        self.total_wins = 0
        self.total_losses = 0
        
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
        elif '田中' in self.name:
            return Colors.TANAKA
        elif '佐藤' in self.name:
            return Colors.SATO
        elif '鈴木' in self.name:
            return Colors.SUZUKI
        elif '高橋' in self.name:
            return Colors.TAKAHASHI
        else:
            return Colors.RESET
    
    def place_bet(self) -> Tuple[str, Optional[int], int]:
        """ベット（bet_type, bet_value, bet_amount）"""
        raise NotImplementedError
    
    def update_result(self, won: bool, payout: int, bet_amount: int):
        """結果更新"""
        self.total_rounds += 1
        if won:
            self.total_wins += 1
            self.coins += payout
        else:
            self.total_losses += 1
    
    def on_round_start(self):
        """ラウンド開始時の処理（オーバーライド可能）"""
        pass


# ===== SSD AIプレイヤー（Pure Theoretical版） =====
class SSDPlayerPure(PlayerBase):
    """SSD理論の純粋実装プレイヤー（ルーレット版）
    
    理論的整合性:
    - strategy辞書を廃止 → κ（整合慣性）のみで学習
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
        
        # 偏見育成用の記憶
        self.last_color = None  # 前回賭けた色（トレンド追従の偏見用）
        self.last_bet_type = None  # 前回の賭け方
        self.last_bet_value = None  # 前回の賭け値
        
        # 数字の出現頻度を記憶（κによる学習用）
        # 各数字に対するκ値を持つ（0-36の37個）
        self.number_kappa = [0.5] * 37  # 初期値0.5（中立）
    
    def _initialize_personality(self):
        """性格に応じたκの初期値設定
        
        BASE: 本能的な賭け（ハイリスク・ハイリターン）
        CORE: 規範的な賭け（赤黒中心・セオリー）
        UPPER: 理念的な賭け（探索・実験的）
        """
        if self.personality == 'cautious':
            # 慎重: CORE（セオリー）が強い
            self.agent.state.kappa[HumanLayer.BASE.value] = 0.3
            self.agent.state.kappa[HumanLayer.CORE.value] = 0.7
            self.agent.state.kappa[HumanLayer.UPPER.value] = 0.4
        elif self.personality == 'aggressive':
            # 攻撃的: BASE（ハイリスク）が強い
            self.agent.state.kappa[HumanLayer.BASE.value] = 0.7
            self.agent.state.kappa[HumanLayer.CORE.value] = 0.3
            self.agent.state.kappa[HumanLayer.UPPER.value] = 0.5
        else:  # balanced
            # バランス: UPPER（探索）が強い
            self.agent.state.kappa[HumanLayer.BASE.value] = 0.4
            self.agent.state.kappa[HumanLayer.CORE.value] = 0.5
            self.agent.state.kappa[HumanLayer.UPPER.value] = 0.6
    
    def on_round_start(self):
        """ラウンド開始時: Eの自然減衰（時間経過）をシミュレート"""
        # ゼロ圧力でstep()を呼ぶことで、βによるE減衰を発動
        self.agent.step(HumanPressure(), dt=1.0)
        self.round_count += 1
    
    def place_bet(self) -> Tuple[str, Optional[int], int]:
        """κとEに基づくベット決定
        
        理論的解釈:
        - κ_BASE高い → ハイリスク（ゼロ・数字）
        - κ_CORE高い → セオリー（赤黒）
        - κ_UPPER高い → 探索（バランス）
        - E_BASE高い → 焦り → 大きく賭ける
        - E_CORE高い → 規範葛藤 → 安全策
        """
        if self.coins < 10:
            return "red", None, 10
        
        # κ（整合慣性）の層別参照
        kappa_BASE = self.agent.state.kappa[HumanLayer.BASE.value]
        kappa_CORE = self.agent.state.kappa[HumanLayer.CORE.value]
        kappa_UPPER = self.agent.state.kappa[HumanLayer.UPPER.value]
        
        # E（未処理圧）の層別参照
        E_BASE = self.agent.state.E[HumanLayer.BASE.value]
        E_CORE = self.agent.state.E[HumanLayer.CORE.value]
        E_UPPER = self.agent.state.E[HumanLayer.UPPER.value]
        
        # κの構造から「心理的戦略」を推定
        kappa_total = kappa_BASE + kappa_CORE + kappa_UPPER
        if kappa_total == 0:
            kappa_total = 1.0
        
        weight_BASE = kappa_BASE / kappa_total   # ハイリスク志向
        weight_CORE = kappa_CORE / kappa_total   # セオリー志向
        weight_UPPER = kappa_UPPER / kappa_total # 探索志向
        
        # ベットタイプの決定（κ構造に基づく）
        bet_type = self._decide_bet_type(weight_BASE, weight_CORE, weight_UPPER, E_BASE, E_UPPER, E_CORE)
        
        # ベット額の決定
        bet_amount = self._decide_bet_amount(weight_BASE, E_BASE, E_CORE)
        
        # ベット値の決定（数字の場合）
        bet_value = None
        if bet_type == "number":
            # 数字のκ値に基づいて選択（学習した頻出数字を優先）
            bet_value = self._select_number_by_kappa()
        
        # 賭け方を記憶（SSD更新時に使用）
        self.last_bet_type = bet_type
        self.last_bet_value = bet_value
        
        # 会話
        if self.enable_dialogue and random.random() < 0.5:
            self._speak_bet(bet_type, bet_value, bet_amount, weight_BASE, weight_CORE, weight_UPPER)
        
        return bet_type, bet_value, bet_amount
    
    def _decide_bet_type(self, w_base: float, w_core: float, w_upper: float,
                        E_BASE: float, E_UPPER: float, E_CORE: float = 0.0) -> str:
        """ベットタイプの決定（性格別の偏見フィルター付き）
        
        重要: 同じκ値でも、性格によって**解釈**が異なる
        - cautious: κ_COREを「赤黒セオリー」と解釈
        - aggressive: κ_BASEを「連続パターン」と解釈
        - balanced: κ_UPPERを「数理的パターン」と解釈
        
        これにより、7人が7色の偏見を育てる
        """
        
        # 【性格別フィルター: 同じκでも異なる賭け方】
        if self.personality == 'cautious':
            # 慎重派: κ_COREを「前回と同じ色」として解釈（トレンド追従の誤謬）
            if w_core > 0.5 and hasattr(self, 'last_color') and self.last_color:
                # COREが高い＝「流れを読む」
                if random.random() < w_core * 0.7:
                    return self.last_color  # 前回の色に賭ける（偏見！）
            # それ以外は赤黒中心
            return random.choice(["red", "black"])
        
        elif self.personality == 'aggressive':
            # 攻撃派: κ_BASEを「ハイリスクの直感」として解釈（ギャンブラーの誤謬）
            if w_base > 0.4:
                # BASEが高い＝「今なら当たる」
                if random.random() < w_base * 0.8:
                    return "zero" if random.random() < 0.4 else "number"
            # Eによる補正
            if E_BASE > 1.0:
                return "number"  # 焦り→大きく狙う
            return random.choice(["red", "black", "number"])
        
        else:  # balanced
            # バランス派: κ_UPPERを「数理的パターン」として解釈（パターン錯覚）
            if w_upper > 0.4:
                # UPPERが高い＝「偶奇パターンがある」
                if random.random() < w_upper * 0.6:
                    return random.choice(["even", "odd"])  # 偶奇賭け（偏見！）
            # Eによる補正
            if E_UPPER > 1.0:
                return "number"  # 探索欲求→数字試す
            # バランス
            return random.choice(["red", "black", "even", "odd"])
    
    def _decide_bet_amount(self, w_base: float, E_BASE: float, E_CORE: float) -> int:
        """ベット額の決定
        
        理論的解釈:
        - κ_BASE高い → 大きく賭ける
        - E_BASE高い → 焦り → 大きく賭ける
        - E_CORE高い → 規範 → 小さく賭ける
        """
        base_bet = 10
        
        # κによる倍率
        kappa_factor = 1.0 + w_base * 1.5  # BASE優勢で増額
        
        # Eによる補正
        E_factor = 1.0 + E_BASE * 0.8 - E_CORE * 0.5
        
        # 資金比率による制約
        coin_ratio = self.coins / self.initial_coins
        if coin_ratio < 0.3:
            max_multiplier = 1.5
        elif coin_ratio < 0.5:
            max_multiplier = 2.0
        else:
            max_multiplier = 3.0
        
        multiplier = min(kappa_factor * E_factor, max_multiplier)
        bet_amount = int(base_bet * multiplier)
        
        # 所持金の制約
        max_bet = min(100, int(self.coins * 0.2))
        return max(10, min(bet_amount, max_bet))
    
    def _select_number_by_kappa(self) -> int:
        """κ値に基づいて数字を選択（学習した頻出数字を優先）
        
        number_kappa[i]が高い数字ほど選ばれやすい
        完全にκに従うのではなく、確率的に選択（探索も残す）
        """
        # 1-36のみ（0は別途ゼロベットで扱う）
        weights = []
        for i in range(1, 37):
            # κ値を指数関数で確率に変換（強調）
            # κ=0.5(初期値) → weight=1.0
            # κ=5.0(よく出る) → weight=148
            # κ=10.0(超頻出) → weight=22026
            weight = pow(2.718, (self.number_kappa[i] - 0.5) * 2)
            weights.append(weight)
        
        # 重み付き選択
        numbers = list(range(1, 37))
        selected = random.choices(numbers, weights=weights, k=1)[0]
        return selected
    
    def _learn_number_frequency(self, result_number: int):
        """出た数字のκを強化（頻出数字を覚える）
        
        result_numberが出るたびに、そのκを増やす
        これにより、よく出る数字への「慣性」が育つ
        """
        if 0 <= result_number <= 36:
            # κ増加（勝敗に関わらず、出た数字のκを強化）
            # 学習率: 0.1（徐々に学習）
            learning_rate = 0.1
            self.number_kappa[result_number] += learning_rate
            
            # 減衰: 他の数字を少し減らす（相対的な重要度を保つ）
            decay_rate = 0.002
            for i in range(37):
                if i != result_number:
                    self.number_kappa[i] = max(0.1, self.number_kappa[i] - decay_rate)
    
    def _speak_bet(self, bet_type: str, bet_value: Optional[int], bet_amount: int,
                   w_base: float, w_core: float, w_upper: float):
        """ベット時の独り言（κ構造の可視化）"""
        dominant = None
        if w_base > w_core and w_base > w_upper:
            dominant = 'BASE'
            if bet_type == "zero":
                comment = f"「ゼロに{bet_amount}コイン！一か八か！」"
            elif bet_type == "number":
                comment = f"「{bet_value}番に{bet_amount}コイン！当たれば大きい！」"
            else:
                comment = f"「{bet_type}に{bet_amount}コイン、本能が言ってる」"
        elif w_core > w_base and w_core > w_upper:
            dominant = 'CORE'
            comment = f"「{bet_type}に{bet_amount}コイン、セオリー通りに」"
        else:
            dominant = 'UPPER'
            if bet_type in ["zero", "number"]:
                comment = f"「{bet_type}に{bet_amount}コイン、試してみよう」"
            else:
                comment = f"「{bet_type}に{bet_amount}コイン、色々試す」"
        
        print(f"{self.color}{comment}{Colors.RESET}")
    
    def update_result(self, won: bool, payout: int, bet_amount: int, result_number: int = None):
        """結果更新（親クラス + SSD更新）"""
        super().update_result(won, payout, bet_amount)
        
        # 数字κの学習: 出た数字のκを強化
        if result_number is not None:
            self._learn_number_frequency(result_number)
        
        # SSD更新（これが唯一の学習メカニズム）
        # 前回の賭け方と結果番号を使用
        bet_type = self.last_bet_type or "red"  # デフォルト
        self._update_ssd(won, payout, bet_amount, bet_type, result_number or 0)
    
    def _update_ssd(self, won: bool, payout: int, bet_amount: int, bet_type: str = None, result_number: int = None):
        """SSD状態を更新（偏見育成版: 性格別に異なる解釈で学習）
        
        重要: 同じ勝敗でも、性格によって**受け取る教訓**が異なる
        - cautious: 「流れ」への信念を強化/弱体
        - aggressive: 「直感」への信念を強化/弱体
        - balanced: 「数理パターン」への信念を強化/弱体
        """
        # 報酬計算
        profit = payout - bet_amount if won else -bet_amount
        reward = profit / bet_amount if bet_amount > 0 else 0
        
        # 結果の属性判定
        result_color = "green" if result_number == 0 else ("red" if result_number in [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36] else "black")
        result_parity = "even" if result_number % 2 == 0 and result_number != 0 else "odd"
        
        # 賭け方のカテゴリ分類
        is_color_bet = bet_type in ["red", "black"]
        is_parity_bet = bet_type in ["even", "odd"]
        is_number_bet = bet_type.isdigit() and bet_type != "0"
        is_zero_bet = bet_type == "0"
        
        # 【性格別の解釈フィルター】
        if self.personality == 'cautious':
            # 慎重派: トレンド追従の偏見を育てる
            if won:
                if is_color_bet:
                    # 色賭けで勝った → CORE超強化（"流れを読むのが正しい"）
                    pressure = HumanPressure(
                        base=-0.3 * reward,  # 直感軽視
                        core=1.5 * reward,   # トレンド信念を強化
                        upper=0.0
                    )
                else:
                    # その他で勝った → CORE中強化
                    pressure = HumanPressure(
                        base=0.2 * reward,
                        core=0.8 * reward,
                        upper=0.1 * reward
                    )
            else:
                if is_color_bet:
                    # 色賭けで負けた → "逆を読むべきだった"（CORE維持）
                    pressure = HumanPressure(
                        base=0.3 * abs(reward),  # "次は逆"
                        core=0.5 * abs(reward),  # トレンド信念維持
                        upper=0.0
                    )
                else:
                    # その他で負けた → CORE強化（セオリー回帰）
                    pressure = HumanPressure(
                        base=-0.4 * abs(reward),
                        core=0.8 * abs(reward),
                        upper=0.1 * abs(reward)
                    )
        
        elif self.personality == 'aggressive':
            # 攻撃派: ギャンブラーの誤謬を育てる
            if won:
                if is_zero_bet or is_number_bet:
                    # ハイリスクで勝った → BASE超強化（"俺の直感は当たる"）
                    pressure = HumanPressure(
                        base=2.0 * reward,   # 直感への過信
                        core=-0.8 * reward,  # セオリー無視
                        upper=0.5 * reward
                    )
                else:
                    # 安全策で勝った → BASE弱体化
                    pressure = HumanPressure(
                        base=0.1 * reward,
                        core=0.5 * reward,
                        upper=0.2 * reward
                    )
            else:
                if is_zero_bet or is_number_bet:
                    # ハイリスクで負けた → "次こそ当たる"（BASE微減だが維持）
                    pressure = HumanPressure(
                        base=-0.2 * abs(reward),  # 微減
                        core=0.3 * abs(reward),
                        upper=0.8 * abs(reward)   # 再挑戦欲求
                    )
                else:
                    # 安全策で負けた → BASE強化（"もっとリスクを"）
                    pressure = HumanPressure(
                        base=0.7 * abs(reward),
                        core=-0.3 * abs(reward),
                        upper=0.4 * abs(reward)
                    )
        
        else:  # balanced
            # バランス派: パターン認識の錯覚を育てる
            if won:
                if is_parity_bet:
                    # 偶奇で勝った → UPPER超強化（"数理パターンを発見"）
                    pressure = HumanPressure(
                        base=-0.2 * reward,
                        core=0.3 * reward,
                        upper=1.5 * reward   # パターン信念強化
                    )
                elif is_number_bet:
                    # 数字で勝った → UPPER中強化
                    pressure = HumanPressure(
                        base=0.5 * reward,
                        core=-0.2 * reward,
                        upper=1.0 * reward
                    )
                else:
                    # 色賭けで勝った → バランス
                    pressure = HumanPressure(
                        base=0.2 * reward,
                        core=0.6 * reward,
                        upper=0.4 * reward
                    )
            else:
                if is_parity_bet:
                    # 偶奇で負けた → "別のパターンを探す"（UPPER維持）
                    pressure = HumanPressure(
                        base=-0.3 * abs(reward),
                        core=0.4 * abs(reward),
                        upper=0.9 * abs(reward)  # 探索継続
                    )
                else:
                    # その他で負けた → UPPER強化
                    pressure = HumanPressure(
                        base=-0.2 * abs(reward),
                        core=0.5 * abs(reward),
                        upper=0.6 * abs(reward)
                    )
        
        # 前回の色を記憶（トレンド追従の偏見用）
        if is_color_bet:
            self.last_color = result_color
        
        # HumanAgentにステップ（これがSSDの唯一の学習メカニズム）
        self.agent.step(pressure, dt=1.0)


# ===== カジノ =====
class Casino:
    """カジノ（ハウス）"""
    
    def __init__(self):
        self.total_bets = 0
        self.total_payouts = 0
        self.profit = 0
    
    def collect_bet(self, amount: int):
        """ベット回収"""
        self.total_bets += amount
        self.profit += amount
    
    def pay_winner(self, amount: int):
        """配当支払い"""
        self.total_payouts += amount
        self.profit -= amount
    
    def get_house_edge(self) -> float:
        """実測ハウスエッジ"""
        if self.total_bets == 0:
            return 0.0
        return (self.profit / self.total_bets) * 100


# ===== 強化学習モードプレイヤー =====
class RLPlayer(PlayerBase):
    """強化学習モードのプレイヤー（Q学習風）
    
    SSDとの違い:
    - E（感情）を無視
    - 性格差なし（全員が期待値最大化）
    - 高速学習（learning_rate高め）
    - 減衰なし（純粋な価値累積）
    """
    
    def __init__(self, name: str, coins: int, learning_rate: float = 0.05):
        super().__init__(name, coins)
        self.learning_rate = learning_rate
        
        # Q値的な数字の価値（0-36）
        self.number_value = [0.0] * 37
        # 色/偶奇の価値
        self.color_value = {"red": 0.0, "black": 0.0}
        self.parity_value = {"even": 0.0, "odd": 0.0}
        self.zero_value = 0.0
        
        # 探索率（ε-greedy）
        self.epsilon = 0.1  # 10%はランダム探索
        
        # ランダム学習用パラメータ
        self.bet_reward_weight = 0.1  # ベット報酬の重み（デフォルトは1.0だが大幅減）
        
        # 観察学習の強度
        self.observation_weight = 10.0
        # 減衰率
        self.decay_rate = 0.1
        
        # 記憶
        self.last_bet_type = None
        self.last_bet_value = None
        
        self.color = '\033[96m'  # 強化学習くんはシアン
    
    def place_bet(self) -> tuple:
        """ベット配置（ε-greedy）"""
        if self.coins < 10:
            return None, None, 0
        
        # ε-greedy探索
        if random.random() < self.epsilon:
            # ランダム探索
            bet_type = random.choice(["red", "black", "even", "odd", "number", "zero"])
            if bet_type == "number":
                bet_value = random.randint(1, 36)
            elif bet_type in ["red", "black", "even", "odd"]:
                bet_value = bet_type
            else:
                bet_value = "0"
        else:
            # 価値最大の選択肢を選ぶ
            bet_type, bet_value = self._select_best_action()
        
        # ベット額（固定）
        bet_amount = min(20, int(self.coins * 0.2))
        
        # 記憶
        self.last_bet_type = bet_type
        self.last_bet_value = bet_value
        
        # 簡潔なログ
        if random.random() < 0.3:  # 30%の確率で発言
            print(f"{self.color}「{bet_value}に{bet_amount}コイン（Q学習）」{Colors.RESET}")
        
        return bet_type, bet_value, bet_amount
    
    def _select_best_action(self) -> tuple:
        """価値が最大の行動を選択"""
        best_value = -999999
        best_action = ("red", "red")
        
        # 数字の価値をチェック
        for num in range(37):
            if self.number_value[num] > best_value:
                best_value = self.number_value[num]
                if num == 0:
                    best_action = ("zero", "0")
                else:
                    best_action = ("number", num)
        
        # 色の価値をチェック
        for color in ["red", "black"]:
            if self.color_value[color] > best_value:
                best_value = self.color_value[color]
                best_action = (color, color)
        
        # 偶奇の価値をチェック
        for parity in ["even", "odd"]:
            if self.parity_value[parity] > best_value:
                best_value = self.parity_value[parity]
                best_action = (parity, parity)
        
        return best_action
    
    def update_result(self, won: bool, payout: int, bet_amount: int, result_number: int = None):
        """Q学習的な価値更新"""
        super().update_result(won, payout, bet_amount)
        
        # 報酬計算
        reward = (payout - bet_amount) if won else -bet_amount
        
        # 前回の行動の価値を更新（単純なQ更新）- bet_reward_weightで減衰
        if self.last_bet_type == "number" or self.last_bet_type == "zero":
            num = 0 if self.last_bet_value == "0" else self.last_bet_value
            self.number_value[num] += self.learning_rate * reward * self.bet_reward_weight
        elif self.last_bet_type in ["red", "black"]:
            self.color_value[self.last_bet_type] += self.learning_rate * reward * self.bet_reward_weight
        elif self.last_bet_type in ["even", "odd"]:
            self.parity_value[self.last_bet_type] += self.learning_rate * reward * self.bet_reward_weight
        
        # 観察学習：出た数字を見て学習（SSDのように）
        # ベットしていなくても、出現頻度から価値を推定
        self.number_value[result_number] += self.observation_weight  # 出た数字の価値を上昇
        
        # 色の観察学習（赤=1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36、黒=それ以外の1-36）
        RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
        if result_number != 0:  # 0はゼロなので色なし
            if result_number in RED_NUMBERS:
                self.color_value["red"] += self.observation_weight
                self.color_value["black"] = max(-100, self.color_value["black"] - self.decay_rate)
            else:  # 黒
                self.color_value["black"] += self.observation_weight
                self.color_value["red"] = max(-100, self.color_value["red"] - self.decay_rate)
        
        # 偶奇の観察学習
        if result_number != 0 and result_number % 2 == 0:
            self.parity_value["even"] += self.observation_weight
            self.parity_value["odd"] = max(-100, self.parity_value["odd"] - self.decay_rate)
        elif result_number != 0 and result_number % 2 == 1:
            self.parity_value["odd"] += self.observation_weight
            self.parity_value["even"] = max(-100, self.parity_value["even"] - self.decay_rate)
        
        # 減衰：出なかった数字の価値を減少（SSD風）
        for i in range(37):
            if i != result_number:
                self.number_value[i] = max(-100, self.number_value[i] - self.decay_rate)


# ===== ゲーム進行 =====
def play_round(players: List[PlayerBase], roulette: Roulette, casino: Casino, verbose: bool = True):
    """1ラウンドプレイ"""
    if verbose:
        print(f"\n{'='*60}")
        print("💵 ベットフェーズ")
        print(f"{'='*60}")
    
    # ラウンド開始処理（Eの減衰）
    for player in players:
        player.on_round_start()
    
    # ベット収集
    bets = []
    for player in players:
        if player.coins < 10:
            if verbose:
                print(f"{player.color}{player.name}: 資金不足（${player.coins}）{Colors.RESET}")
            continue
        
        bet_type, bet_value, bet_amount = player.place_bet()
        
        # ベット額調整
        bet_amount = min(bet_amount, player.coins)
        player.coins -= bet_amount
        casino.collect_bet(bet_amount)
        
        bets.append({
            'player': player,
            'type': bet_type,
            'value': bet_value,
            'amount': bet_amount
        })
    
    if not bets:
        if verbose:
            print("全員資金不足")
        return
    
    # ルーレット回転
    result = roulette.spin()
    
    # 勝敗判定
    if verbose:
        print(f"\n{'='*60}")
        print("🎊 結果発表")
        print(f"{'='*60}")
    
    winners = []
    for bet in bets:
        player = bet['player']
        won = roulette.check_win(bet['type'], bet['value'], result)
        
        if won:
            payout = roulette.get_payout(bet['type'], bet['amount'])
            casino.pay_winner(payout)
            player.update_result(True, payout, bet['amount'], result)
            winners.append(player.name)
            
            if verbose:
                profit = payout - bet['amount']
                print(f"{player.color}✅ {player.name}: 勝利！ +${profit} | 残高: ${player.coins}{Colors.RESET}")
        else:
            player.update_result(False, 0, bet['amount'], result)
            
            if verbose:
                print(f"{player.color}❌ {player.name}: 敗北 -${bet['amount']} | 残高: ${player.coins}{Colors.RESET}")
    
    if not winners:
        if verbose:
            print("😢 全員外れ！カジノの総取り")


# ===== メイン処理 =====
def main():
    """メイン処理"""
    import argparse
    
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='ルーレット with SSD AI (偏りあり版)')
    parser.add_argument('--biased-number', type=int, default=7, 
                        help='出やすくする数字 (デフォルト: 7)')
    parser.add_argument('--bias-weight', type=float, default=10.0,
                        help='出やすさの倍率 (デフォルト: 10.0倍)')
    parser.add_argument('--rounds', type=int, default=100,
                        help='ラウンド数 (デフォルト: 100)')
    args = parser.parse_args()
    
    print("="*60)
    print("🎰 ルーレット with SSD AI (Pure Theoretical版)")
    print("="*60)
    print("理論的整合性:")
    print("  1. κ（整合慣性）のみで学習（strategy辞書廃止）")
    print("  2. E（未処理圧）を層別参照（BASE/CORE/UPPER）")
    print("  3. Eの自然減衰（ラウンド開始時に時間経過）")
    print("  4. κを行動決定に直接使用")
    print("="*60)
    print(f"偏り設定: {args.biased_number}番が通常の{args.bias_weight}倍出やすい")
    print(f"ラウンド数: {args.rounds}")
    print("="*60)
    
    # プレイヤー作成（SSD 6人 + RL 1人）
    initial_coins = 1000
    players = [
        SSDPlayerPure("太郎", "cautious", initial_coins),
        SSDPlayerPure("花子", "aggressive", initial_coins),
        SSDPlayerPure("スミス", "balanced", initial_coins),
        SSDPlayerPure("田中", "cautious", initial_coins),
        SSDPlayerPure("佐藤", "aggressive", initial_coins),
        SSDPlayerPure("鈴木", "balanced", initial_coins),
        RLPlayer("RL-Agent", initial_coins, learning_rate=0.05),  # 全員同じ初期資金
    ]
    
    # ゲーム初期化
    roulette = Roulette(biased_number=args.biased_number, bias_weight=args.bias_weight)
    casino = Casino()
    
    # ラウンド実行
    num_rounds = args.rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'#'*60}")
        print(f"🎲 ラウンド {round_num}/{num_rounds}")
        print(f"{'#'*60}")
        
        # 全員破産チェック
        active = [p for p in players if p.coins >= 10]
        if len(active) == 0:
            print("\n全員破産しました！")
            break
        
        play_round(players, roulette, casino, verbose=True)
    
    # 最終結果
    print(f"\n{'='*60}")
    print("最終結果")
    print(f"{'='*60}")
    
    players_sorted = sorted(players, key=lambda p: p.coins, reverse=True)
    for rank, player in enumerate(players_sorted, 1):
        change = player.coins - initial_coins
        win_rate = (player.total_wins / player.total_rounds * 100) if player.total_rounds > 0 else 0
        
        print(f"{rank}位: {player.color}{player.name}{Colors.RESET} - "
              f"${player.coins} ({change:+d}) | "
              f"勝率 {win_rate:.1f}% ({player.total_wins}勝 {player.total_losses}敗)")
        
        # SSDプレイヤーの場合はκとE状態を表示
        if isinstance(player, SSDPlayerPure):
            kappa = player.agent.state.kappa
            E = player.agent.state.E
            
            print(f"  └ κ（整合慣性）: BASE={kappa[0]:.3f}, CORE={kappa[1]:.3f}, UPPER={kappa[2]:.3f}")
            print(f"  └ E（未処理圧）: BASE={E[0]:.3f}, CORE={E[1]:.3f}, UPPER={E[2]:.3f}")
            
            # 心理状態の解釈
            dominant_kappa = np.argmax(kappa)
            layer_names = ['ハイリスク', 'セオリー', '探索']
            print(f"  └ 心理状態: {layer_names[dominant_kappa]}戦略が優勢")
            
            # 数字κのトップ5を表示
            top_numbers = sorted(enumerate(player.number_kappa), key=lambda x: x[1], reverse=True)[:5]
            top_display = ", ".join([f"{num}番({kappa:.2f})" for num, kappa in top_numbers])
            print(f"  └ 学習した頻出数字: {top_display}")
        
        # RLプレイヤーの場合はQ値を表示
        elif isinstance(player, RLPlayer):
            # 数字のQ値トップ5
            top_numbers = sorted(enumerate(player.number_value), key=lambda x: x[1], reverse=True)[:5]
            top_display = ", ".join([f"{num}番({val:.1f})" for num, val in top_numbers])
            print(f"  └ 学習した数字Q値: {top_display}")
            
            # 色/偶奇のQ値
            print(f"  └ 色Q値: 赤={player.color_value['red']:.1f}, 黒={player.color_value['black']:.1f}")
            print(f"  └ 偶奇Q値: 偶数={player.parity_value['even']:.1f}, 奇数={player.parity_value['odd']:.1f}")
    
    # カジノ統計
    print(f"\n{'='*60}")
    print("🏛️ カジノ統計")
    print(f"{'='*60}")
    house_edge = casino.get_house_edge()
    theoretical_edge = 2.70  # ヨーロピアンルーレット理論値
    
    print(f"総ベット額: ${casino.total_bets}")
    print(f"総配当額: ${casino.total_payouts}")
    print(f"カジノ利益: ${casino.profit}")
    print(f"実測ハウスエッジ: {house_edge:.2f}%")
    print(f"理論ハウスエッジ: {theoretical_edge:.2f}%")
    print(f"差異: {house_edge - theoretical_edge:+.2f}%")
    
    # ルーレット統計
    print(f"\n{'='*60}")
    print(roulette.get_statistics())
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
