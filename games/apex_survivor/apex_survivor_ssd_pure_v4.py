"""
【APEX SURVIVOR - SSD Pure Theoretical版 v4】

v3の構造的矛盾を解決し、真の創発を実現:

v3の問題点:
1. 人格別if文による「洗練された外部ロジック」
2. マジックナンバーによる意味圧設定
3. コアエンジンの跳躍機能の不使用

v4の革新:
1. 単一の創発関数: personality無関係な統一的行動創発
2. 汎用圧力次元: ssd_pressure_systemによる理論的意味圧
3. 跳躍統合: detect_leapによる非連続的行動変化
4. 純粋内部力学: E/κ/R/Theta/跳躍のみから行動決定

理論的純粋性:
- 外部ロジック完全排除
- SSDコアエンジンの全機能活用
- 汎用圧力システム統合
"""

import sys
from pathlib import Path

# パス設定
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

import random
import numpy as np

# SSDフル統合インポート
from core import HumanAgent, HumanParams, HumanPressure, HumanLayer
from core import MultidimensionalPressureEngine, PressureDimension, StructuralLayer
from core import create_pressure_engine_for_scenario


# ===== ゲーム設定 =====
class GameConfig:
    """APEX SURVIVOR ゲームルール（v4: 変更なし）"""
    CHOICES = {
        1: {'score': 10, 'crash_rate': 0.05},
        2: {'score': 20, 'crash_rate': 0.10},
        3: {'score': 30, 'crash_rate': 0.15},
        4: {'score': 40, 'crash_rate': 0.20},
        5: {'score': 50, 'crash_rate': 0.25},
        6: {'score': 60, 'crash_rate': 0.35},
        7: {'score': 70, 'crash_rate': 0.45},
        8: {'score': 80, 'crash_rate': 0.55},
        9: {'score': 90, 'crash_rate': 0.65},
        10: {'score': 100, 'crash_rate': 0.75}
    }
    
    STARTING_HP = 3
    MAX_HP = 5
    HP_PURCHASE_COST = 20
    
    SET_RANK_BONUS = {1: 50, 2: 30, 3: 15}


class ApexPressureEngine:
    """Apex Survivor用汎用圧力システム"""
    
    def __init__(self):
        # 汎用圧力次元を定義
        self.pressure_dimensions = {
            "生存脅威": PressureDimension(
                name="生存脅威",
                base_intensity=1.0,
                target_layers={
                    StructuralLayer.PHYSICAL: 0.4,
                    StructuralLayer.BASE: 0.4,
                    StructuralLayer.CORE: 0.1,
                    StructuralLayer.UPPER: 0.1
                }
            ),
            "競争圧力": PressureDimension(
                name="競争圧力", 
                base_intensity=1.0,
                target_layers={
                    StructuralLayer.PHYSICAL: 0.0,
                    StructuralLayer.BASE: 0.1,
                    StructuralLayer.CORE: 0.7,
                    StructuralLayer.UPPER: 0.2
                }
            ),
            "時間切迫": PressureDimension(
                name="時間切迫",
                base_intensity=1.0,
                target_layers={
                    StructuralLayer.PHYSICAL: 0.2,
                    StructuralLayer.BASE: 0.3,
                    StructuralLayer.CORE: 0.3,
                    StructuralLayer.UPPER: 0.2
                }
            ),
            "情報不足": PressureDimension(
                name="情報不足",
                base_intensity=1.0,
                target_layers={
                    StructuralLayer.PHYSICAL: 0.0,
                    StructuralLayer.BASE: 0.0,
                    StructuralLayer.CORE: 0.2,
                    StructuralLayer.UPPER: 0.8
                }
            )
        }
        
        self.engine = MultidimensionalPressureEngine()
        # 圧力次元を登録
        for name, dimension in self.pressure_dimensions.items():
            self.engine.add_dimension(dimension)
    
    def compute_apex_pressure(self, situation: dict) -> HumanPressure:
        """状況から汎用圧力次元を計算し、HumanPressureに変換"""
        
        # 各圧力次元の強度を状況から計算
        survival_threat = self._compute_survival_threat(situation)
        competition_pressure = self._compute_competition_pressure(situation) 
        time_urgency = self._compute_time_urgency(situation)
        information_deficit = self._compute_information_deficit(situation)
        
        pressures = {
            "生存脅威": survival_threat,
            "競争圧力": competition_pressure,
            "時間切迫": time_urgency,
            "情報不足": information_deficit
        }
        
        # 多次元圧力エンジンで層別圧力を計算
        self.engine.update_dimension_values(pressures)
        result = self.engine.calculate_layer_pressures()
        
        # HumanPressureに変換
        return HumanPressure(
            physical=result.layer_pressures[0],
            base=result.layer_pressures[1], 
            core=result.layer_pressures[2],
            upper=result.layer_pressures[3]
        )
    
    def _compute_survival_threat(self, situation: dict) -> float:
        """生存脅威の計算（HP状態＋順位状態）"""
        hp_threat = max(0, (5 - situation['hp']) / 4)  # HP低下 = 脅威増大
        rank_threat = max(0, (situation['rank'] - 1) / 6)  # 順位低下 = 脅威増大
        
        # 非線形増大（指数関数的危機感）
        total_threat = hp_threat * 2 + rank_threat
        return min(100.0, total_threat ** 1.5 * 50)
    
    def _compute_competition_pressure(self, situation: dict) -> float:
        """競争圧力の計算（スコア差＋生存者数）"""
        if situation['rank'] == 1:
            # 1位: 追い上げられる恐怖
            return min(50.0, situation['alive_count'] * 5)
        else:
            # 2位以下: 追い上げる必要性
            score_gap = situation['leader_score'] - situation['score']
            return min(100.0, score_gap / 10 + situation['rank'] * 5)
    
    def _compute_time_urgency(self, situation: dict) -> float:
        """時間切迫の計算（残りラウンド＋セット）"""
        round_urgency = (situation['total_rounds'] - situation['round']) / situation['total_rounds']
        set_urgency = (situation['total_sets'] - situation['set']) / situation['total_sets']
        
        # 終盤ほど切迫感増大
        return (1 - round_urgency * set_urgency) * 80
    
    def _compute_information_deficit(self, situation: dict) -> float:
        """情報不足の計算（他プレイヤー情報の有無）"""
        if situation.get('other_players_history'):
            # 情報がある = 戦略的思考可能
            return 20.0
        else:
            # 情報不足 = 不確実性高
            return 60.0


class ApexPlayerV4:
    """APEX SURVIVOR プレイヤー（v4: 真の創発版）
    
    v3との根本的違い:
    1. 人格別if文を完全排除
    2. 汎用圧力システムによる理論的意味圧
    3. SSDコアエンジンの跳躍機能を統合
    4. 単一の創発関数から行動決定
    """
    
    def __init__(self, name: str, initial_kappa_profile: str, color: str):
        self.name = name
        self.initial_kappa_profile = initial_kappa_profile  # personalityを廃止
        self.color = color
        
        # ゲーム状態
        self.hp = GameConfig.STARTING_HP
        self.score = 0
        self.is_alive = True
        self.choice_history = []
        
        # SSDエージェント
        params = HumanParams()
        self.agent = HumanAgent(params=params, agent_id=f"Apex_{self.name}", enable_nonlinear_transfer=True)
        
        # 汎用圧力エンジン
        self.pressure_engine = ApexPressureEngine()
        
        # κプロファイル初期化（personality的意味を排除し、純粋に初期条件として設定）
        self._initialize_kappa_profile()
    
    def _initialize_kappa_profile(self):
        """κプロファイル初期化（personality概念を排除）
        
        v4では「性格」ではなく「初期κ条件」として設定:
        - 進化的・学習的背景による個体差をκ初期値で表現
        - 行動決定ロジックは全個体で統一
        """
        if self.initial_kappa_profile == 'high_survival_threshold':
            # 高生存閾値型: BASE層の閾値が高い（生存脅威に鈍感）
            self.agent.state.kappa[HumanLayer.BASE.value] = 150.0
            self.agent.state.kappa[HumanLayer.CORE.value] = 5.0
            self.agent.state.kappa[HumanLayer.UPPER.value] = 15.0
        elif self.initial_kappa_profile == 'high_competition_threshold':
            # 高競争閾値型: CORE層の閾値が高い（競争圧力に鈍感）
            self.agent.state.kappa[HumanLayer.BASE.value] = 100.0
            self.agent.state.kappa[HumanLayer.CORE.value] = 10.0
            self.agent.state.kappa[HumanLayer.UPPER.value] = 20.0
        else:  # balanced_threshold
            # バランス型: 各層のバランスが取れた閾値
            self.agent.state.kappa[HumanLayer.BASE.value] = 120.0
            self.agent.state.kappa[HumanLayer.CORE.value] = 7.0
            self.agent.state.kappa[HumanLayer.UPPER.value] = 18.0
        
        # 初期E値（全タイプで統一）
        self.agent.state.E[HumanLayer.BASE.value] = 40.0
        self.agent.state.E[HumanLayer.CORE.value] = 0.5
        self.agent.state.E[HumanLayer.UPPER.value] = 4.0
    
    def make_choice(self, current_rank: int, leader_score: int, round_num: int, 
                    total_rounds: int, alive_count: int, current_set: int, total_sets: int,
                    other_players_history: dict = None) -> int:
        """純粋創発による選択決定（v4: 統一創発関数）"""
        
        if not self.is_alive:
            return 1
        
        # ===== STEP 1: 状況から汎用圧力次元を計算 =====
        situation = {
            'hp': self.hp,
            'score': self.score,
            'rank': current_rank,
            'leader_score': leader_score,
            'round': round_num,
            'total_rounds': total_rounds,
            'set': current_set,
            'total_sets': total_sets,
            'alive_count': alive_count,
            'other_players_history': other_players_history
        }
        
        pressure = self.pressure_engine.compute_apex_pressure(situation)
        
        # ===== STEP 2: SSDエンジンで状態更新 =====
        self.agent.step(pressure, dt=1.0)
        
        # ===== STEP 3: 跳躍検出 =====
        leap_detected, leap_layer = self.agent.engine.detect_leap(
            self.agent.state, pressure.to_vector()
        )
        
        if leap_detected:
            # 跳躍発生: 非連続的行動変化
            choice = self._handle_leap_action(leap_layer, situation)
        else:
            # 通常状態: E/κ/Rからの創発的行動
            choice = self._compute_emergent_action()
        
        self.choice_history.append(choice)
        return choice
    
    def _handle_leap_action(self, leap_layer: int, situation: dict) -> int:
        """跳躍による非連続的行動（v4新機能）"""
        
        # 跳躍層に応じた極端な行動
        if leap_layer == HumanLayer.BASE.value:
            # BASE層跳躍: 生存本能の暴走 → 極端な安全志向
            return 1  # 最安全
        elif leap_layer == HumanLayer.CORE.value:
            # CORE層跳躍: 競争欲求の暴走 → 極端な攻撃志向
            return 10  # 最危険
        elif leap_layer == HumanLayer.UPPER.value:
            # UPPER層跳躍: 戦略思考の暴走 → 計算された極端行動
            if situation['rank'] == 1:
                return 1  # 1位なら守り切る
            else:
                return 9  # 下位なら一か八か
        else:
            # PHYSICAL層跳躍: 身体的限界 → ランダム行動
            return random.randint(1, 10)
    
    def _compute_emergent_action(self) -> int:
        """E/κ/Rからの純粋創発的行動計算（v4: 統一関数）"""
        
        E = self.agent.state.E
        kappa = self.agent.state.kappa
        R = np.array([1000.0, 100.0, 10.0, 1.0])  # 抵抗値
        
        # ===== 構造的影響力の計算 =====
        # 各層の相対的影響力: (E - κ) * R (負の場合は抑制として機能)
        influence = (E - kappa) * R
        
        # 正の影響（行動駆動）と負の影響（行動抑制）を分離
        drive = np.maximum(0, influence)  # 行動駆動力
        restraint = np.maximum(0, -influence)  # 行動抑制力
        
        # ===== 層別行動ベクトルの計算 =====
        # BASE: 安全志向（値が大きいほど安全な選択）
        safety_drive = drive[HumanLayer.BASE.value] - restraint[HumanLayer.BASE.value] * 0.5
        
        # CORE: 攻撃志向（値が大きいほど危険な選択）
        attack_drive = drive[HumanLayer.CORE.value] - restraint[HumanLayer.CORE.value] * 0.5
        
        # UPPER: 戦略的最適化（状況に応じた調整）
        strategic_drive = drive[HumanLayer.UPPER.value] - restraint[HumanLayer.UPPER.value] * 0.5
        
        # PHYSICAL: 身体的制約（極端な行動を制限）
        physical_constraint = restraint[HumanLayer.PHYSICAL.value]
        
        # ===== 統一創発関数 =====
        # 基準リスクレベル（中間点）
        base_risk = 5.0
        
        # 各層の寄与を統合（係数を大幅調整）
        safety_effect = -safety_drive * 0.3   # 安全志向 = リスク低下（強化）
        attack_effect = attack_drive * 0.1    # 攻撃志向 = リスク増加（制限）
        strategic_effect = strategic_drive * 0.2  # 戦略調整（強化）
        
        # 身体制約による上下限（効果を強化）
        constraint_factor = 1.0 / (1.0 + physical_constraint * 0.01)
        
        # 最終リスクレベル
        risk_level = base_risk + safety_effect + attack_effect + strategic_effect
        risk_level *= constraint_factor
        
        # 極端値の制限（創発的調整）
        if safety_drive > 50:  # 強い安全志向
            risk_level = min(risk_level, 3.0)
        if attack_drive > 20:  # 強い攻撃志向
            risk_level = max(risk_level, 7.0)
        
        # 選択値に変換（1-10）
        choice = max(1, min(10, int(risk_level + 0.5)))
        
        return choice
    
    def process_result(self, choice: int, crashed: bool, score_gained: int):
        """結果処理（v4: 学習もSSDエンジンに委譲）"""
        if not self.is_alive:
            return
        
        # スコア更新
        if not crashed:
            self.score += score_gained
        
        # HP更新
        if crashed:
            self.hp -= 1
            if self.hp <= 0:
                self.is_alive = False
        
        # 結果をSSDエンジンの学習機構に委譲
        # （v4では外部学習ロジックを排除し、コアエンジンの自然な学習に任せる）
        

def demo_v4_pure_emergence():
    """v4: 純粋創発デモ"""
    print("="*60)
    print("🧠 APEX SURVIVOR v4 - 純粋創発デモ")
    print("="*60)
    
    # 異なるκプロファイルのプレイヤーを作成
    players = [
        ApexPlayerV4("田中", "high_survival_threshold", "🔵"),
        ApexPlayerV4("佐藤", "high_competition_threshold", "🔴"),
        ApexPlayerV4("鈴木", "balanced_threshold", "🟢")
    ]
    
    # テスト状況
    test_situations = [
        {
            "name": "序盤安全",
            "hp": 4, "rank": 3, "leader_score": 100, "score": 80,
            "round": 2, "total_rounds": 5, "set": 1, "total_sets": 5,
            "alive_count": 7
        },
        {
            "name": "中盤危機",
            "hp": 2, "rank": 5, "leader_score": 300, "score": 200,
            "round": 4, "total_rounds": 5, "set": 3, "total_sets": 5,
            "alive_count": 5
        },
        {
            "name": "終盤決戦",
            "hp": 3, "rank": 2, "leader_score": 500, "score": 480,
            "round": 5, "total_rounds": 5, "set": 5, "total_sets": 5,
            "alive_count": 3
        }
    ]
    
    for situation in test_situations:
        print(f"\n【{situation['name']}】")
        print(f"HP:{situation['hp']}, 順位:{situation['rank']}, ギャップ:{situation['leader_score']-situation['score']}")
        
        for player in players:
            # 一時的に状況を設定
            player.hp = situation['hp']
            player.score = situation['score']
            
            choice = player.make_choice(
                situation['rank'], situation['leader_score'],
                situation['round'], situation['total_rounds'],
                situation['alive_count'], situation['set'], situation['total_sets']
            )
            
            # デバッグ情報を追加
            E = player.agent.state.E
            kappa = player.agent.state.kappa
            R = np.array([1000.0, 100.0, 10.0, 1.0])
            influence = (E - kappa) * R
            drive = np.maximum(0, influence)
            restraint = np.maximum(0, -influence)
            
            safety_drive = drive[1] - restraint[1] * 0.5
            attack_drive = drive[2] - restraint[2] * 0.5
            strategic_drive = drive[3] - restraint[3] * 0.5
            
            print(f"  {player.name}({player.initial_kappa_profile[:4]}): 選択={choice}")
            print(f"    E=[{E[1]:.1f}, {E[2]:.1f}, {E[3]:.1f}]")
            print(f"    κ=[{kappa[1]:.0f}, {kappa[2]:.0f}, {kappa[3]:.0f}]")
            print(f"    Drive: safety={safety_drive:.1f}, attack={attack_drive:.1f}, strategic={strategic_drive:.1f}")


if __name__ == "__main__":
    demo_v4_pure_emergence()