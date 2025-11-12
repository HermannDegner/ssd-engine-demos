"""
人狼ゲーム 発話生成モジュール
Werewolf Game Narrator

【Phase 10.4b: 発話生成】
理論的コア（v10.2.1）に自然言語説明を追加:
1. 投票理由の自然言語説明
2. 内部状態の言語化
3. 概念の言語的記述
4. ゲーム進行のナレーション
5. XAI（説明可能AI）としての応用

作成日: 2025年11月7日
バージョン: 10.4b
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional

# 理論的コアをインポート
# 理論的コア (同じフォルダから)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from werewolf_game_v10_2_1_causal import (
    PlayerV10Causal,
    WerewolfGameV10Causal
)

# 親ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

# coreモジュールのパス追加
core_path = os.path.join(grandparent_dir, 'core')
sys.path.insert(0, core_path)

# extensionsモジュールのパス追加
extensions_path = os.path.join(grandparent_dir, 'extensions')
sys.path.insert(0, extensions_path)

from extensions.ssd_memory_structure import Concept
from core.ssd_human_module import HumanLayer


class NarrativeGenerator:
    """
    自然言語による説明生成
    
    【設計原則】
    - SSD理論の用語を一般的な言葉に翻訳
    - 因果関係を明確に説明
    - 複数レベルの詳細度（簡潔版・詳細版）
    """
    
    @staticmethod
    def signal_to_behavior(signals: np.ndarray) -> str:
        """
        シグナルベクトルを行動記述に変換
        
        Args:
            signals: [姿勢, 表情, 音声, 攻撃性, 防御性, 協調性, 理念] (7次元)
        
        Returns:
            自然言語記述
        """
        posture, expression, voice, aggression, defense, cooperation, ideology = signals
        
        parts = []
        
        # 姿勢
        if posture > 0.6:
            parts.append("堂々とした態度")
        elif posture < 0.3:
            parts.append("萎縮した様子")
        
        # 表情
        if expression > 0.6:
            parts.append("緊張した表情")
        elif expression < 0.3:
            parts.append("穏やかな表情")
        
        # 音声
        if voice > 0.6:
            parts.append("強い口調")
        elif voice < 0.3:
            parts.append("小さな声")
        
        # 社会的行動
        if aggression > 0.5:
            parts.append("攻撃的な発言")
        elif defense > 0.5:
            parts.append("防御的な態度")
        elif cooperation > 0.5:
            parts.append("協調的な姿勢")
        
        # 理念
        if ideology > 0.5:
            parts.append("理念的な主張")
        
        if parts:
            return "、".join(parts)
        else:
            return "特徴的な行動なし"
    
    @staticmethod
    def E_to_internal_state(E: np.ndarray) -> str:
        """
        E（未処理圧）を内部状態の記述に変換
        
        Args:
            E: [PHYSICAL, BASE, CORE, UPPER] (4次元)
        
        Returns:
            自然言語記述
        """
        physical, base, core, upper = E
        
        parts = []
        
        if physical > 0.5:
            parts.append("身体的な疲労")
        
        if base > 0.3:
            parts.append("生存への不安")
        elif base > 0.1:
            parts.append("軽度の緊張")
        
        if core > 0.5:
            parts.append("強い罪悪感")
        elif core > 0.3:
            parts.append("規範的な葛藤")
        elif core > 0.1:
            parts.append("社会的なプレッシャー")
        
        if upper > 0.3:
            parts.append("理念的な迷い")
        elif upper > 0.1:
            parts.append("価値観の揺らぎ")
        
        if parts:
            return "、".join(parts) + "を感じている"
        else:
            return "心理的に安定している"
    
    @staticmethod
    def pressure_to_threat_description(pressure_base: float, pressure_core: float, 
                                      pressure_upper: float) -> str:
        """
        意味圧を脅威の記述に変換
        
        Args:
            pressure_base: BASE層意味圧（生存脅威）
            pressure_core: CORE層意味圧（規範的葛藤）
            pressure_upper: UPPER層意味圧（理念的不協和）
        
        Returns:
            脅威レベルの記述
        """
        # 総合脅威レベル
        total_threat = pressure_base + 0.5 * pressure_core + 0.3 * pressure_upper
        
        if total_threat > 0.7:
            level = "非常に危険"
        elif total_threat > 0.4:
            level = "やや疑わしい"
        elif total_threat > 0.1:
            level = "少し気になる"
        elif total_threat > -0.3:
            level = "中立的"
        else:
            level = "信頼できる"
        
        # 詳細
        details = []
        if abs(pressure_base) > 0.3:
            if pressure_base > 0:
                details.append("直接的な脅威を感じる")
            else:
                details.append("安心感を覚える")
        
        if abs(pressure_core) > 0.3:
            if pressure_core > 0:
                details.append("何か隠している印象")
            else:
                details.append("誠実な印象")
        
        if abs(pressure_upper) > 0.3:
            if pressure_upper > 0:
                details.append("価値観の違いを感じる")
            else:
                details.append("価値観が合う")
        
        if details:
            return f"{level}（{'/'.join(details)}）"
        else:
            return level
    
    @staticmethod
    def concept_to_description(concept: Concept) -> str:
        """
        概念を自然言語で記述
        
        Args:
            concept: 概念オブジェクト
        
        Returns:
            概念の説明文
        """
        name = concept.name
        n_members = concept.cluster.n_memories
        avg_outcome = concept.cluster.avg_outcome
        variance = concept.cluster.variance
        
        # 名前を解析
        parts = name.split('_')
        
        # プロトタイプシグナル
        prototype = concept.cluster.prototype_signal
        behavior = NarrativeGenerator.signal_to_behavior(prototype)
        
        # 結果の解釈
        if avg_outcome > 0.5:
            outcome_desc = "危険なパターン"
        elif avg_outcome > 0:
            outcome_desc = "やや疑わしいパターン"
        elif avg_outcome > -0.5:
            outcome_desc = "中立的なパターン"
        else:
            outcome_desc = "安全なパターン"
        
        # 確信度
        if variance < 0.1:
            confidence = "明確"
        elif variance < 0.3:
            confidence = "ある程度明確"
        else:
            confidence = "曖昧"
        
        return (f"「{name}」概念:\n"
                f"  特徴: {behavior}\n"
                f"  評価: {outcome_desc}\n"
                f"  確信度: {confidence}（{n_members}件の経験から形成）")


class PlayerNarrator(PlayerV10Causal):
    """
    発話機能を持つプレイヤー（v10.2.1を拡張）
    
    理論的コアは継承、発話機能のみ追加
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.narrative_generator = NarrativeGenerator()
    
    def explain_vote_reason(self, target_id: int, target_name: str = None, verbose: bool = True) -> str:
        """
        投票理由を自然言語で説明
        
        Args:
            target_id: 投票対象
            target_name: 対象の名前（オプション）
            verbose: 詳細版かどうか
        
        Returns:
            投票理由の説明文
        """
        if target_name is None:
            target_name = f"Player{target_id}"
        
        if target_id not in self.last_decision_explanation:
            return f"{target_name}については観測データがありません。"
        
        info = self.last_decision_explanation[target_id]
        pressure = info['meaning_pressure']
        primary_concept = info['primary_concept']
        
        # 簡潔版
        threat_desc = self.narrative_generator.pressure_to_threat_description(
            pressure.base, pressure.core, pressure.upper
        )
        
        if not verbose:
            if primary_concept:
                return f"{target_name}は「{primary_concept}」パターンに該当し、{threat_desc}と判断しました。"
            else:
                return f"{target_name}は初見のパターンですが、{threat_desc}と判断しました。"
        
        # 詳細版
        explanation = [
            f"【{target_name}への投票理由】",
            ""
        ]
        
        # 観測したシグナル
        if target_id in self.observation_history and self.observation_history[target_id]:
            signals = self.observation_history[target_id][-1]
            behavior = self.narrative_generator.signal_to_behavior(signals)
            explanation.append(f"1. 観測した行動:")
            explanation.append(f"   {behavior}")
            explanation.append("")
        
        # 概念マッチング
        if primary_concept:
            matching_concept = next((c for c in self.learned_concepts if c.name == primary_concept), None)
            if matching_concept:
                explanation.append(f"2. 活性化した概念:")
                explanation.append(f"   {self.narrative_generator.concept_to_description(matching_concept)}")
                explanation.append("")
        else:
            explanation.append(f"2. 概念マッチング:")
            explanation.append(f"   初見のパターンです。既存の概念には該当しませんでした。")
            explanation.append("")
        
        # 意味圧の解釈
        explanation.append(f"3. 感じた脅威:")
        explanation.append(f"   {threat_desc}")
        explanation.append(f"   - 生存脅威（BASE層）: {pressure.base:.2f}")
        explanation.append(f"   - 規範的葛藤（CORE層）: {pressure.core:.2f}")
        explanation.append(f"   - 理念的不協和（UPPER層）: {pressure.upper:.2f}")
        explanation.append("")
        
        # 自分の内部状態
        E = self.agent.state.E
        internal_desc = self.narrative_generator.E_to_internal_state(E)
        explanation.append(f"4. 自分の心理状態:")
        explanation.append(f"   {internal_desc}")
        explanation.append(f"   - E_BASE: {E[1]:.3f}（生存不安）")
        explanation.append(f"   - E_CORE: {E[2]:.3f}（罪悪感）")
        explanation.append(f"   - E_UPPER: {E[3]:.3f}（価値観の揺らぎ）")
        explanation.append("")
        
        # 結論
        explanation.append(f"5. 結論:")
        explanation.append(f"   これらの理由から、{target_name}に投票しました。")
        
        return "\n".join(explanation)
    
    def describe_self(self) -> str:
        """
        自分自身の状態を説明
        
        Returns:
            自己説明文
        """
        E = self.agent.state.E
        kappa = self.agent.state.kappa
        
        description = [
            f"【{self.name}の状態】",
            ""
        ]
        
        # 心理状態
        internal_desc = self.narrative_generator.E_to_internal_state(E)
        description.append(f"心理状態: {internal_desc}")
        description.append("")
        
        # 学習進度
        avg_kappa = np.mean(kappa)
        if avg_kappa > 0.8:
            learning_desc = "経験豊富"
        elif avg_kappa > 0.5:
            learning_desc = "ある程度学習済み"
        else:
            learning_desc = "学習中"
        
        description.append(f"学習進度: {learning_desc}（平均κ={avg_kappa:.2f}）")
        description.append("")
        
        # 概念
        if self.learned_concepts:
            description.append(f"形成した概念: {len(self.learned_concepts)}個")
            for concept in self.learned_concepts[:3]:  # 上位3つ
                description.append(f"  - {concept.name}")
        else:
            description.append("形成した概念: なし（経験不足）")
        
        return "\n".join(description)


class NarrativeGame(WerewolfGameV10Causal):
    """
    ナレーション付き人狼ゲーム
    
    理論的コアは継承、ナレーションのみ追加
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # プレイヤーをNarrator版に差し替え
        for pid, player in self.players.items():
            narrator = PlayerNarrator(
                player_id=pid,
                name=player.name,
                role=player.role
            )
            # 状態を引き継ぎ
            narrator.agent = player.agent
            narrator.structured_memory = player.structured_memory
            narrator.learned_concepts = player.learned_concepts
            self.players[pid] = narrator
        
        self.narrator = NarrativeGenerator()
    
    def narrate_day_start(self, day: int):
        """昼フェーズ開始のナレーション"""
        alive = self.get_alive_players()
        werewolves = self.get_alive_werewolves()
        
        print(f"\n{'='*70}")
        print(f"【Day {day}】")
        print(f"{'='*70}")
        print(f"生存者: {len(alive)}人")
        print(f"人狼: {len(werewolves)}人（正体不明）")
        print(f"村人: {len(alive) - len(werewolves)}人")
        print()
    
    def narrate_execution(self, executed_id: int):
        """処刑のナレーション"""
        executed = self.players[executed_id]
        print(f"\n【処刑】")
        print(f"{executed.name}が投票により処刑されました。")
        print(f"正体: {executed.role}")
        print()
    
    def narrate_vote_details(self, voter_id: int, target_id: int, detailed: bool = False):
        """投票詳細のナレーション"""
        voter = self.players[voter_id]
        target = self.players[target_id]
        
        if detailed:
            print(voter.explain_vote_reason(target_id, target_name=target.name, verbose=True))
        else:
            explanation = voter.explain_suspicion(target_id)
            reason = voter.explain_vote_reason(target_id, target_name=target.name, verbose=False)
            print(f"{voter.name} → {target.name}: {reason}")
    
    def play_game_with_narrative(self, max_days: int = 10, detailed_votes: bool = False) -> str:
        """ナレーション付きでゲームをプレイ"""
        print("="*70)
        print("人狼ゲーム v10.4b (ナレーション版)")
        print("="*70)
        print("\nSSD理論に基づく意思決定プロセスを自然言語で説明します。")
        print()
        
        for day in range(1, max_days + 1):
            self.narrate_day_start(day)
            
            # 昼フェーズ実行
            alive = self.get_alive_players()
            
            # シグナル生成
            signals_map = {pid: self.players[pid].generate_signals() for pid in alive}
            
            # 解釈フェーズ
            for pid in alive:
                self.players[pid].last_pressure_map.clear()
            
            for observer_id in alive:
                for target_id in alive:
                    if target_id != observer_id:
                        self.players[observer_id].observe_player(
                            target_id, signals_map[target_id], self.current_time
                        )
            
            # 蓄積フェーズ
            for pid in alive:
                self.players[pid].accumulate_pressures(dt=1.0)
            
            # 投票
            print("【投票フェーズ】")
            votes = {}
            for voter_id in alive:
                target = self.players[voter_id].decide_vote(alive)
                if target != -1:
                    votes[target] = votes.get(target, 0) + 1
                    self.narrate_vote_details(voter_id, target, detailed=detailed_votes)
            
            # 処刑
            if votes:
                executed_id = max(votes.items(), key=lambda x: x[1])[0]
                executed = self.players[executed_id]
                executed_role = executed.role
                executed.is_alive = False
                
                self.narrate_execution(executed_id)
                
                # 学習
                executed.learn_from_outcome('executed', current_time=self.current_time)
                for pid in alive:
                    if pid != executed_id:
                        self.players[pid].learn_from_outcome(
                            'survived',
                            executed_player_id=executed_id,
                            executed_player_role=executed_role,
                            current_time=self.current_time
                        )
            
            self.current_time += 1.0
            self.day_count = day
            
            # 終了判定
            result = self.check_game_end()
            if result:
                print(f"\n{'='*70}")
                print(f"【ゲーム終了】")
                print(f"{'='*70}")
                print(f"結果: {result}")
                print()
                
                # 最終学習
                for player in self.players.values():
                    outcome = 'won' if (
                        (player.role == 'werewolf' and result == 'werewolf_win') or
                        (player.role == 'villager' and result == 'villager_win')
                    ) else 'lost'
                    player.learn_from_outcome(outcome, current_time=self.current_time)
                
                return result
        
        return 'villager_win'


def demo_narrator():
    """ナレーション版デモ"""
    print("="*70)
    print("人狼ゲーム ナレーション版デモ")
    print("="*70)
    print("\n【特徴】")
    print("- 投票理由を自然言語で説明")
    print("- SSD理論の用語を一般的な言葉に翻訳")
    print("- 意思決定プロセスの完全な透明性")
    print()
    
    # 学習のため数ゲームをプレイ
    print("学習のため3ゲームをプレイ中...\n")
    for game_num in range(3):
        game = WerewolfGameV10Causal(num_players=7, num_werewolves=2)
        if game_num > 0:
            # 学習を引き継ぎ
            for pid in range(7):
                game.players[pid].structured_memory = prev_players[pid].structured_memory
                game.players[pid].learned_concepts = prev_players[pid].learned_concepts
        prev_players = {pid: p for pid, p in game.players.items()}
        game.play_game(max_days=5, verbose=False)
    
    # ナレーション版でプレイ
    print("\n" + "="*70)
    print("ナレーション版ゲーム開始")
    print("="*70)
    
    narrative_game = NarrativeGame(num_players=7, num_werewolves=2)
    # 学習を引き継ぎ
    for pid in range(7):
        narrative_game.players[pid].structured_memory = prev_players[pid].structured_memory
        narrative_game.players[pid].learned_concepts = prev_players[pid].learned_concepts
    
    result = narrative_game.play_game_with_narrative(max_days=5, detailed_votes=False)
    
    print("\n" + "="*70)
    print("プレイヤーの状態")
    print("="*70)
    
    for pid, player in narrative_game.players.items():
        if player.is_alive:
            print(player.describe_self())
            print()
    
    print("="*70)
    print("Phase 10.4b 完了: 自然言語説明機能追加")
    print("="*70)
    print("\n理論的コア（v10.2.1）は不変、ナレーションのみ追加")
    print("XAI（説明可能AI）としての応用可能性を実証")


if __name__ == "__main__":
    demo_narrator()
