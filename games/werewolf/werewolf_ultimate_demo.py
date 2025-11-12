"""
人狼ゲーム 統合デモ
Werewolf Game - Ultimate Demo

【Phase 10.4 完全版】
理論的コア（v10.2.1）+ 全フレーバー統合:
1. 可視化（werewolf_visualizer.py）
2. 自然言語説明（werewolf_narrator.py）
3. 拡張役職（werewolf_extended_roles.py）

作成日: 2025年11月7日
バージョン: 10.4 Ultimate
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Dict, Tuple, Optional

# 理論的コア (同じフォルダから)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from werewolf_game_v10_2_1_causal import (
    PlayerV10Causal,
    WerewolfGameV10Causal
)

# 拡張役職
from werewolf_extended_roles import (
    ExtendedPlayer,
    ExtendedGame,
    Role
)

# ナレーション
from werewolf_narrator import (
    NarrativeGenerator,
    PlayerNarrator
)

# 可視化
from werewolf_visualizer import WerewolfVisualizer

# 親ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

# coreモジュールのパス追加
core_path = os.path.join(grandparent_dir, 'core')
sys.path.insert(0, core_path)

from ssd_engine.core.ssd_human_module import HumanPressure


class UltimatePlayer(ExtendedPlayer):
    """
    究極プレイヤー: 拡張役職 + ナレーション機能
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.narrative_generator = NarrativeGenerator()
    
    def explain_vote_reason_ultimate(self, target_id: int, target_name: str = None, 
                                     verbose: bool = True) -> str:
        """
        投票理由を自然言語で説明（役職情報含む）
        """
        if target_name is None:
            target_name = f"Player{target_id}"
        
        if target_id not in self.last_decision_explanation:
            return f"{target_name}については観測データがありません。"
        
        info = self.last_decision_explanation[target_id]
        pressure = info['meaning_pressure']
        primary_concept = info['primary_concept']
        
        # 脅威記述
        threat_desc = self.narrative_generator.pressure_to_threat_description(
            pressure.base, pressure.core, pressure.upper
        )
        
        # 役職による情報追加
        role_info = ""
        if self.role == Role.FORTUNE_TELLER.value and target_id in self.divine_results:
            divine_result = self.divine_results[target_id]
            role_info = f"（占い結果: {divine_result}）"
        
        if not verbose:
            if primary_concept:
                return f"{target_name}は「{primary_concept}」パターンに該当し、{threat_desc}と判断{role_info}"
            else:
                return f"{target_name}は初見のパターンですが、{threat_desc}と判断{role_info}"
        
        # 詳細版
        explanation = [
            f"【{target_name}への投票理由】",
            f"自分: {self.name}（{self.role}）",
            ""
        ]
        
        # 観測シグナル
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
                concept_desc = self.narrative_generator.concept_to_description(matching_concept)
                explanation.append(f"2. 活性化した概念:")
                explanation.append(f"   {concept_desc}")
                explanation.append("")
        else:
            explanation.append(f"2. 概念マッチング:")
            explanation.append(f"   初見のパターン")
            explanation.append("")
        
        # 役職能力による情報
        if self.role == Role.FORTUNE_TELLER.value and target_id in self.divine_results:
            explanation.append(f"3. 占い結果:")
            explanation.append(f"   {target_name}は{self.divine_results[target_id]}判定")
            explanation.append("")
        
        # 意味圧
        explanation.append(f"{'4' if self.role != Role.FORTUNE_TELLER.value or target_id not in self.divine_results else '4'}. 感じた脅威:")
        explanation.append(f"   {threat_desc}")
        explanation.append(f"   - 生存脅威（BASE層）: {pressure.base:.2f}")
        explanation.append(f"   - 規範的葛藤（CORE層）: {pressure.core:.2f}")
        explanation.append(f"   - 理念的不協和（UPPER層）: {pressure.upper:.2f}")
        explanation.append("")
        
        # 内部状態
        E = self.agent.state.E
        internal_desc = self.narrative_generator.E_to_internal_state(E)
        explanation.append(f"5. 自分の心理状態:")
        explanation.append(f"   {internal_desc}")
        
        # 結論
        explanation.append(f"\n6. 結論:")
        explanation.append(f"   これらの理由から、{target_name}に投票しました。")
        
        return "\n".join(explanation)


class UltimateGame(ExtendedGame):
    """
    究極ゲーム: 拡張役職 + ナレーション + 可視化
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # プレイヤーをUltimate版に差し替え
        for pid, player in self.players.items():
            ultimate = UltimatePlayer(
                player_id=pid,
                name=player.name,
                role=player.role
            )
            # 状態を引き継ぎ
            ultimate.agent = player.agent
            ultimate.structured_memory = player.structured_memory
            ultimate.learned_concepts = player.learned_concepts
            ultimate.divine_results = player.divine_results
            ultimate.medium_results = player.medium_results
            self.players[pid] = ultimate
        
        # 可視化器
        self.visualizer = WerewolfVisualizer(self)
        self.narrator = NarrativeGenerator()
    
    def narrate_night_phase(self):
        """夜フェーズのナレーション"""
        print(f"\n{'='*70}")
        print(f"【Night {self.day_count}】")
        print(f"{'='*70}")
        print("夜が訪れました。各役職が能力を使用します...\n")
    
    def narrate_day_start(self, day: int):
        """昼フェーズ開始のナレーション"""
        alive = self.get_alive_players()
        werewolves = self.get_alive_werewolves()
        
        print(f"\n{'='*70}")
        print(f"【Day {day}】")
        print(f"{'='*70}")
        print(f"生存者: {len(alive)}人")
        print(f"人狼陣営: {len(werewolves) + sum(1 for p in self.players.values() if p.is_alive and p.role == Role.MADMAN.value)}人（正体不明）")
        print(f"村人陣営: {len(alive) - len(werewolves) - sum(1 for p in self.players.values() if p.is_alive and p.role == Role.MADMAN.value)}人")
        print()
    
    def narrate_execution(self, executed_id: int):
        """処刑のナレーション"""
        executed = self.players[executed_id]
        print(f"\n【処刑】")
        print(f"{executed.name}が投票により処刑されました。")
        print(f"正体: {executed.role}")
        
        # 役職能力の公開
        if executed.role == Role.FORTUNE_TELLER.value and executed.divine_results:
            print(f"\n{executed.name}の占い結果:")
            for pid, result in executed.divine_results.items():
                print(f"  - {self.players[pid].name}: {result}")
        
        print()
    
    def play_ultimate_game(self, max_days: int = 10, 
                          detailed_votes: bool = False,
                          show_night_actions: bool = True,
                          visualize_each_day: bool = False) -> str:
        """
        究極版ゲームをプレイ
        
        Args:
            max_days: 最大日数
            detailed_votes: 詳細な投票理由を表示
            show_night_actions: 夜の行動を表示
            visualize_each_day: 各日ごとに可視化
        """
        print("="*70)
        print("人狼ゲーム v10.4 Ultimate (統合デモ)")
        print("="*70)
        print("\n【統合機能】")
        print("- SSD理論コア（v10.2.1）")
        print("- 拡張役職（占い師、霊媒師、狂人）")
        print("- 自然言語説明（XAI）")
        print("- リアルタイム可視化（オプション）")
        print()
        
        # 役職構成を表示
        role_counts = {}
        for player in self.players.values():
            role_counts[player.role] = role_counts.get(player.role, 0) + 1
        
        print("【役職構成】")
        for role, count in role_counts.items():
            print(f"  {role}: {count}人")
        print()
        
        for day in range(1, max_days + 1):
            # 夜フェーズ
            if day > 1:
                self.narrate_night_phase()
                
                alive = self.get_alive_players()
                
                # 人狼の襲撃
                werewolves = [pid for pid in alive if self.players[pid].role == Role.WEREWOLF.value]
                if werewolves:
                    villagers = [pid for pid in alive 
                                if pid not in werewolves and self.players[pid].role != Role.MADMAN.value]
                    if villagers:
                        target = np.random.choice(villagers)
                        self.night_death = target
                        if show_night_actions:
                            print(f"人狼が{self.players[target].name}を襲撃しました...")
                
                # 占い師
                fortune_tellers = [pid for pid in alive 
                                  if self.players[pid].role == Role.FORTUNE_TELLER.value]
                for ft_id in fortune_tellers:
                    candidates = [pid for pid in alive if pid != ft_id]
                    if candidates:
                        undivinied = [c for c in candidates 
                                    if c not in self.players[ft_id].divine_results]
                        if undivinied:
                            target = np.random.choice(undivinied)
                        else:
                            target = np.random.choice(candidates)
                        
                        result = self.players[ft_id].use_ability(target, self)
                        if show_night_actions:
                            print(f"{result}")
                
                # 霊媒師
                mediums = [pid for pid in alive if self.players[pid].role == Role.MEDIUM.value]
                for med_id in mediums:
                    result = self.players[med_id].use_ability(-1, self)
                    if show_night_actions and "まだ" not in result:
                        print(f"{result}")
                
                print()
            
            # 昼フェーズ
            self.narrate_day_start(day)
            
            alive = self.get_alive_players()
            
            # 夜の死亡発表
            if self.night_death is not None:
                victim = self.players[self.night_death]
                victim.is_alive = False
                print(f"【襲撃】{victim.name}が人狼に襲撃されました（{victim.role}）\n")
                self.night_death = None
                alive = self.get_alive_players()
            
            # 役職公開の判断
            print("【役職公開】")
            any_claim = False
            for pid in alive:
                claimed = self.players[pid].decide_claim()
                if claimed and self.players[pid].claimed_role != claimed:
                    self.players[pid].claimed_role = claimed
                    print(f"{self.players[pid].name}が{claimed}を名乗り出ました")
                    any_claim = True
            if not any_claim:
                print("（役職公開なし）")
            print()
            
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
            
            # 可視化（オプション）
            if visualize_each_day:
                self.visualizer.record_day(day)
                self.visualizer.plot_signal_heatmap(day=day)
            
            # 投票
            print("【投票フェーズ】")
            votes: Dict[int, int] = {}
            
            for voter_id in alive:
                target = self.players[voter_id].decide_vote_with_info(alive)
                if target != -1:
                    votes[target] = votes.get(target, 0) + 1
                    
                    if detailed_votes:
                        print(self.players[voter_id].explain_vote_reason_ultimate(
                            target, self.players[target].name, verbose=True))
                        print()
                    else:
                        reason = self.players[voter_id].explain_vote_reason_ultimate(
                            target, self.players[target].name, verbose=False)
                        print(f"{self.players[voter_id].name} → {self.players[target].name}: {reason}")
            
            print()
            
            # 処刑
            if votes:
                executed_id = max(votes.items(), key=lambda x: x[1])[0]
                executed = self.players[executed_id]
                executed_role = executed.role
                executed.is_alive = False
                
                self.execution_history.append(executed_id)
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
                print(f"{'='*70}")
                print(f"【ゲーム終了】")
                print(f"{'='*70}")
                print(f"結果: {result}")
                print()
                
                # 最終学習
                for player in self.players.values():
                    outcome = 'won' if (
                        (player.role in [Role.WEREWOLF.value, Role.MADMAN.value] and result == 'werewolf_win') or
                        (player.role not in [Role.WEREWOLF.value, Role.MADMAN.value] and result == 'villager_win')
                    ) else 'lost'
                    player.learn_from_outcome(outcome, current_time=self.current_time)
                
                return result
        
        return 'villager_win'


def demo_ultimate():
    """究極版デモ"""
    print("="*70)
    print("人狼ゲーム 統合デモ（Ultimate Edition）")
    print("="*70)
    print("\n【Phase 10完全版】")
    print("- 理論的コア: v10.2.1（SSD完全統合）")
    print("- フレーバー1: 可視化（概念ネットワーク、意味圧フロー）")
    print("- フレーバー2: 自然言語説明（XAI対応）")
    print("- フレーバー3: 拡張役職（占い師、霊媒師、狂人）")
    print()
    
    # 学習フェーズ
    print("="*70)
    print("学習フェーズ: 5ゲームをプレイ")
    print("="*70)
    print()
    
    for game_num in range(5):
        print(f"学習ゲーム {game_num + 1}/5...")
        game = ExtendedGame(
            num_players=9,
            num_werewolves=2,
            has_fortune_teller=True,
            has_medium=True,
            has_madman=True,
            has_knight=False
        )
        
        if game_num > 0:
            # 学習を引き継ぎ
            for pid in range(9):
                if pid < len(prev_players):
                    game.players[pid].structured_memory = prev_players[pid].structured_memory
                    game.players[pid].learned_concepts = prev_players[pid].learned_concepts
                    game.players[pid].agent.state.kappa = prev_players[pid].agent.state.kappa
        
        prev_players = {pid: p for pid, p in game.players.items()}
        game.play_extended_game(max_days=10, verbose=False)
    
    print("\n学習完了！\n")
    
    # 本番ゲーム
    print("="*70)
    print("本番ゲーム開始（全機能統合）")
    print("="*70)
    print()
    
    final_game = UltimateGame(
        num_players=9,
        num_werewolves=2,
        has_fortune_teller=True,
        has_medium=True,
        has_madman=True,
        has_knight=False
    )
    
    # 学習を引き継ぎ
    for pid in range(9):
        if pid < len(prev_players):
            final_game.players[pid].structured_memory = prev_players[pid].structured_memory
            final_game.players[pid].learned_concepts = prev_players[pid].learned_concepts
            final_game.players[pid].agent.state.kappa = prev_players[pid].agent.state.kappa
    
    result = final_game.play_ultimate_game(
        max_days=10,
        detailed_votes=False,  # True にすると詳細な投票理由
        show_night_actions=True,
        visualize_each_day=False  # True にすると各日ごとに可視化
    )
    
    # 最終統計
    print("="*70)
    print("最終統計")
    print("="*70)
    
    for pid, player in final_game.players.items():
        stats = player.get_learning_stats()
        print(f"\n{player.name} ({player.role}):")
        print(f"  生存: {'はい' if player.is_alive else 'いいえ'}")
        print(f"  E状態: BASE={stats['E'][1]:.3f}, CORE={stats['E'][2]:.3f}, UPPER={stats['E'][3]:.3f}")
        print(f"  概念数: {stats['n_concepts']}個")
        
        if player.role == Role.FORTUNE_TELLER.value:
            print(f"  占い回数: {len(player.divine_results)}回")
            werewolf_found = sum(1 for r in player.divine_results.values() if r == "werewolf")
            print(f"  人狼発見: {werewolf_found}人")
        
        elif player.role == Role.MEDIUM.value:
            print(f"  霊媒回数: {len(player.medium_results)}回")
        
        elif player.role == Role.WEREWOLF.value:
            print(f"  罪悪感: {stats['internal_guilt']:.3f}")
        
        if stats['top_concepts']:
            print(f"  主要概念: {stats['top_concepts'][0][0]}")
    
    # 可視化生成
    print("\n" + "="*70)
    print("可視化生成中...")
    print("="*70)
    print()
    
    # E状態の時系列（人狼）
    werewolves = [pid for pid, p in final_game.players.items() if p.role == Role.WEREWOLF.value]
    if werewolves and final_game.visualizer.E_history[werewolves[0]]:
        print("1. E状態の時系列変化（人狼の罪悪感蓄積）")
        final_game.visualizer.plot_E_evolution(werewolves[0])
    
    # 概念ネットワーク
    for pid, player in final_game.players.items():
        if player.learned_concepts:
            print(f"\n2. 概念ネットワーク ({player.name})")
            final_game.visualizer.plot_concept_network(pid)
            break
    
    print("\n" + "="*70)
    print("Phase 10 完全統合デモ完了")
    print("="*70)
    print("\n【達成事項】")
    print("- SSD理論（構造主観力学）の完全実装")
    print("- 構造化記憶と概念形成")
    print("- 意味圧システムの統合")
    print("- 拡張役職システム")
    print("- XAI（説明可能AI）対応")
    print("- 可視化システム")
    print("\n理論的整合性を保ちながら、実用的な機能を追加する")
    print("「フレーバー設計」の成功例として完成しました。")


if __name__ == "__main__":
    demo_ultimate()
