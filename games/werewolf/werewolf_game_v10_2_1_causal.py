"""
人狼ゲーム v10.2.1: 因果律の修正版
Werewolf Game v10.2.1: Corrected Causality

【Phase 10.2.1 - 因果律の修正】
v10.2の構造的矛盾を解消:
1. 「観測（意味圧の解釈）」と「蓄積（E更新）」の完全分離
2. 投票ロジックの修正（Eではなく意味圧に基づく）
3. SSD理論との完全整合

作成日: 2025年11月7日
バージョン: 10.2.1 (Causal Correction)
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional
import argparse
import time

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

from ssd_engine.extensions.ssd_memory_structure import (
    StructuredMemoryStore,
    Concept,
    cosine_similarity
)

# SSDコアモジュールのインポート
from ssd_engine.core.ssd_human_module import HumanAgent, HumanLayer, HumanPressure


class PlayerV10Causal:
    """
    v10.2.1プレイヤー: 因果律を修正したSSD統合版
    
    【修正点】
    - observe_player()はagent.step()を呼ばず、HumanPressureを返すだけ
    - 意味圧の蓄積はday_phase側で一括処理
    - 投票は蓄積されたEではなく、解釈された意味圧に基づく
    """
    
    def __init__(self, 
                 player_id: int,
                 name: str,
                 role: str,
                 max_clusters: int = 30):
        self.player_id = player_id
        self.name = name
        self.role = role  # ゲームマスターのみが知る
        self.is_alive = True
        
        # === SSDコアエンジンの統合 ===
        self.agent = HumanAgent(agent_id=str(player_id))
        
        # === 構造化記憶システム ===
        self.structured_memory = StructuredMemoryStore(
            max_clusters=max_clusters,
            cluster_threshold=0.35,
            min_concept_size=3
        )
        
        self.learned_concepts: List[Concept] = []
        
        # === ゲーム固有の状態 ===
        self.observation_history: Dict[int, List[np.ndarray]] = {}
        
        # 統計
        self.games_played = 0
        self.times_executed = 0
        self.times_survived = 0
        
        # 説明可能性
        self.last_decision_explanation: Dict = {}
        
        # 【NEW】意味圧マップ（観測結果を記録）
        self.last_pressure_map: Dict[int, HumanPressure] = {}
        
        # 人狼の罪悪感蓄積（プレイヤーは意識しない）
        self._internal_guilt = 0.0
    
    def observe_player(self, 
                      target_id: int,
                      target_signals: np.ndarray,
                      current_time: float) -> HumanPressure:
        """
        他プレイヤーを観測し、意味圧を解釈
        
        【修正点】
        - agent.step()を呼ばない（状態を更新しない）
        - HumanPressureオブジェクトを返すだけ
        """
        # 構造化記憶による解釈
        kappa = self.agent.state.kappa
        pressure_interpretation = self.structured_memory.interpret_with_structure(
            signal=target_signals,
            kappa=kappa,
            use_concepts=True
        )
        
        # 観測履歴に記録
        if target_id not in self.observation_history:
            self.observation_history[target_id] = []
        self.observation_history[target_id].append(target_signals.copy())
        
        # 意味圧に変換
        meaning_pressure = HumanPressure(
            physical=0.0,
            base=pressure_interpretation[1],   # BASE層: 生存脅威
            core=pressure_interpretation[2],   # CORE層: 規範的葛藤
            upper=pressure_interpretation[3]   # UPPER層: 理念的不協和
        )
        
        # 説明用に記録
        activated_concepts = [
            c for c in self.learned_concepts
            if c.matches(target_signals)
        ]
        
        primary_concept = activated_concepts[0].name if activated_concepts else None
        
        # 意味圧マップに記録
        self.last_pressure_map[target_id] = meaning_pressure
        
        # 説明情報を記録
        self.last_decision_explanation[target_id] = {
            'meaning_pressure': meaning_pressure,
            'primary_concept': primary_concept,
            'interpretation': pressure_interpretation.copy()
        }
        
        # 【重要】agent.step()は呼ばない
        return meaning_pressure
    
    def accumulate_pressures(self, dt: float = 1.0):
        """
        観測した全ての意味圧を集計し、内部状態を更新
        
        【新設】
        - 全他者から受けた意味圧を合算
        - agent.step()を1回だけ呼ぶ
        """
        # 人狼の内部的罪悪感
        internal_pressure = HumanPressure()
        if self.role == 'werewolf' and self.is_alive:
            internal_pressure = HumanPressure(
                core=0.05 * dt,  # 規範的葛藤
                upper=0.02 * dt   # 理念的葛藤
            )
            self._internal_guilt += 0.05 * dt
        
        # 他者から受けた意味圧を集計
        total_base = internal_pressure.base
        total_core = internal_pressure.core
        total_upper = internal_pressure.upper
        
        for target_id, pressure in self.last_pressure_map.items():
            total_base += pressure.base
            total_core += pressure.core
            total_upper += pressure.upper
        
        # 集計した意味圧で状態更新
        total_pressure = HumanPressure(
            physical=0.0,
            base=total_base,
            core=total_core,
            upper=total_upper
        )
        
        # 【重要】agent.step()は全観測後に1回だけ
        self.agent.step(total_pressure, dt=dt)
        
        # 説明用に現在のE状態を記録
        for target_id in self.last_decision_explanation.keys():
            self.last_decision_explanation[target_id]['E_state_after'] = self.agent.state.E.copy()
    
    def explain_suspicion(self, target_id: int) -> str:
        """疑惑の理由を説明（意味圧ベース）"""
        if target_id not in self.last_decision_explanation:
            return "観測データなし"
        
        info = self.last_decision_explanation[target_id]
        pressure = info['meaning_pressure']
        primary_concept = info['primary_concept']
        
        # 脅威レベル = BASE層意味圧
        threat_level = pressure.base
        
        if primary_concept:
            return f"脅威 {threat_level:.2f}: '{primary_concept}' -> 意味圧(BASE={pressure.base:.2f})"
        else:
            return f"脅威 {threat_level:.2f}: 新規 -> 意味圧(BASE={pressure.base:.2f})"
    
    def explain_internal_state(self) -> str:
        """内部状態の説明（E蓄積状態）"""
        E = self.agent.state.E
        return f"内部状態: E_BASE={E[1]:.3f}, E_CORE={E[2]:.3f}, E_UPPER={E[3]:.3f}"
    
    def generate_signals(self) -> np.ndarray:
        """
        シグナルを創発的に生成
        
        【継承】v10.2と同じ
        - 蓄積されたE（内部状態）から創発
        - 役割認識なし
        """
        signals = np.zeros(7)
        E = self.agent.state.E
        kappa = self.agent.state.kappa
        
        # 基本シグナル（PHYSICAL/BASE層）
        signals[0] = E[0] * 0.3  # 姿勢
        signals[1] = E[1] * 0.4 + E[2] * 0.2  # 表情
        signals[2] = E[0] * 0.2 + E[1] * 0.3  # 音声
        
        # 社会的シグナル（CORE/UPPER層）
        if E[2] > 0.5:  # CORE層の葛藤 -> 防御的
            signals[3] += 0.3 * E[2]  # 攻撃性
            signals[4] += 0.4 * E[2]  # 防御性
        else:
            signals[5] += 0.5 * (1.0 - E[2])  # 協調性
        
        signals[6] = E[3] * 0.4  # 理念的発言
        
        # κによるノイズ制御
        avg_kappa = np.mean(kappa)
        noise_level = 0.1 * (1.0 - avg_kappa)
        signals += np.random.randn(7) * noise_level
        
        return np.clip(signals, 0.0, 1.0)
    
    def decide_vote(self, alive_players: List[int]) -> int:
        """
        投票先を決定
        
        【修正点】
        - 蓄積されたE（内部状態）ではなく
        - 解釈された意味圧（直接的脅威）に基づく
        """
        candidates = [p for p in alive_players if p != self.player_id]
        if not candidates:
            return -1
        
        # 各候補から受けた意味圧（BASE層 = 生存脅威）
        threat_levels = {}
        for pid in candidates:
            if pid in self.last_pressure_map:
                # 【重要】意味圧そのものを使用
                threat_levels[pid] = self.last_pressure_map[pid].base
            else:
                threat_levels[pid] = 0.0
        
        # 最も脅威的な相手に投票
        if threat_levels:
            return max(threat_levels.items(), key=lambda x: x[1])[0]
        else:
            return np.random.choice(candidates)
    
    def learn_from_outcome(self, 
                          outcome: str,
                          executed_player_id: Optional[int] = None,
                          executed_player_role: Optional[str] = None,
                          current_time: float = 0.0):
        """
        ゲーム結果から学習
        
        【継承】v10.2と同じ
        """
        if outcome == 'executed':
            # 自分が処刑 -> 自分のシグナルを記憶
            my_signals = self.generate_signals()
            self.structured_memory.add_memory(
                signal=my_signals,
                layer=1,  # BASE層
                outcome=-0.8,
                timestamp=current_time
            )
            # κの上昇（トラウマ）
            self.agent.state.kappa[1] = min(1.0, self.agent.state.kappa[1] + 0.2)
            self.times_executed += 1
        
        elif outcome == 'survived':
            # 他人が処刑、正解から学習
            if executed_player_id is not None and executed_player_role is not None:
                if executed_player_id in self.observation_history and self.observation_history[executed_player_id]:
                    executed_signals = self.observation_history[executed_player_id][-1]
                    
                    if executed_player_role == 'werewolf':
                        # 正解: 人狼を処刑 -> 良い記憶
                        self.structured_memory.add_memory(
                            signal=executed_signals,
                            layer=1,  # BASE層
                            outcome=+0.9,
                            timestamp=current_time
                        )
                        self.agent.state.kappa[2] = min(1.0, self.agent.state.kappa[2] + 0.1)
                    
                    else:  # villager
                        # 誤り: 村人を処刑 -> 悪い記憶
                        self.structured_memory.add_memory(
                            signal=executed_signals,
                            layer=2,  # CORE層
                            outcome=-0.7,
                            timestamp=current_time
                        )
                        self.agent.state.kappa[2] = max(0.1, self.agent.state.kappa[2] - 0.05)
            
            self.times_survived += 1
        
        elif outcome == 'won':
            self.agent.state.kappa = np.minimum(1.0, self.agent.state.kappa + 0.1)
        
        elif outcome == 'lost':
            self.agent.state.kappa = np.minimum(1.0, self.agent.state.kappa + 0.03)
        
        self.games_played += 1
        
        # 概念抽出
        if self.games_played % 2 == 0:
            self.learned_concepts = self.structured_memory.extract_concepts()
    
    def get_learning_stats(self) -> Dict:
        """学習統計"""
        mem_stats = self.structured_memory.get_statistics()
        
        return {
            'games_played': self.games_played,
            'times_executed': self.times_executed,
            'survival_rate': self.times_survived / max(1, self.games_played),
            'n_concepts': mem_stats['n_concepts'],
            'E': self.agent.state.E.copy(),
            'kappa': self.agent.state.kappa.copy(),
            'internal_guilt': self._internal_guilt if self.role == 'werewolf' else 0.0,
            'top_concepts': mem_stats['concepts']
        }


class WerewolfGameV10Causal:
    """人狼ゲーム v10.2.1 因果律修正版"""
    
    def __init__(self, num_players: int = 7, num_werewolves: int = 2):
        self.num_players = num_players
        self.num_werewolves = num_werewolves
        
        # プレイヤー初期化
        self.players: Dict[int, PlayerV10Causal] = {}
        self._initialize_players()
        
        self.day_count = 0
        self.game_log: List[str] = []
        self.current_time = 0.0
    
    def _initialize_players(self):
        """プレイヤー初期化"""
        roles = ['werewolf'] * self.num_werewolves + \
                ['villager'] * (self.num_players - self.num_werewolves)
        np.random.shuffle(roles)
        
        for i in range(self.num_players):
            player = PlayerV10Causal(
                player_id=i,
                name=f"Player{i}",
                role=roles[i]
            )
            self.players[i] = player
    
    def get_alive_players(self) -> List[int]:
        return [p for p, player in self.players.items() if player.is_alive]
    
    def get_alive_werewolves(self) -> List[int]:
        return [p for p, player in self.players.items() 
                if player.is_alive and player.role == 'werewolf']
    
    def check_game_end(self) -> Optional[str]:
        werewolves = self.get_alive_werewolves()
        villagers = [p for p, player in self.players.items() 
                    if player.is_alive and player.role == 'villager']
        
        if len(werewolves) == 0:
            return 'villager_win'
        elif len(werewolves) >= len(villagers):
            return 'werewolf_win'
        return None
    
    def day_phase(self, verbose: bool = True):
        """
        昼フェーズ
        
        【修正された因果律】
        1. シグナル生成（前日までのE状態から）
        2. 解釈フェーズ（全員が全他者を観測、意味圧を解釈）
        3. 蓄積フェーズ（解釈された意味圧を集計してE更新）
        4. 投票（解釈された意味圧に基づく）
        """
        self.day_count += 1
        alive = self.get_alive_players()
        
        self.game_log.append(f"\n=== Day {self.day_count} ===")
        self.game_log.append(f"生存者: {len(alive)}人")
        
        # === フェーズ1: シグナル生成（前日までのE状態から） ===
        signals_map = {}
        for pid in alive:
            signals_map[pid] = self.players[pid].generate_signals()
        
        # === フェーズ2: 解釈フェーズ（意味圧の解釈のみ） ===
        # 全員の意味圧マップをクリア
        for pid in alive:
            self.players[pid].last_pressure_map.clear()
        
        # 全員が全他者を観測し、意味圧を解釈
        for observer_id in alive:
            for target_id in alive:
                if target_id != observer_id:
                    # observe_player()は意味圧を返すだけ（E更新しない）
                    self.players[observer_id].observe_player(
                        target_id, signals_map[target_id], self.current_time
                    )
        
        # === フェーズ3: 蓄積フェーズ（E状態の一括更新） ===
        for pid in alive:
            # 全観測結果を集計してagent.step()を1回だけ実行
            self.players[pid].accumulate_pressures(dt=1.0)
        
        # === フェーズ4: 投票（解釈された意味圧に基づく） ===
        votes: Dict[int, int] = {}
        vote_details: List[str] = []
        
        for voter_id in alive:
            target = self.players[voter_id].decide_vote(alive)
            if target != -1:
                votes[target] = votes.get(target, 0) + 1
                explanation = self.players[voter_id].explain_suspicion(target)
                internal_state = self.players[voter_id].explain_internal_state()
                vote_details.append(
                    f"  {self.players[voter_id].name} -> {self.players[target].name}:\n"
                    f"    {explanation}\n"
                    f"    {internal_state}"
                )
        
        # 処刑
        if votes:
            executed_id = max(votes.items(), key=lambda x: x[1])[0]
            executed = self.players[executed_id]
            executed_role = executed.role
            executed.is_alive = False
            
            self.game_log.append(f"投票結果: {executed.name} ({executed_role}) 処刑")
            
            if verbose:
                self.game_log.append("\n投票詳細:")
                for detail in vote_details[:5]:
                    self.game_log.append(detail)
            
            # 学習（正解から）
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
    
    def play_game(self, max_days: int = 10, verbose: bool = True) -> str:
        """ゲームをプレイ"""
        self.game_log.clear()
        self.game_log.append("=" * 70)
        self.game_log.append("人狼ゲーム v10.2.1 (因果律修正版)")
        self.game_log.append("=" * 70)
        
        for day in range(max_days):
            self.day_phase(verbose=verbose)
            
            result = self.check_game_end()
            if result:
                self.game_log.append(f"\n{'='*70}")
                self.game_log.append(f"ゲーム終了: {result}")
                self.game_log.append(f"{'='*70}")
                
                for player in self.players.values():
                    outcome = 'won' if (
                        (player.role == 'werewolf' and result == 'werewolf_win') or
                        (player.role == 'villager' and result == 'villager_win')
                    ) else 'lost'
                    player.learn_from_outcome(outcome, current_time=self.current_time)
                
                if verbose:
                    for log in self.game_log:
                        print(log)
                
                return result
        
        if verbose:
            for log in self.game_log:
                print(log)
        return 'villager_win'


def demo_causal():
    """因果律修正版デモ"""
    print("="*70)
    print("人狼ゲーム v10.2.1 - 因果律修正版")
    print("="*70)
    print("\n【修正された因果律】")
    print("1. シグナル生成（前日までのE状態から）")
    print("2. 解釈フェーズ（全員が全他者を観測、意味圧を解釈）")
    print("3. 蓄積フェーズ（解釈された意味圧を集計してE更新）")
    print("4. 投票（解釈された意味圧に基づく）")
    print()
    print("【理論的整合性】")
    print("- 意味圧（HumanPressure）: 他者からの直接的脅威 -> 投票の引き金")
    print("- E（未処理圧）: 蓄積されたストレス -> シグナル生成の源泉")
    print()
    
    game = WerewolfGameV10Causal(num_players=7, num_werewolves=2)
    result = game.play_game(verbose=True)
    
    print("\n" + "="*70)
    print("学習統計（因果律修正版）")
    print("="*70)
    
    for pid, player in game.players.items():
        stats = player.get_learning_stats()
        print(f"\n{player.name} ({player.role}):")
        print(f"  E: {stats['E']}")
        print(f"  κ: {stats['kappa']}")
        print(f"  概念: {stats['n_concepts']}個")
        
        if player.role == 'werewolf':
            print(f"  罪悪感: {stats['internal_guilt']:.3f}")
        
        if stats['top_concepts']:
            print(f"  主要概念: {stats['top_concepts'][0][0]}")


if __name__ == "__main__":
    demo_causal()
    
    print("\n\n" + "="*70)
    print("Phase 10.2.1 完了 - SSD因果律との完全整合")
    print("="*70)
    print("\n- 「解釈」と「蓄積」の完全分離")
    print("- 投票は意味圧（直接的脅威）に基づく")
    print("- Eは蓄積されたストレス（シグナル生成の源泉）")
    print("- 観測順序による汚染を完全排除")
