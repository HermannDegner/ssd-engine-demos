"""
人狼ゲーム 拡張役職版
Werewolf Game with Extended Roles

【Phase 10.4c: ゲーム性強化】
理論的コア（v10.2.1）に役職を追加:
1. 占い師（Fortune Teller）: 毎夜1人の役割を知る
2. 霊媒師（Medium）: 処刑された人の役割を知る
3. 狂人（Madman）: 人狼陣営だが人狼を知らない
4. 騎士（Knight）: 毎夜1人を護衛
5. 拡張版ゲームロジック

作成日: 2025年11月7日
バージョン: 10.4c
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional
from enum import Enum

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

from core.ssd_human_module import HumanPressure


class Role(Enum):
    """役職の列挙型"""
    VILLAGER = "villager"          # 村人
    WEREWOLF = "werewolf"          # 人狼
    FORTUNE_TELLER = "fortune_teller"  # 占い師
    MEDIUM = "medium"              # 霊媒師
    MADMAN = "madman"              # 狂人
    KNIGHT = "knight"              # 騎士


class ExtendedPlayer(PlayerV10Causal):
    """
    拡張役職プレイヤー
    
    理論的コア（v10.2.1）を継承し、役職能力を追加
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 役職固有の情報
        self.divine_results: Dict[int, str] = {}  # 占い結果 {player_id: role}
        self.medium_results: Dict[int, str] = {}  # 霊媒結果 {player_id: role}
        self.protected_history: List[int] = []    # 護衛履歴
        
        # 公開情報（他プレイヤーが知っている）
        self.claimed_role: Optional[str] = None   # 自称役職
        self.claimed_divine_results: Dict[int, str] = {}  # 公開した占い結果
    
    def use_ability(self, target_id: int, game_state: 'ExtendedGame') -> Optional[str]:
        """
        役職能力を使用
        
        Args:
            target_id: 対象プレイヤー
            game_state: ゲーム状態
        
        Returns:
            能力使用の結果メッセージ
        """
        if self.role == Role.FORTUNE_TELLER.value:
            return self._divine(target_id, game_state)
        
        elif self.role == Role.MEDIUM.value:
            return self._check_medium(game_state)
        
        elif self.role == Role.KNIGHT.value:
            return self._protect(target_id, game_state)
        
        return None
    
    def _divine(self, target_id: int, game_state: 'ExtendedGame') -> str:
        """占い"""
        target = game_state.players[target_id]
        
        # 人狼か人間か（狂人は人間判定）
        if target.role == Role.WEREWOLF.value:
            result = "werewolf"
        else:
            result = "human"
        
        self.divine_results[target_id] = result
        
        # SSD理論的解釈: 情報取得による意味圧の変化
        if result == "werewolf":
            # 人狼を発見 -> CORE層に強い圧力（規範的使命感）
            info_pressure = HumanPressure(
                base=0.3,   # 脅威認識
                core=0.4,   # 「人狼を告発すべき」という規範的圧力
                upper=0.2   # 「村を守る」という理念的圧力
            )
        else:
            # 人間を確認 -> 安心感（負の圧力）
            info_pressure = HumanPressure(
                base=-0.2,  # 安心
                core=-0.1
            )
        
        self.agent.step(info_pressure, dt=0.1)
        
        return f"{target.name}を占いました: {result}"
    
    def _check_medium(self, game_state: 'ExtendedGame') -> str:
        """霊媒（前日の処刑者を調査）"""
        if not game_state.execution_history:
            return "まだ処刑者がいません"
        
        last_executed_id = game_state.execution_history[-1]
        last_executed = game_state.players[last_executed_id]
        
        # 人狼か人間か
        if last_executed.role == Role.WEREWOLF.value:
            result = "werewolf"
        else:
            result = "human"
        
        self.medium_results[last_executed_id] = result
        
        # SSD理論的解釈: 判断の正否による内部状態変化
        if result == "werewolf":
            # 正解 -> κ上昇（学習の強化）
            self.agent.state.kappa[2] = min(1.0, self.agent.state.kappa[2] + 0.15)
            validation_pressure = HumanPressure(core=-0.2)  # 安堵
        else:
            # 誤判断 -> κ低下、罪悪感
            self.agent.state.kappa[2] = max(0.1, self.agent.state.kappa[2] - 0.1)
            validation_pressure = HumanPressure(
                core=0.3,   # 罪悪感
                upper=0.2   # 理念的苦悩
            )
        
        self.agent.step(validation_pressure, dt=0.1)
        
        return f"{last_executed.name}を霊媒しました: {result}"
    
    def _protect(self, target_id: int, game_state: 'ExtendedGame') -> str:
        """護衛"""
        target = game_state.players[target_id]
        self.protected_history.append(target_id)
        
        # SSD理論的解釈: 保護行動による使命感
        protection_pressure = HumanPressure(
            core=0.1,   # 責任感
            upper=0.15  # 「村を守る」という使命
        )
        self.agent.step(protection_pressure, dt=0.1)
        
        return f"{target.name}を護衛しました"
    
    def decide_claim(self) -> Optional[str]:
        """
        役職を公開するか決定（SSD理論ベース）
        
        Returns:
            公開する役職（Noneなら公開しない）
        """
        # 既に公開済み
        if self.claimed_role is not None:
            return self.claimed_role
        
        E = self.agent.state.E
        kappa = self.agent.state.kappa
        
        # 占い師: 人狼を見つけたら高確率で公開
        if self.role == Role.FORTUNE_TELLER.value:
            werewolf_found = any(r == "werewolf" for r in self.divine_results.values())
            if werewolf_found:
                # E_CORE（規範的圧力）とE_UPPER（理念的圧力）が高い -> 告発の義務感
                if E[2] > 0.3 or E[3] > 0.2:
                    self.claimed_role = Role.FORTUNE_TELLER.value
                    return self.claimed_role
        
        # 霊媒師: 人狼処刑を確認したら公開
        elif self.role == Role.MEDIUM.value:
            werewolf_executed = any(r == "werewolf" for r in self.medium_results.values())
            if werewolf_executed and E[2] > 0.2:
                self.claimed_role = Role.MEDIUM.value
                return self.claimed_role
        
        # 狂人: ランダムに占い師を騙る（人狼を守るため）
        elif self.role == Role.MADMAN.value:
            if np.random.rand() < 0.3 and self.games_played > 1:
                self.claimed_role = Role.FORTUNE_TELLER.value  # 偽占い師
                return self.claimed_role
        
        return None
    
    def decide_vote_with_info(self, alive_players: List[int]) -> int:
        """
        役職情報を加味した投票
        
        Args:
            alive_players: 生存者リスト
        
        Returns:
            投票先
        """
        candidates = [p for p in alive_players if p != self.player_id]
        if not candidates:
            return -1
        
        # 基本の意味圧ベース疑惑
        base_suspicions = {}
        for pid in candidates:
            if pid in self.last_pressure_map:
                base_suspicions[pid] = self.last_pressure_map[pid].base
            else:
                base_suspicions[pid] = 0.0
        
        # 役職情報による修正
        if self.role == Role.FORTUNE_TELLER.value:
            # 占いで人狼判定した相手を最優先
            for pid, result in self.divine_results.items():
                if pid in candidates and result == "werewolf":
                    base_suspicions[pid] += 2.0  # 大幅に疑惑上昇
        
        elif self.role == Role.MEDIUM.value:
            # 霊媒結果を信頼度に反映（間接的）
            # 人狼を正しく処刑できた -> 全体の判断力に自信
            werewolf_count = sum(1 for r in self.medium_results.values() if r == "werewolf")
            if werewolf_count > 0:
                # 自分の判断に自信 -> 意味圧をそのまま使う
                pass
        
        elif self.role == Role.MADMAN.value:
            # 狂人: 村人っぽい人を攻撃（人狼を守る）
            # 意味圧が低い（信頼できそう）相手を優先的に疑う
            for pid in candidates:
                if base_suspicions[pid] < 0:  # 信頼されてる
                    base_suspicions[pid] += 0.5  # 疑惑を捏造
        
        # 最大疑惑の相手に投票
        if base_suspicions:
            return max(base_suspicions.items(), key=lambda x: x[1])[0]
        else:
            return np.random.choice(candidates)


class ExtendedGame(WerewolfGameV10Causal):
    """
    拡張役職版人狼ゲーム
    
    理論的コア（v10.2.1）を継承し、役職システムを追加
    """
    
    def __init__(self, 
                 num_players: int = 9,
                 num_werewolves: int = 2,
                 has_fortune_teller: bool = True,
                 has_medium: bool = True,
                 has_madman: bool = True,
                 has_knight: bool = False):
        
        self.num_players = num_players
        self.num_werewolves = num_werewolves
        self.has_fortune_teller = has_fortune_teller
        self.has_medium = has_medium
        self.has_madman = has_madman
        self.has_knight = has_knight
        
        # プレイヤー初期化
        self.players: Dict[int, ExtendedPlayer] = {}
        self._initialize_extended_players()
        
        self.day_count = 0
        self.game_log: List[str] = []
        self.current_time = 0.0
        self.execution_history: List[int] = []  # 処刑履歴
        self.night_death: Optional[int] = None   # 夜の襲撃対象
    
    def _initialize_extended_players(self):
        """役職を含むプレイヤー初期化"""
        roles = []
        
        # 人狼
        roles.extend([Role.WEREWOLF.value] * self.num_werewolves)
        
        # 特殊役職
        if self.has_fortune_teller:
            roles.append(Role.FORTUNE_TELLER.value)
        if self.has_medium:
            roles.append(Role.MEDIUM.value)
        if self.has_madman:
            roles.append(Role.MADMAN.value)
        if self.has_knight:
            roles.append(Role.KNIGHT.value)
        
        # 残りは村人
        remaining = self.num_players - len(roles)
        roles.extend([Role.VILLAGER.value] * remaining)
        
        np.random.shuffle(roles)
        
        for i in range(self.num_players):
            player = ExtendedPlayer(
                player_id=i,
                name=f"Player{i}",
                role=roles[i]
            )
            self.players[i] = player
    
    def night_phase(self):
        """夜フェーズ"""
        alive = self.get_alive_players()
        
        # 人狼の襲撃
        werewolves = [pid for pid in alive if self.players[pid].role == Role.WEREWOLF.value]
        if werewolves:
            # 最も疑われていない村人を襲撃
            villagers = [pid for pid in alive 
                        if pid not in werewolves and self.players[pid].role != Role.MADMAN.value]
            if villagers:
                # ランダムに選択（簡易版）
                target = np.random.choice(villagers)
                self.night_death = target
        
        # 占い師の占い
        fortune_tellers = [pid for pid in alive 
                          if self.players[pid].role == Role.FORTUNE_TELLER.value]
        for ft_id in fortune_tellers:
            candidates = [pid for pid in alive if pid != ft_id]
            if candidates:
                # まだ占ってない人を優先
                undivinied = [c for c in candidates 
                            if c not in self.players[ft_id].divine_results]
                if undivinied:
                    target = np.random.choice(undivinied)
                else:
                    target = np.random.choice(candidates)
                
                result = self.players[ft_id].use_ability(target, self)
                self.game_log.append(f"  {self.players[ft_id].name}（占い師）: {result}")
        
        # 霊媒師の霊媒
        mediums = [pid for pid in alive if self.players[pid].role == Role.MEDIUM.value]
        for med_id in mediums:
            result = self.players[med_id].use_ability(-1, self)
            if "まだ" not in result:
                self.game_log.append(f"  {self.players[med_id].name}（霊媒師）: {result}")
        
        # 騎士の護衛
        knights = [pid for pid in alive if self.players[pid].role == Role.KNIGHT.value]
        for knight_id in knights:
            candidates = [pid for pid in alive if pid != knight_id]
            if candidates:
                # ランダムに護衛
                target = np.random.choice(candidates)
                result = self.players[knight_id].use_ability(target, self)
                self.game_log.append(f"  {self.players[knight_id].name}（騎士）: {result}")
                
                # 襲撃から守った場合
                if target == self.night_death:
                    self.game_log.append(f"  -> {self.players[target].name}は護衛されました！")
                    self.night_death = None
    
    def day_phase(self, verbose: bool = True):
        """昼フェーズ（拡張版）"""
        self.day_count += 1
        alive = self.get_alive_players()
        
        self.game_log.append(f"\n=== Day {self.day_count} ===")
        
        # 夜の死亡発表
        if self.night_death is not None:
            victim = self.players[self.night_death]
            victim.is_alive = False
            self.game_log.append(f"{victim.name}が人狼に襲撃されました（{victim.role}）")
            self.night_death = None
            alive = self.get_alive_players()
        
        self.game_log.append(f"生存者: {len(alive)}人")
        
        # 役職公開の判断
        for pid in alive:
            claimed = self.players[pid].decide_claim()
            if claimed and self.players[pid].claimed_role is None:
                self.game_log.append(f"{self.players[pid].name}が{claimed}を名乗り出ました")
        
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
        
        # 投票（役職情報を加味）
        votes: Dict[int, int] = {}
        vote_details: List[str] = []
        
        for voter_id in alive:
            target = self.players[voter_id].decide_vote_with_info(alive)
            if target != -1:
                votes[target] = votes.get(target, 0) + 1
                explanation = self.players[voter_id].explain_suspicion(target)
                vote_details.append(
                    f"  {self.players[voter_id].name} -> {self.players[target].name}: {explanation}"
                )
        
        # 処刑
        if votes:
            executed_id = max(votes.items(), key=lambda x: x[1])[0]
            executed = self.players[executed_id]
            executed_role = executed.role
            executed.is_alive = False
            
            self.execution_history.append(executed_id)
            
            self.game_log.append(f"投票結果: {executed.name} ({executed_role}) 処刑")
            
            if verbose:
                self.game_log.append("\n投票詳細:")
                for detail in vote_details[:5]:
                    self.game_log.append(detail)
            
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
    
    def play_extended_game(self, max_days: int = 10, verbose: bool = True) -> str:
        """拡張版ゲームをプレイ"""
        self.game_log.clear()
        self.game_log.append("=" * 70)
        self.game_log.append("人狼ゲーム v10.4c (拡張役職版)")
        self.game_log.append("=" * 70)
        
        # 役職構成を表示
        role_counts = {}
        for player in self.players.values():
            role_counts[player.role] = role_counts.get(player.role, 0) + 1
        
        self.game_log.append("\n【役職構成】")
        for role, count in role_counts.items():
            self.game_log.append(f"  {role}: {count}人")
        
        for day in range(max_days):
            # 夜フェーズ
            if day > 0:
                self.game_log.append(f"\n=== Night {day} ===")
                self.night_phase()
            
            # 昼フェーズ
            self.day_phase(verbose=verbose)
            
            result = self.check_game_end()
            if result:
                self.game_log.append(f"\n{'='*70}")
                self.game_log.append(f"ゲーム終了: {result}")
                self.game_log.append(f"{'='*70}")
                
                for player in self.players.values():
                    outcome = 'won' if (
                        (player.role in [Role.WEREWOLF.value, Role.MADMAN.value] and result == 'werewolf_win') or
                        (player.role not in [Role.WEREWOLF.value, Role.MADMAN.value] and result == 'villager_win')
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


def demo_extended():
    """拡張役職版デモ"""
    print("="*70)
    print("人狼ゲーム 拡張役職版デモ")
    print("="*70)
    print("\n【追加役職】")
    print("- 占い師: 毎夜1人の正体を知る")
    print("- 霊媒師: 処刑された人の正体を知る")
    print("- 狂人: 人狼陣営だが人狼を知らない")
    print("- （騎士: 毎夜1人を護衛）※今回は無効")
    print()
    print("【SSD理論との統合】")
    print("- 占い結果による意味圧の変化（人狼発見 -> CORE層圧力上昇）")
    print("- 霊媒結果による学習（κの調整）")
    print("- 役職公開の判断（E状態ベース）")
    print()
    
    # 学習のため数ゲームをプレイ
    print("学習のため3ゲームをプレイ中...\n")
    for game_num in range(3):
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
    
    # 最終ゲーム（詳細表示）
    print("="*70)
    print("拡張版ゲーム開始（詳細表示）")
    print("="*70)
    print()
    
    final_game = ExtendedGame(
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
    
    result = final_game.play_extended_game(max_days=10, verbose=True)
    
    print("\n" + "="*70)
    print("最終統計")
    print("="*70)
    
    for pid, player in final_game.players.items():
        stats = player.get_learning_stats()
        print(f"\n{player.name} ({player.role}):")
        print(f"  生存: {'はい' if player.is_alive else 'いいえ'}")
        print(f"  概念: {stats['n_concepts']}個")
        
        if player.role == Role.FORTUNE_TELLER.value:
            print(f"  占い回数: {len(player.divine_results)}")
            werewolf_found = sum(1 for r in player.divine_results.values() if r == "werewolf")
            print(f"  人狼発見: {werewolf_found}人")
        
        elif player.role == Role.MEDIUM.value:
            print(f"  霊媒回数: {len(player.medium_results)}")
        
        elif player.role == Role.WEREWOLF.value:
            print(f"  罪悪感: {stats['internal_guilt']:.3f}")
    
    print("\n" + "="*70)
    print("Phase 10.4c 完了: ゲーム性強化")
    print("="*70)
    print("\n理論的コア（v10.2.1）は不変、役職システムのみ追加")
    print("SSD理論と役職能力の統合を実証")


if __name__ == "__main__":
    demo_extended()
