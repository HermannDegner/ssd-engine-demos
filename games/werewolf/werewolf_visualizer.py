"""
人狼ゲーム 可視化モジュール
Werewolf Game Visualizer

【Phase 10.4: 可視化】
理論的コア（v10.2.1）の動作を視覚化:
1. 概念ネットワーク（類似度ベースのグラフ）
2. 意味圧の時系列変化
3. E（未処理圧）の蓄積プロセス
4. シグナルのヒートマップ
5. 投票パターンの分析

作成日: 2025年11月7日
バージョン: 10.4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional
import sys
import os

# 理論的コアをインポート
# 理論的コア (archiveフォルダーから)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
archive_dir = os.path.join(parent_dir, 'archive')
sys.path.insert(0, archive_dir)

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

from ssd_memory_structure import Concept, cosine_similarity


class WerewolfVisualizer:
    """
    人狼ゲームの可視化クラス
    
    【可視化項目】
    - 概念ネットワーク: 学習した概念の関係性
    - 意味圧フロー: 観測 -> 意味圧 -> E蓄積
    - 投票パターン: 誰が誰を疑ったか
    - E状態遷移: 各層の未処理圧の時系列
    """
    
    def __init__(self, game: WerewolfGameV10Causal):
        self.game = game
        self.pressure_history: Dict[int, List[Dict]] = {
            pid: [] for pid in range(game.num_players)
        }
        self.E_history: Dict[int, List[np.ndarray]] = {
            pid: [] for pid in range(game.num_players)
        }
        self.vote_history: List[Dict] = []
        
        # プロット設定
        plt.rcParams['font.family'] = 'Yu Gothic'  # 日本語フォント
        plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
    
    def record_day(self, day: int):
        """1日分のデータを記録"""
        # 各プレイヤーのE状態を記録
        for pid, player in self.game.players.items():
            if player.is_alive:
                self.E_history[pid].append(player.agent.state.E.copy())
                
                # 意味圧マップを記録
                pressure_snapshot = {}
                for target_id, pressure in player.last_pressure_map.items():
                    pressure_snapshot[target_id] = {
                        'base': pressure.base,
                        'core': pressure.core,
                        'upper': pressure.upper
                    }
                self.pressure_history[pid].append(pressure_snapshot)
    
    def plot_concept_network(self, player_id: int, save_path: Optional[str] = None):
        """
        概念ネットワークを可視化
        
        類似度が高い概念同士をエッジで結ぶ
        """
        player = self.game.players[player_id]
        concepts = player.learned_concepts
        
        if not concepts:
            print(f"Player{player_id}はまだ概念を形成していません")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 概念の座標（円形配置）
        n = len(concepts)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        positions = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
        
        # エッジ（類似度が0.6以上の概念ペア）
        for i in range(n):
            for j in range(i+1, n):
                sim = cosine_similarity(concepts[i].cluster.prototype_signal, concepts[j].cluster.prototype_signal)
                if sim > 0.6:
                    x_coords = [positions[i][0], positions[j][0]]
                    y_coords = [positions[i][1], positions[j][1]]
                    ax.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=sim*3)
        
        # ノード（概念）
        for i, concept in enumerate(concepts):
            x, y = positions[i]
            
            # 色: 重要度によって色分け
            importance = concept.importance if hasattr(concept, 'importance') else 0.5
            color = plt.cm.RdYlBu_r(importance)
            
            # サイズ: メンバー数
            n_members = concept.cluster.n_memories if hasattr(concept, 'cluster') else 5
            prototype = concept.cluster.prototype_signal if hasattr(concept, 'cluster') else np.zeros(7)
            size = 100 + n_members * 20
            
            ax.scatter(x, y, s=size, c=[color], alpha=0.6, edgecolors='black', linewidth=2)
            
            # ラベル
            ax.text(x, y+0.15, concept.name, ha='center', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 凡例
        legend_elements = [
            mpatches.Patch(color='darkblue', label='重要度: 高'),
            mpatches.Patch(color='yellow', label='重要度: 中'),
            mpatches.Patch(color='darkred', label='重要度: 低')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{player.name}の概念ネットワーク ({player.role})\n'
                    f'概念数: {len(concepts)}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pressure_flow(self, observer_id: int, target_id: int, day: int,
                          save_path: Optional[str] = None):
        """
        意味圧フローを可視化
        
        観測 -> 構造化記憶 -> 意味圧 -> HumanAgent -> E蓄積
        """
        observer = self.game.players[observer_id]
        target = self.game.players[target_id]
        
        if day >= len(self.pressure_history[observer_id]):
            print(f"Day {day}のデータがありません")
            return
        
        pressure_data = self.pressure_history[observer_id][day]
        if target_id not in pressure_data:
            print(f"Player{observer_id}はDay{day}にPlayer{target_id}を観測していません")
            return
        
        pressure = pressure_data[target_id]
        E_state = self.E_history[observer_id][day] if day < len(self.E_history[observer_id]) else np.zeros(4)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # パイプライン
        stages = ['観測\nシグナル', '構造化\n記憶', '意味圧\n(HumanPressure)', 
                 'HumanAgent\n.step()', 'E蓄積\n(未処理圧)']
        x_positions = np.arange(len(stages))
        
        # ステージボックス
        for i, stage in enumerate(stages):
            ax.add_patch(plt.Rectangle((i-0.3, 0.3), 0.6, 0.4, 
                                      facecolor='lightblue', edgecolor='black', linewidth=2))
            ax.text(i, 0.5, stage, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 矢印
        for i in range(len(stages)-1):
            ax.annotate('', xy=(i+0.35, 0.5), xytext=(i+0.65, 0.5),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # データ表示
        # 1. 観測シグナル
        if target_id in observer.observation_history and observer.observation_history[target_id]:
            signals = observer.observation_history[target_id][min(day, len(observer.observation_history[target_id])-1)]
            ax.text(0, 0.1, f'7次元\n平均: {np.mean(signals):.2f}', 
                   ha='center', fontsize=8, color='darkgreen')
        
        # 2. 構造化記憶
        activated = [c.name for c in observer.learned_concepts 
                    if target_id in observer.observation_history and 
                    observer.observation_history[target_id] and
                    c.matches(observer.observation_history[target_id][-1])]
        ax.text(1, 0.1, f'概念:\n{activated[0] if activated else "新規"}', 
               ha='center', fontsize=8, color='darkblue')
        
        # 3. 意味圧
        ax.text(2, 0.1, f'BASE: {pressure["base"]:.2f}\n'
                        f'CORE: {pressure["core"]:.2f}\n'
                        f'UPPER: {pressure["upper"]:.2f}', 
               ha='center', fontsize=8, color='darkred')
        
        # 4. HumanAgent.step()
        ax.text(3, 0.1, 'dt=0.1\nE更新', ha='center', fontsize=8, color='purple')
        
        # 5. E蓄積
        ax.text(4, 0.1, f'E_BASE: {E_state[1]:.3f}\n'
                        f'E_CORE: {E_state[2]:.3f}\n'
                        f'E_UPPER: {E_state[3]:.3f}', 
               ha='center', fontsize=8, color='darkgreen')
        
        ax.set_xlim(-0.5, len(stages)-0.5)
        ax.set_ylim(0, 0.8)
        ax.axis('off')
        ax.set_title(f'Day {day}: {observer.name}が{target.name}を観測\n'
                    f'意味圧フロー（SSD理論パイプライン）', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_E_evolution(self, player_id: int, save_path: Optional[str] = None):
        """
        E（未処理圧）の時系列変化
        """
        player = self.game.players[player_id]
        E_data = self.E_history[player_id]
        
        if not E_data:
            print(f"Player{player_id}のE履歴がありません")
            return
        
        E_array = np.array(E_data)  # (days, 4)
        days = np.arange(len(E_data))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        layer_names = ['PHYSICAL', 'BASE', 'CORE', 'UPPER']
        colors = ['red', 'orange', 'blue', 'purple']
        
        for layer_idx in range(4):
            ax.plot(days, E_array[:, layer_idx], marker='o', 
                   label=f'{layer_names[layer_idx]}層', color=colors[layer_idx], linewidth=2)
        
        # 人狼の罪悪感を強調
        if player.role == 'werewolf':
            ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.3, label='罪悪感ベースライン')
        
        ax.set_xlabel('日数', fontsize=12)
        ax.set_ylabel('E (未処理圧)', fontsize=12)
        ax.set_title(f'{player.name}のE状態遷移 ({player.role})\n'
                    f'蓄積されたストレスの時系列', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_signal_heatmap(self, day: int, save_path: Optional[str] = None):
        """
        全プレイヤーのシグナルをヒートマップで可視化
        """
        alive = self.game.get_alive_players()
        signal_names = ['姿勢', '表情', '音声', '攻撃性', '防御性', '協調性', '理念']
        
        # シグナル収集
        signals_matrix = []
        player_labels = []
        for pid in alive:
            player = self.game.players[pid]
            if pid in player.observation_history and player.observation_history:
                # 最新のシグナルを使用
                signals = player.generate_signals()
                signals_matrix.append(signals)
                player_labels.append(f'{player.name}\n({player.role})')
        
        if not signals_matrix:
            print("シグナルデータがありません")
            return
        
        signals_matrix = np.array(signals_matrix)  # (players, 7)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(signals_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 軸設定
        ax.set_xticks(np.arange(7))
        ax.set_yticks(np.arange(len(player_labels)))
        ax.set_xticklabels(signal_names)
        ax.set_yticklabels(player_labels)
        
        # 値を表示
        for i in range(len(player_labels)):
            for j in range(7):
                text = ax.text(j, i, f'{signals_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black', fontsize=8)
        
        ax.set_title(f'Day {day}: 全プレイヤーのシグナル分布\n'
                    f'（創発的生成 - E状態から）', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, label='シグナル強度')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_vote_network(self, day: int, save_path: Optional[str] = None):
        """
        投票ネットワークを可視化
        
        誰が誰に投票したか（意味圧の大きさで線の太さを変える）
        """
        alive = self.game.get_alive_players()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # プレイヤーの位置（円形配置）
        n = len(alive)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        positions = {pid: (np.cos(angles[i])*2, np.sin(angles[i])*2) 
                    for i, pid in enumerate(alive)}
        
        # 投票エッジ（意味圧ベース）
        for voter_id in alive:
            voter = self.game.players[voter_id]
            target = voter.decide_vote(alive)
            
            if target != -1 and target in positions:
                x_coords = [positions[voter_id][0], positions[target][0]]
                y_coords = [positions[voter_id][1], positions[target][1]]
                
                # 意味圧の大きさで太さを決定
                if target in voter.last_pressure_map:
                    threat = voter.last_pressure_map[target].base
                    linewidth = max(0.5, abs(threat) * 5)
                    color = 'red' if threat > 0 else 'blue'
                else:
                    linewidth = 0.5
                    color = 'gray'
                
                ax.annotate('', xy=(x_coords[1], y_coords[1]), 
                           xytext=(x_coords[0], y_coords[0]),
                           arrowprops=dict(arrowstyle='->', lw=linewidth, color=color, alpha=0.6))
        
        # ノード（プレイヤー）
        for pid in alive:
            player = self.game.players[pid]
            x, y = positions[pid]
            
            color = 'darkred' if player.role == 'werewolf' else 'lightblue'
            ax.scatter(x, y, s=500, c=color, edgecolors='black', linewidth=3, zorder=10)
            ax.text(x, y, player.name, ha='center', va='center', 
                   fontsize=10, fontweight='bold', zorder=11)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Day {day}: 投票ネットワーク\n'
                    f'（矢印の太さ = 意味圧の大きさ）', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comprehensive_dashboard(self, day: int, save_path: Optional[str] = None):
        """
        総合ダッシュボード
        
        1ページに全ての可視化を配置
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. シグナルヒートマップ
        ax1 = fig.add_subplot(gs[0, :2])
        alive = self.game.get_alive_players()
        signal_names = ['姿勢', '表情', '音声', '攻撃性', '防御性', '協調性', '理念']
        signals_matrix = []
        player_labels = []
        for pid in alive:
            player = self.game.players[pid]
            signals = player.generate_signals()
            signals_matrix.append(signals)
            player_labels.append(f'{player.name}')
        
        if signals_matrix:
            signals_matrix = np.array(signals_matrix)
            im1 = ax1.imshow(signals_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(np.arange(7))
            ax1.set_yticks(np.arange(len(player_labels)))
            ax1.set_xticklabels(signal_names, fontsize=8)
            ax1.set_yticklabels(player_labels, fontsize=8)
            ax1.set_title('シグナル分布（創発的生成）', fontsize=10, fontweight='bold')
            plt.colorbar(im1, ax=ax1)
        
        # 2. E状態（人狼のみ）
        ax2 = fig.add_subplot(gs[0, 2])
        werewolves = [pid for pid in alive if self.game.players[pid].role == 'werewolf']
        if werewolves and self.E_history[werewolves[0]]:
            E_data = np.array(self.E_history[werewolves[0]])
            days = np.arange(len(E_data))
            ax2.plot(days, E_data[:, 2], 'b-', label='E_CORE', linewidth=2)
            ax2.plot(days, E_data[:, 3], 'p-', label='E_UPPER', linewidth=2)
            ax2.set_xlabel('Day', fontsize=8)
            ax2.set_ylabel('E', fontsize=8)
            ax2.set_title(f'{self.game.players[werewolves[0]].name}の罪悪感', fontsize=10, fontweight='bold')
            ax2.legend(fontsize=7)
            ax2.grid(True, alpha=0.3)
        
        # 3. 投票ネットワーク
        ax3 = fig.add_subplot(gs[1:, :])
        n = len(alive)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        positions = {pid: (np.cos(angles[i])*2, np.sin(angles[i])*2) 
                    for i, pid in enumerate(alive)}
        
        for voter_id in alive:
            voter = self.game.players[voter_id]
            target = voter.decide_vote(alive)
            
            if target != -1 and target in positions:
                x_coords = [positions[voter_id][0], positions[target][0]]
                y_coords = [positions[voter_id][1], positions[target][1]]
                
                if target in voter.last_pressure_map:
                    threat = voter.last_pressure_map[target].base
                    linewidth = max(0.5, abs(threat) * 5)
                    color = 'red' if threat > 0 else 'blue'
                else:
                    linewidth = 0.5
                    color = 'gray'
                
                ax3.annotate('', xy=(x_coords[1], y_coords[1]), 
                           xytext=(x_coords[0], y_coords[0]),
                           arrowprops=dict(arrowstyle='->', lw=linewidth, color=color, alpha=0.6))
        
        for pid in alive:
            player = self.game.players[pid]
            x, y = positions[pid]
            color = 'darkred' if player.role == 'werewolf' else 'lightblue'
            ax3.scatter(x, y, s=500, c=color, edgecolors='black', linewidth=3, zorder=10)
            ax3.text(x, y, player.name, ha='center', va='center', 
                   fontsize=10, fontweight='bold', zorder=11)
        
        ax3.set_xlim(-3, 3)
        ax3.set_ylim(-3, 3)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title('投票ネットワーク（矢印の太さ = 意味圧）', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'Day {day}: 総合ダッシュボード (SSD v10.2.1)', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def demo_visualizer():
    """可視化デモ"""
    print("="*70)
    print("人狼ゲーム 可視化デモ (Phase 10.4)")
    print("="*70)
    print("\n理論的コア（v10.2.1）+ 可視化フレーバー")
    print()
    
    # ゲーム実行（複数回プレイして学習を促進）
    game = WerewolfGameV10Causal(num_players=7, num_werewolves=2)
    
    print("学習のため3ゲームをプレイ中...\n")
    for _ in range(3):
        game = WerewolfGameV10Causal(num_players=7, num_werewolves=2)
        # プレイヤーを引き継ぎ（学習継続）
        if _ > 0:
            for pid in range(7):
                if pid < len(list(game.players.values())):
                    game.players[pid].structured_memory = prev_players[pid].structured_memory
                    game.players[pid].learned_concepts = prev_players[pid].learned_concepts
                    game.players[pid].agent.state.kappa = prev_players[pid].agent.state.kappa
        
        prev_players = {pid: p for pid, p in game.players.items()}
        game.play_game(max_days=5, verbose=False)
    
    # 最終ゲームを可視化用に実行
    game = WerewolfGameV10Causal(num_players=7, num_werewolves=2)
    # 学習結果を引き継ぎ
    for pid in range(7):
        game.players[pid].structured_memory = prev_players[pid].structured_memory
        game.players[pid].learned_concepts = prev_players[pid].learned_concepts
        game.players[pid].agent.state.kappa = prev_players[pid].agent.state.kappa
    
    visualizer = WerewolfVisualizer(game)
    
    print("可視化用ゲームを実行中...\n")
    
    # 各日のデータを記録しながら実行
    for day in range(1, 6):
        game.day_phase(verbose=False)
        visualizer.record_day(day)
        
        result = game.check_game_end()
        if result:
            break
    
    result = game.check_game_end() or 'villager_win'
    print(f"\nゲーム終了: {result}\n")
    
    # 可視化
    print("="*70)
    print("可視化を生成中...")
    print("="*70)
    
    # 1. E状態の時系列（人狼）
    werewolves = [pid for pid, p in game.players.items() if p.role == 'werewolf']
    if werewolves and visualizer.E_history[werewolves[0]]:
        print("\n1. E状態の時系列変化（人狼の罪悪感蓄積）")
        visualizer.plot_E_evolution(werewolves[0])
    
    # 2. シグナルヒートマップ
    print("\n2. シグナルヒートマップ（創発的生成）")
    visualizer.plot_signal_heatmap(day=min(2, game.day_count))
    
    # 3. 意味圧フロー
    if werewolves and len(visualizer.pressure_history[werewolves[0]]) > 1:
        print("\n3. 意味圧フロー（SSD理論パイプライン）")
        pressure_day = visualizer.pressure_history[werewolves[0]][1]
        if pressure_day:
            target = list(pressure_day.keys())[0]
            visualizer.plot_pressure_flow(werewolves[0], target, day=1)
    
    # 4. 投票ネットワーク
    print("\n4. 投票ネットワーク")
    visualizer.plot_vote_network(day=min(2, game.day_count))
    
    # 5. 概念ネットワーク（学習が進んでいれば）
    for pid, player in game.players.items():
        if player.learned_concepts:
            print(f"\n5. 概念ネットワーク ({player.name})")
            visualizer.plot_concept_network(pid)
            break
    
    print("\n" + "="*70)
    print("Phase 10.4 完了: 可視化フレーバー追加")
    print("="*70)
    print("\n理論的コア（v10.2.1）は不変、可視化のみ追加")


if __name__ == "__main__":
    demo_visualizer()
