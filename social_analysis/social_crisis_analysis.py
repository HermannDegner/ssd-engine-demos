"""
社会危機分析 - SSD Engine を用いた集団パニックと社会崩壊の分析
================================================================

このデモでは、以下の危機的社会現象をシミュレートします:

1. 集団パニック（Mass Panic）
   - デマの拡散と集団心理
   - 恐怖の伝染メカニズム

2. 社会規範の崩壊（Norm Breakdown）
   - ストレス下での規範違反の連鎖
   - 秩序の喪失プロセス

3. 集団分極化の極端化（Extreme Polarization）
   - エコーチェンバー効果
   - 対話不能状態への移行

4. カリスマ的リーダーシップ（Charismatic Leadership）
   - 危機時の強力なリーダーの影響
   - 同調圧力とアイデンティティ変容
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssd_engine.core.ssd_human_module import HumanAgent, HumanPressure, HumanLayer, HumanParams
from ssd_engine.extensions.ssd_social_dynamics import (
    SocialCouplingParams, RelationshipMatrix, SocialDynamicsEngine
)
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False


class CrisisAnalyzer:
    """社会危機分析器"""
    
    def __init__(self, num_agents: int, scenario_name: str):
        self.num_agents = num_agents
        self.scenario_name = scenario_name
        self.agents: List[HumanAgent] = []
        self.society = None
        self.history = {
            'E_physical': [],
            'E_base': [],
            'E_core': [],
            'E_upper': [],
            'kappa_base': [],
            'kappa_core': [],
            'kappa_upper': [],
            'panic_level': [],
            'norm_violation': [],
        }
        
    def calculate_panic_level(self) -> float:
        """集団パニックレベルの計算"""
        # BASE層のエネルギーが高いほどパニック
        base_energies = [agent.state.E[HumanLayer.BASE.value] for agent in self.agents]
        return np.mean(base_energies)
    
    def calculate_norm_violation(self) -> float:
        """規範違反レベルの計算"""
        # CORE層のエネルギーが高く、κが低いほど規範違反
        violations = []
        for agent in self.agents:
            E_core = agent.state.E[HumanLayer.CORE.value]
            kappa_core = agent.state.kappa[HumanLayer.CORE.value]
            violation = E_core / (kappa_core + 0.1)  # κが低いほど高い
            violations.append(violation)
        return np.mean(violations)
    
    def record_state(self):
        """状態の記録"""
        self.history['E_physical'].append([a.state.E[HumanLayer.PHYSICAL.value] for a in self.agents])
        self.history['E_base'].append([a.state.E[HumanLayer.BASE.value] for a in self.agents])
        self.history['E_core'].append([a.state.E[HumanLayer.CORE.value] for a in self.agents])
        self.history['E_upper'].append([a.state.E[HumanLayer.UPPER.value] for a in self.agents])
        self.history['kappa_base'].append([a.state.kappa[HumanLayer.BASE.value] for a in self.agents])
        self.history['kappa_core'].append([a.state.kappa[HumanLayer.CORE.value] for a in self.agents])
        self.history['kappa_upper'].append([a.state.kappa[HumanLayer.UPPER.value] for a in self.agents])
        
        self.history['panic_level'].append(self.calculate_panic_level())
        self.history['norm_violation'].append(self.calculate_norm_violation())
    
    def visualize_crisis(self):
        """危機の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'社会危機分析: {self.scenario_name}', fontsize=16, fontweight='bold')
        
        steps = range(len(self.history['panic_level']))
        
        # パニックレベルの推移
        ax1 = axes[0, 0]
        ax1.plot(steps, self.history['panic_level'], 'r-', linewidth=2, label='パニックレベル')
        ax1.fill_between(steps, 0, self.history['panic_level'], alpha=0.3, color='red')
        ax1.set_title('集団パニックレベル', fontweight='bold')
        ax1.set_xlabel('ステップ')
        ax1.set_ylabel('パニック度')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 規範違反レベルの推移
        ax2 = axes[0, 1]
        ax2.plot(steps, self.history['norm_violation'], 'orange', linewidth=2, label='規範違反')
        ax2.fill_between(steps, 0, self.history['norm_violation'], alpha=0.3, color='orange')
        ax2.set_title('規範違反レベル', fontweight='bold')
        ax2.set_xlabel('ステップ')
        ax2.set_ylabel('違反度')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # BASE層エネルギー分布
        ax3 = axes[0, 2]
        for i in range(min(5, self.num_agents)):  # 最大5人表示
            E_base_i = [step[i] for step in self.history['E_base']]
            ax3.plot(steps, E_base_i, label=f'Agent {i+1}', alpha=0.7)
        ax3.set_title('BASE層エネルギー（恐怖・怒り）', fontweight='bold')
        ax3.set_xlabel('ステップ')
        ax3.set_ylabel('E_BASE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # CORE層κ（規範の強さ）
        ax4 = axes[1, 0]
        for i in range(min(5, self.num_agents)):
            kappa_core_i = [step[i] for step in self.history['kappa_core']]
            ax4.plot(steps, kappa_core_i, label=f'Agent {i+1}', alpha=0.7)
        ax4.set_title('CORE層κ（規範の強さ）', fontweight='bold')
        ax4.set_xlabel('ステップ')
        ax4.set_ylabel('κ_CORE')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # UPPER層κ（イデオロギーの強さ）
        ax5 = axes[1, 1]
        for i in range(min(5, self.num_agents)):
            kappa_upper_i = [step[i] for step in self.history['kappa_upper']]
            ax5.plot(steps, kappa_upper_i, label=f'Agent {i+1}', alpha=0.7)
        ax5.set_title('UPPER層κ（イデオロギーの強さ）', fontweight='bold')
        ax5.set_xlabel('ステップ')
        ax5.set_ylabel('κ_UPPER')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # エネルギー総量
        ax6 = axes[1, 2]
        total_E = []
        for i in range(len(steps)):
            total = 0
            for layer in ['E_base', 'E_core', 'E_upper']:
                total += np.sum(self.history[layer][i])
            total_E.append(total)
        ax6.plot(steps, total_E, 'purple', linewidth=2, label='総エネルギー')
        ax6.set_title('システム全体のエネルギー', fontweight='bold')
        ax6.set_xlabel('ステップ')
        ax6.set_ylabel('総E')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'crisis_analysis_{self.scenario_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 可視化結果を保存: {filename}")
        plt.show()


class MassPanicScenario(CrisisAnalyzer):
    """集団パニックシナリオ
    
    デマや恐怖が急速に拡散し、集団パニックに至るプロセス
    """
    
    def __init__(self, num_agents: int = 20):
        super().__init__(num_agents, "集団パニック")
        
    def setup(self):
        """セットアップ"""
        print("\n" + "="*60)
        print("シナリオ: 集団パニック")
        print("="*60)
        print("\n状況: 大規模災害の噂が広まり、人々がパニック状態に...")
        
        # エージェント作成（初期状態は平穏）
        for i in range(self.num_agents):
            agent = HumanAgent()
            agent.state.E[HumanLayer.BASE.value] = np.random.uniform(0.5, 1.5)
            agent.state.kappa[HumanLayer.CORE.value] = np.random.uniform(1.0, 2.0)
            self.agents.append(agent)
        
        # 協力的な関係（恐怖が伝播しやすい）
        relationships = RelationshipMatrix.create_cooperative(self.num_agents)
        
        # 社会的カップリングを強化（恐怖の伝染を強く）
        coupling_params = SocialCouplingParams()
        coupling_params.zeta_base = 0.15  # BASE層エネルギー伝播を強化
        
        self.society = SocialDynamicsEngine(
            agents=self.agents,
            relationships=relationships,
            params=coupling_params
        )
        
    def run(self, num_steps: int = 150):
        """実行"""
        print(f"\nシミュレーション開始（{num_steps}ステップ）...")
        
        for step in range(num_steps):
            self.record_state()
            
            # フェーズ1: 平穏期（0-30）
            if step < 30:
                pressure = HumanPressure()
                pressure.base = np.random.uniform(0.1, 0.5)
                
            # フェーズ2: デマ発生（30-50）
            elif step < 50:
                pressure = HumanPressure()
                # 1人に強い恐怖（デマの発信源）
                if step == 30:
                    print(f"\nStep {step}: ⚠️ デマ発生！最初の恐怖反応...")
                    self.agents[0].state.E[HumanLayer.BASE.value] = 15.0
                pressure.base = np.random.uniform(1.0, 3.0)
                
            # フェーズ3: パニック拡大（50-100）
            elif step < 100:
                if step == 50:
                    print(f"Step {step}: 🔥 恐怖が伝播中...")
                pressure = HumanPressure()
                pressure.base = np.random.uniform(2.0, 5.0)
                
            # フェーズ4: 収束試行（100-150）
            else:
                if step == 100:
                    print(f"Step {step}: 🛑 沈静化の試み...")
                pressure = HumanPressure()
                pressure.base = np.random.uniform(0.5, 1.0)
                pressure.core = 1.5  # 規範的圧力（落ち着こう）
            
            # 圧力適用
            for agent in self.agents:
                agent.step(pressure)
            
            # 社会的相互作用（恐怖の伝播）
            self.society.step()
            
            # 進捗表示
            if (step + 1) % 30 == 0:
                panic = self.calculate_panic_level()
                print(f"  Step {step+1}: パニックレベル = {panic:.2f}")
        
        print("\n✅ シミュレーション完了")


class NormBreakdownScenario(CrisisAnalyzer):
    """規範崩壊シナリオ
    
    ストレスの蓄積により、社会規範が次々と破られていくプロセス
    """
    
    def __init__(self, num_agents: int = 15):
        super().__init__(num_agents, "規範の崩壊")
        
    def setup(self):
        """セットアップ"""
        print("\n" + "="*60)
        print("シナリオ: 規範の崩壊")
        print("="*60)
        print("\n状況: 経済危機下で、人々が規範を破り始める...")
        
        # エージェント作成（最初は規範意識が高い）
        for i in range(self.num_agents):
            agent = HumanAgent()
            agent.state.kappa[HumanLayer.CORE.value] = np.random.uniform(2.0, 3.0)  # 高い規範
            agent.state.kappa[HumanLayer.BASE.value] = np.random.uniform(1.0, 1.5)
            self.agents.append(agent)
        
        # 1人だけ規範が弱い（最初の違反者）
        self.agents[0].state.kappa[HumanLayer.CORE.value] = 0.8
        
        # 協力的関係（規範違反が伝播）
        relationships = RelationshipMatrix.create_cooperative(self.num_agents)
        coupling_params = SocialCouplingParams()
        
        self.society = SocialDynamicsEngine(
            agents=self.agents,
            relationships=relationships,
            params=coupling_params
        )
        
    def run(self, num_steps: int = 200):
        """実行"""
        print(f"\nシミュレーション開始（{num_steps}ステップ）...")
        
        for step in range(num_steps):
            self.record_state()
            
            # フェーズ1: 正常期（0-40）
            if step < 40:
                pressure = HumanPressure()
                pressure.core = 0.5  # 軽い規範的圧力
                
            # フェーズ2: ストレス増加（40-80）
            elif step < 80:
                if step == 40:
                    print(f"\nStep {step}: 📉 経済危機開始...")
                pressure = HumanPressure()
                pressure.base = np.random.uniform(2.0, 4.0)  # 生存圧力
                pressure.core = np.random.uniform(1.0, 2.0)  # 規範との葛藤
                
            # フェーズ3: 最初の違反（80-120）
            elif step < 120:
                if step == 80:
                    print(f"Step {step}: ⚠️ 最初の規範違反...")
                    # 最初の違反者のCORE κを破壊
                    self.agents[0].state.E[HumanLayer.CORE.value] = 10.0
                
                pressure = HumanPressure()
                pressure.base = np.random.uniform(3.0, 6.0)
                pressure.core = np.random.uniform(1.5, 3.0)
                
            # フェーズ4: 連鎖崩壊（120-200）
            else:
                if step == 120:
                    print(f"Step {step}: 🔥 規範崩壊の連鎖...")
                pressure = HumanPressure()
                pressure.base = np.random.uniform(4.0, 7.0)
                pressure.core = np.random.uniform(2.0, 4.0)
            
            # 圧力適用
            for agent in self.agents:
                agent.step(pressure)
            
            # 社会的相互作用
            self.society.step()
            
            # 進捗表示
            if (step + 1) % 40 == 0:
                violation = self.calculate_norm_violation()
                print(f"  Step {step+1}: 規範違反レベル = {violation:.2f}")
        
        print("\n✅ シミュレーション完了")


class CharismaticLeaderScenario(CrisisAnalyzer):
    """カリスマ的リーダーシナリオ
    
    強力なリーダーが出現し、集団を特定の方向に導くプロセス
    """
    
    def __init__(self, num_agents: int = 12):
        super().__init__(num_agents, "カリスマ的リーダーシップ")
        self.leader_index = 0
        
    def setup(self):
        """セットアップ"""
        print("\n" + "="*60)
        print("シナリオ: カリスマ的リーダーシップ")
        print("="*60)
        print("\n状況: 危機的状況下で強力なリーダーが現れる...")
        
        # リーダー（Agent 0）
        leader = HumanAgent()
        leader.state.kappa[HumanLayer.UPPER.value] = 4.0  # 極めて強いイデオロギー
        leader.state.kappa[HumanLayer.CORE.value] = 3.5   # 強い規範意識
        leader.state.kappa[HumanLayer.BASE.value] = 2.0   # 本能的カリスマ
        self.agents.append(leader)
        
        # フォロワー（残りのエージェント）
        for i in range(1, self.num_agents):
            agent = HumanAgent()
            agent.state.kappa[HumanLayer.UPPER.value] = np.random.uniform(0.5, 1.0)
            agent.state.kappa[HumanLayer.CORE.value] = np.random.uniform(1.0, 1.5)
            agent.state.E[HumanLayer.BASE.value] = np.random.uniform(2.0, 4.0)  # 不安
            self.agents.append(agent)
        
        # 関係性: リーダーへの一方的な影響
        relation_matrix = np.zeros((self.num_agents, self.num_agents))
        for i in range(1, self.num_agents):
            relation_matrix[i, 0] = 0.9  # フォロワー→リーダー（強い影響）
            relation_matrix[0, i] = 0.3  # リーダー→フォロワー（弱い影響）
        
        relationships = RelationshipMatrix(matrix=relation_matrix)
        
        # κ伝播を強化（リーダーのイデオロギーが伝わりやすい）
        coupling_params = SocialCouplingParams()
        coupling_params.xi_upper = 0.15  # UPPER層κ伝播強化
        coupling_params.xi_core = 0.12   # CORE層κ伝播強化
        
        self.society = SocialDynamicsEngine(
            agents=self.agents,
            relationships=relationships,
            params=coupling_params
        )
        
        print(f"  Agent 1: カリスマ的リーダー")
        print(f"  Agent 2-{self.num_agents}: 不安な民衆")
        
    def run(self, num_steps: int = 150):
        """実行"""
        print(f"\nシミュレーション開始（{num_steps}ステップ）...")
        
        for step in range(num_steps):
            self.record_state()
            
            # リーダーには特別な圧力（使命感）
            leader_pressure = HumanPressure()
            leader_pressure.upper = 2.0  # 強いイデオロギー的圧力
            self.agents[0].step(leader_pressure)
            
            # フォロワーには不安と混乱
            follower_pressure = HumanPressure()
            
            if step < 50:
                # フェーズ1: 混乱期
                follower_pressure.base = np.random.uniform(3.0, 5.0)
                if step == 0:
                    print(f"\nStep {step}: 😰 混乱と不安の時期...")
                    
            elif step < 100:
                # フェーズ2: リーダー台頭
                follower_pressure.base = np.random.uniform(2.0, 4.0)
                follower_pressure.upper = np.random.uniform(0.5, 1.5)
                if step == 50:
                    print(f"Step {step}: 👑 リーダーの影響力拡大...")
                    
            else:
                # フェーズ3: 統一期
                follower_pressure.base = np.random.uniform(1.0, 2.0)
                follower_pressure.upper = np.random.uniform(1.0, 2.0)
                if step == 100:
                    print(f"Step {step}: 🎯 集団の統一化...")
            
            # フォロワーに圧力適用
            for i in range(1, self.num_agents):
                self.agents[i].step(follower_pressure)
            
            # 社会的相互作用（リーダーの影響伝播）
            self.society.step()
            
            # 進捗表示
            if (step + 1) % 30 == 0:
                # イデオロギーの統一度を計算
                upper_kappas = [a.state.kappa[HumanLayer.UPPER.value] for a in self.agents[1:]]
                uniformity = 1.0 / (np.std(upper_kappas) + 0.1)
                print(f"  Step {step+1}: イデオロギー統一度 = {uniformity:.2f}")
        
        print("\n✅ シミュレーション完了")


def main():
    """メイン実行"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║          SSDエンジン 社会危機分析                              ║
║      Social Crisis Analysis with SSD Engine                    ║
╚════════════════════════════════════════════════════════════════╝

このデモでは、危機的な社会現象を分析します:

1. 集団パニック
   - デマの拡散と恐怖の伝染

2. 規範の崩壊
   - ストレス下での規範違反の連鎖

3. カリスマ的リーダーシップ
   - 危機時のリーダーの影響力
    """)
    
    print("\n実行するシナリオを選択してください:")
    print("1: 集団パニック")
    print("2: 規範の崩壊")
    print("3: カリスマ的リーダーシップ")
    print("4: すべて実行")
    
    choice = input("\n選択 (1-4): ").strip()
    
    scenarios = []
    
    if choice == '1':
        scenario = MassPanicScenario()
        scenario.setup()
        scenario.run()
        scenario.visualize_crisis()
        
    elif choice == '2':
        scenario = NormBreakdownScenario()
        scenario.setup()
        scenario.run()
        scenario.visualize_crisis()
        
    elif choice == '3':
        scenario = CharismaticLeaderScenario()
        scenario.setup()
        scenario.run()
        scenario.visualize_crisis()
        
    elif choice == '4':
        for ScenarioClass in [MassPanicScenario, NormBreakdownScenario, CharismaticLeaderScenario]:
            scenario = ScenarioClass()
            scenario.setup()
            scenario.run()
            scenario.visualize_crisis()
            print("\n" + "="*60)
            input("次のシナリオに進むにはEnterキーを押してください...")
    
    print("\n✅ すべての分析が完了しました！")


if __name__ == "__main__":
    main()
