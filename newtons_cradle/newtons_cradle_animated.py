"""
ニュートンのゆりかご - SSD Core Engine駆動版
Newton's Cradle Driven by SSD Core Engine

【SSDコアエンジンで駆動】
- 運動をSSD 4層レイヤーで表現
- 重力・衝突を意味圧として入力
- E/κダイナミクスから運動が創発
- 跳躍による非線形挙動

作成日: 2025年11月7日
バージョン: 2.0 (SSD Core Driven)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import sys
import os

# 親ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

# coreモジュールのパス追加
core_path = os.path.join(grandparent_dir, 'core')
sys.path.insert(0, core_path)

from core.ssd_core_engine import (
    SSDCoreEngine, SSDCoreParams, SSDCoreState, 
    create_default_state, LeapType
)


class Ball:
    """球（SSD Core Engine駆動版）
    
    4層レイヤー構造:
    Layer 0: 位置（慣性・角度）
    Layer 1: 速度（運動量）
    Layer 2: 加速度（力の応答）
    Layer 3: エネルギー散逸
    """
    
    def __init__(self, ball_id: int, initial_position: float, mass: float = 1.0):
        self.ball_id = ball_id
        self.mass = mass
        
        # 物理状態（SSDから創発）
        self.position = initial_position  # 角度（ラジアン×弦長）
        self.velocity = 0.0
        self.acceleration = 0.0
        
        # SSD Core Engine
        params = SSDCoreParams(
            num_layers=4,
            # 位置、速度、加速度、散逸の4層
            R_values=[100.0, 50.0, 20.0, 10.0],
            gamma_values=[0.2, 0.15, 0.1, 0.05],
            beta_values=[0.01, 0.02, 0.05, 0.1],
            eta_values=[0.8, 0.6, 0.4, 0.2],
            lambda_values=[0.005, 0.01, 0.02, 0.05],
            kappa_min_values=[0.5, 0.4, 0.3, 0.2],
            Theta_values=[150.0, 100.0, 50.0, 30.0],
            G0=0.3,
            g=0.5,
            log_align=True,
            warmup_steps=10
        )
        self.engine = SSDCoreEngine(params)
        self.state = create_default_state(num_layers=4)
        
        # κ初期値：位置は慣性大、散逸は小
        self.state.kappa = np.array([2.0, 1.5, 1.0, 0.5])
        
        # 衝突記録
        self.collision_count = 0
        self.total_impact = 0.0
        self.last_collision_time = 0.0
        self.leap_count = 0
    
    def apply_gravity_pressure(self, gravity: float, string_length: float) -> np.ndarray:
        """重力による意味圧を計算"""
        angle = self.position / string_length
        
        # 重力による復元力（振り子）
        gravity_force = -gravity * np.sin(angle) / string_length
        
        # 4層への意味圧分配
        # Layer 0 (位置): 角度の偏差
        # Layer 1 (速度): 力による加速要求
        # Layer 2 (加速): 直接的な力
        # Layer 3 (散逸): 速度に比例した抵抗
        pressure = np.array([
            abs(angle) * 10.0,           # 位置偏差圧
            abs(gravity_force) * 20.0,   # 加速要求圧
            abs(gravity_force) * 50.0,   # 力の圧力
            abs(self.velocity) * 5.0     # 運動抵抗圧
        ])
        
        return pressure
    
    def apply_collision_pressure(self, impact_velocity: float) -> np.ndarray:
        """衝突による意味圧を計算"""
        impact_magnitude = abs(impact_velocity)
        
        # 衝突は全層に強い圧力
        pressure = np.array([
            impact_magnitude * 30.0,  # 位置への衝撃
            impact_magnitude * 50.0,  # 速度変化要求
            impact_magnitude * 80.0,  # 急激な力
            impact_magnitude * 40.0   # 散逸増加
        ])
        
        return pressure
    
    def update_from_ssd(self, dt: float, gravity: float, string_length: float):
        """SSDエンジンから物理状態を更新
        
        【物理学対応 - Core Engineの電気回路アナロジーを活用】
        - Ohm's law: j = (G0 + g·κ)·p̂ ← Engineで計算済み
        - エネルギー生成: γ·residual/R ← 抵抗での散逸
        - エネルギー減衰: β·E ← キャパシタの放電
        - κ学習: η·usage - λ·κ ← インダクタンスの変化
        
        物理解釈:
        - 意味圧 p → 電圧（重力ポテンシャル）
        - 整合流 j → 電流（運動応答）
        - 未処理圧 E → 蓄積電荷（キャパシタ）
        - 慣性 κ → インダクタンス（慣性質量）
        - 抵抗 R → 粘性・摩擦
        """
        # 重力による意味圧を計算
        gravity_pressure = self.apply_gravity_pressure(gravity, string_length)
        
        # === SSD Core Engineステップ実行 ===
        # ここでOhm's law、エネルギー生成/減衰、κ更新が全て行われる
        old_E = self.state.E.copy()
        old_kappa = self.state.kappa.copy()
        
        self.state = self.engine.step(self.state, gravity_pressure, dt=dt)
        
        # === Engineの診断情報を取得 ===
        diag = self.state.diagnostics
        
        # 整合効率 η = |j| / |p| （どれだけ処理できたか）
        eta_align = diag.get('eta_align_log', 0.5)
        
        # 支配レイヤー（最も影響力が大きい層）
        dominant_layer = diag.get('dominant_layer', 1)
        
        # === 物理状態への変換 ===
        # Layer 1 (速度層) のE蓄積とκから加速度を決定
        velocity_E = self.state.E[1]
        velocity_kappa = self.state.kappa[1]
        
        # 角度から基本加速度（振り子の方程式）
        angle = self.position / string_length
        base_acceleration = -(gravity / string_length) * np.sin(angle)
        
        # SSDによる加速度修正
        # Eが高い = キャパシタに電荷蓄積 = 放電による加速
        # κが高い = インダクタンス大 = 変化しにくい（慣性大）
        E_acceleration = velocity_E / (velocity_kappa * 5.0 + 1.0)
        
        # 整合効率が高い = Ohm's lawで電流が流れやすい = 応答良好
        efficiency_factor = 0.5 + eta_align * 0.5
        
        self.acceleration = base_acceleration * efficiency_factor * (1.0 + E_acceleration * 0.2)
        
        # 速度更新（ニュートンの第二法則）
        self.velocity += self.acceleration * dt
        
        # === 熱力学的散逸の運動への反映 ===
        # Core Engineで既に実装済み: dE = γ·residual - β·E
        # β·E項が熱力学第二法則（エントロピー増大→自然散逸）
        # 
        # 物理解釈:
        # - Layer 3 (散逸層) のE: 摩擦・粘性による熱エネルギー蓄積
        # - β·Eによる減衰: 熱の放散（系外への散逸）
        # - 運動への影響: 蓄積された熱Eが速度を減衰させる
        
        # Engineで更新されたE[3]を使って速度減衰
        # （前ステップの熱蓄積が今の運動に影響）
        heat_dissipation = self.state.E[3] / (self.state.kappa[3] + 1.0)
        damping_coefficient = heat_dissipation * 0.003  # 粘性係数
        self.velocity *= (1.0 - damping_coefficient)
        
        # 位置更新
        self.position += self.velocity * dt
        
        # === 相転移（跳躍） ===
        if self.state.leap_history and len(self.state.leap_history) > self.leap_count:
            self.leap_count = len(self.state.leap_history)
            leap_type = self.state.leap_history[-1][1]
            
            # 跳躍 = 量子トンネル効果 / 臨界現象
            # べき乗則的な摂動（スケールフリー）
            leap_magnitude = 0.08 * (1.0 + np.random.power(2.0))
            self.velocity *= (1.0 + np.random.randn() * leap_magnitude)
            
            # 位置への量子的摂動
            self.position += np.random.randn() * 0.015 * string_length
    
    def apply_impact(self, impact_velocity: float, current_time: float):
        """衝突を適用（SSD駆動）"""
        # 速度を直接設定
        self.velocity = impact_velocity
        
        # 衝突圧を計算
        collision_pressure = self.apply_collision_pressure(impact_velocity)
        
        # SSDに衝突を伝える
        self.state = self.engine.step(self.state, collision_pressure, dt=0.001)
        
        # 記録
        self.collision_count += 1
        self.total_impact += abs(impact_velocity)
        self.last_collision_time = current_time
    
    def get_kinetic_energy(self) -> float:
        """運動エネルギー"""
        return 0.5 * self.mass * (self.velocity ** 2)
    
    def get_potential_energy(self, string_length: float = 2.0) -> float:
        """位置エネルギー"""
        angle = self.position / string_length
        height = string_length * (1.0 - np.cos(angle))
        return self.mass * 9.8 * height


class NewtonsCradleAnimated:
    """ニュートンのゆりかご - アニメーション版"""
    
    def __init__(self, n_balls: int = 5, spacing: float = 1.0,
                 string_length: float = 2.0, initial_release_angle: float = 30.0):
        self.n_balls = n_balls
        self.string_length = string_length
        self.gravity = 9.8
        self.radius = 0.5
        self.mass = 1.0
        self.spacing = spacing
        
        # 球の初期化
        self.balls = []
        for i in range(n_balls):
            initial_pos = 0.0
            ball = Ball(ball_id=i, initial_position=initial_pos, mass=self.mass)
            self.balls.append(ball)
        
        # 初期条件: 最初の球を持ち上げる
        release_angle_rad = np.radians(initial_release_angle)
        self.balls[0].position = release_angle_rad * string_length
        
        # シミュレーション状態
        self.current_time = 0.0
        self.total_steps = 0
        
        # エネルギー履歴
        self.energy_history = []
        self.initial_energy = None
    
    def detect_collisions(self):
        """衝突検出"""
        collisions = []
        
        for i in range(self.n_balls - 1):
            ball1 = self.balls[i]
            ball2 = self.balls[i + 1]
            
            # 支点からの水平位置
            x1 = (i - self.n_balls/2) * self.spacing + self.string_length * np.sin(ball1.position / self.string_length)
            x2 = (i + 1 - self.n_balls/2) * self.spacing + self.string_length * np.sin(ball2.position / self.string_length)
            
            distance = abs(x2 - x1)
            
            if distance <= self.radius * 2.0 * 1.01:
                relative_velocity = ball1.velocity - ball2.velocity
                if (x1 < x2 and relative_velocity > 0) or (x1 > x2 and relative_velocity < 0):
                    collisions.append((i, i+1))
        
        return collisions
    
    def resolve_collision(self, ball1_id: int, ball2_id: int):
        """衝突解決"""
        ball1 = self.balls[ball1_id]
        ball2 = self.balls[ball2_id]
        
        v1 = ball1.velocity
        v2 = ball2.velocity
        m1 = ball1.mass
        m2 = ball2.mass
        
        # 完全弾性衝突
        v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        v2_new = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
        
        # 衝突適用
        ball1.apply_impact(v1_new, self.current_time)
        ball2.apply_impact(v2_new, self.current_time)
    
    def step(self, dt: float = 0.001):
        """1ステップ（SSD駆動）"""
        # 各球をSSDで更新
        for ball in self.balls:
            ball.update_from_ssd(dt, self.gravity, self.string_length)
        
        # 衝突検出・解決
        collisions = self.detect_collisions()
        for ball1_id, ball2_id in collisions:
            self.resolve_collision(ball1_id, ball2_id)
        
        # 時刻更新
        self.current_time += dt
        self.total_steps += 1
        
        # エネルギー記録
        total_energy = sum(b.get_kinetic_energy() + b.get_potential_energy(self.string_length) 
                          for b in self.balls)
        self.energy_history.append(total_energy)
        
        if self.initial_energy is None:
            self.initial_energy = total_energy
    
    def get_ball_position_xy(self, ball_id: int) -> tuple:
        """球のXY座標"""
        ball = self.balls[ball_id]
        angle = ball.position / self.string_length
        
        # 支点位置
        support_x = (ball_id - self.n_balls/2) * self.spacing
        
        # 球の位置
        x = support_x + self.string_length * np.sin(angle)
        y = -self.string_length * np.cos(angle)
        
        return (x, y)
    
    def get_support_position(self, ball_id: int) -> float:
        """支点のX座標"""
        return (ball_id - self.n_balls/2) * self.spacing


class CradleVisualizer:
    """アニメーションビジュアライザー"""
    
    def __init__(self, cradle: NewtonsCradleAnimated):
        self.cradle = cradle
        
        # Figure作成
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("Newton's Cradle with SSD - ニュートンのゆりかご", 
                         fontsize=16, fontweight='bold')
        
        # サブプロット
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        self.ax_pendulum = self.fig.add_subplot(gs[0, :])  # 上段全体: 振り子
        self.ax_energy = self.fig.add_subplot(gs[1, 0])    # 下段左: エネルギー
        self.ax_ssd = self.fig.add_subplot(gs[1, 1])       # 下段右: SSD状態
        
        # 初期化
        self.init_pendulum_plot()
    
    def init_pendulum_plot(self):
        """振り子プロット初期化"""
        self.ax_pendulum.clear()
        self.ax_pendulum.set_xlim(-4, 4)
        self.ax_pendulum.set_ylim(-3, 1)
        self.ax_pendulum.set_aspect('equal')
        self.ax_pendulum.set_title('Physical Simulation', fontweight='bold', fontsize=12)
        self.ax_pendulum.grid(True, alpha=0.3)
        
        # 支点を描画
        support_positions = [self.cradle.get_support_position(i) for i in range(self.cradle.n_balls)]
        self.ax_pendulum.plot(support_positions, [0] * len(support_positions), 
                             'ko-', markersize=10, linewidth=3, zorder=5)
        self.ax_pendulum.axhline(y=0, color='black', linewidth=2, alpha=0.5)
    
    def update_frame(self, frame):
        """フレーム更新"""
        # 複数ステップ実行（スムーズなアニメーション）
        for _ in range(5):
            self.cradle.step(dt=0.001)
        
        # 描画更新
        self.draw_pendulums()
        self.draw_energy()
        self.draw_ssd_state()
        
        return []
    
    def draw_pendulums(self):
        """振り子描画"""
        self.ax_pendulum.clear()
        self.ax_pendulum.set_xlim(-4, 4)
        self.ax_pendulum.set_ylim(-3, 1)
        self.ax_pendulum.set_aspect('equal')
        self.ax_pendulum.set_title(f'Physical Simulation (t={self.cradle.current_time:.2f}s)', 
                                   fontweight='bold', fontsize=12)
        self.ax_pendulum.grid(True, alpha=0.3)
        
        # 支点
        support_positions = [self.cradle.get_support_position(i) for i in range(self.cradle.n_balls)]
        self.ax_pendulum.plot(support_positions, [0] * len(support_positions), 
                             'ko-', markersize=10, linewidth=3, zorder=5)
        self.ax_pendulum.axhline(y=0, color='black', linewidth=2, alpha=0.5)
        
        # 各球
        for i in range(self.cradle.n_balls):
            ball = self.cradle.balls[i]
            support_x = self.cradle.get_support_position(i)
            ball_x, ball_y = self.cradle.get_ball_position_xy(i)
            
            # 紐
            self.ax_pendulum.plot([support_x, ball_x], [0, ball_y], 
                                 'k-', linewidth=2, alpha=0.7, zorder=1)
            
            # 球（衝突中は赤、通常は青）
            color = 'red' if ball.collision_count > 0 and (self.cradle.current_time - ball.last_collision_time) < 0.1 else 'blue'
            circle = Circle((ball_x, ball_y), self.cradle.radius, 
                          color=color, alpha=0.8, zorder=10)
            self.ax_pendulum.add_patch(circle)
            
            # 球のID
            self.ax_pendulum.text(ball_x, ball_y, str(i), 
                                ha='center', va='center', 
                                fontsize=10, fontweight='bold', color='white', zorder=11)
            
            # 速度ベクトル
            if abs(ball.velocity) > 0.1:
                angle = ball.position / self.cradle.string_length
                vx = ball.velocity * np.cos(angle) * 0.3
                vy = ball.velocity * np.sin(angle) * 0.3
                self.ax_pendulum.arrow(ball_x, ball_y, vx, vy,
                                      head_width=0.15, head_length=0.1,
                                      fc='green', ec='green', alpha=0.7, zorder=9)
    
    def draw_energy(self):
        """エネルギープロット"""
        self.ax_energy.clear()
        self.ax_energy.set_title('Energy Conservation', fontweight='bold', fontsize=11)
        self.ax_energy.set_xlabel('Time Step')
        self.ax_energy.set_ylabel('Energy (J)')
        
        if len(self.cradle.energy_history) > 0:
            steps = list(range(len(self.cradle.energy_history)))
            self.ax_energy.plot(steps, self.cradle.energy_history, 
                              'b-', linewidth=1.5, label='Total Energy')
            
            if self.cradle.initial_energy is not None:
                self.ax_energy.axhline(self.cradle.initial_energy, 
                                      color='r', linestyle='--', linewidth=1.5, 
                                      alpha=0.7, label='Initial Energy')
            
            self.ax_energy.legend(fontsize=9)
            self.ax_energy.grid(True, alpha=0.3)
    
    def draw_ssd_state(self):
        """SSD状態プロット（Core Engine版）"""
        self.ax_ssd.clear()
        self.ax_ssd.set_title('SSD Core Engine State', fontweight='bold', fontsize=11)
        self.ax_ssd.set_xlabel('Ball ID')
        self.ax_ssd.set_ylabel('E (Energy Accumulation)')
        
        ball_ids = list(range(self.cradle.n_balls))
        
        # E蓄積（4層: 位置、速度、加速度、散逸）
        E_layer0 = [b.state.E[0] for b in self.cradle.balls]
        E_layer1 = [b.state.E[1] for b in self.cradle.balls]
        E_layer2 = [b.state.E[2] for b in self.cradle.balls]
        E_layer3 = [b.state.E[3] for b in self.cradle.balls]
        
        width = 0.2
        self.ax_ssd.bar([i - 1.5*width for i in ball_ids], E_layer0, width, 
                       label='Position', alpha=0.7, color='blue')
        self.ax_ssd.bar([i - 0.5*width for i in ball_ids], E_layer1, width, 
                       label='Velocity', alpha=0.7, color='green')
        self.ax_ssd.bar([i + 0.5*width for i in ball_ids], E_layer2, width, 
                       label='Accel', alpha=0.7, color='orange')
        self.ax_ssd.bar([i + 1.5*width for i in ball_ids], E_layer3, width, 
                       label='Dissipation', alpha=0.7, color='red')
        
        self.ax_ssd.legend(fontsize=8, loc='upper right')
        self.ax_ssd.grid(True, alpha=0.3)
        
        # 統計情報（SSD Core版）
        total_collisions = sum(b.collision_count for b in self.cradle.balls)
        total_leaps = sum(b.leap_count for b in self.cradle.balls)
        total_E = sum(np.sum(b.state.E) for b in self.cradle.balls)
        avg_kappa = np.mean([np.mean(b.state.kappa) for b in self.cradle.balls])
        
        stats_text = f'Collisions: {total_collisions} | Leaps: {total_leaps} | Total E: {total_E:.2f} | Avg κ: {avg_kappa:.3f}'
        self.ax_ssd.text(0.5, 0.95, stats_text, 
                        transform=self.ax_ssd.transAxes,
                        ha='center', va='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def animate(self, frames: int = 1000, interval: int = 20):
        """アニメーション開始"""
        print(f"\nアニメーション開始: {frames}フレーム")
        print("ウィンドウを閉じると終了します\n")
        
        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_pendulum_plot,
            frames=frames,
            interval=interval,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim


def demo_classic():
    """クラシックデモ: 1球（SSD Core駆動）"""
    print("="*70)
    print("Newton's Cradle - SSD Core Engine Driven (1 Ball)")
    print("="*70)
    print("\nシナリオ: 左端の球を30度持ち上げて離す")
    print("期待: SSDエンジンの数式から運動が創発")
    print("  - 重力 → 意味圧 p")
    print("  - Ohm's law: j = (G0 + g·κ)·p̂")
    print("  - E蓄積 → 非線形挙動")
    print("  - 跳躍 → ランダム摂動\n")
    
    cradle = NewtonsCradleAnimated(
        n_balls=5,
        spacing=1.0,
        string_length=2.0,
        initial_release_angle=30.0
    )
    
    viz = CradleVisualizer(cradle)
    viz.animate(frames=1000, interval=20)


def demo_multiple():
    """複数球デモ: 2球（SSD Core駆動）"""
    print("="*70)
    print("Newton's Cradle - SSD Core Engine (2 Balls)")
    print("="*70)
    print("\nシナリオ: 左端2球を持ち上げて離す")
    print("期待: SSD 4層レイヤーから複雑な運動が創発\n")
    
    cradle = NewtonsCradleAnimated(
        n_balls=5,
        spacing=1.0,
        string_length=2.0,
        initial_release_angle=30.0
    )
    
    # 2球目も持ち上げる
    cradle.balls[1].position = np.radians(29.0) * cradle.string_length
    
    viz = CradleVisualizer(cradle)
    viz.animate(frames=1000, interval=20)


def demo_extreme():
    """極端デモ: 大きな角度（SSD非線形性）"""
    print("="*70)
    print("Newton's Cradle - SSD Nonlinear Dynamics (60 degrees)")
    print("="*70)
    print("\nシナリオ: 左端の球を60度持ち上げて離す")
    print("期待: 大きな意味圧 → Log-Alignment適応")
    print("      E蓄積増加 → 跳躍発生 → カオス的挙動\n")
    
    cradle = NewtonsCradleAnimated(
        n_balls=5,
        spacing=1.0,
        string_length=2.0,
        initial_release_angle=60.0
    )
    
    viz = CradleVisualizer(cradle)
    viz.animate(frames=1500, interval=20)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "multiple":
            demo_multiple()
        elif sys.argv[1] == "extreme":
            demo_extreme()
        else:
            demo_classic()
    else:
        demo_classic()
    
    print("\n" + "="*70)
    print("デモ完了!")
    print("="*70)
    print("\n💡 Tip:")
    print("  python examples/newtons_cradle/newtons_cradle_animated.py           # 1球（SSD Core駆動）")
    print("  python examples/newtons_cradle/newtons_cradle_animated.py multiple  # 2球（4層レイヤー）")
    print("  python examples/newtons_cradle/newtons_cradle_animated.py extreme   # 極端デモ（非線形性）")
    print("\n🔬 SSD Core Engine:")
    print("  - 4層レイヤー: Position, Velocity, Acceleration, Dissipation")
    print("  - Log-Alignment: p̂ = sign(p)·log(1+α_t|p|)/log(b)")
    print("  - Ohm's law: j = (G0 + g·κ)·p̂")
    print("  - E蓄積 → 跳躍 → カオス")
