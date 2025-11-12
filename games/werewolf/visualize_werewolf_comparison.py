"""
人狼ゲーム バージョン比較 - 可視化スクリプト
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False

# データ
versions = ['v6\n(2024前期)', 'v7\n(2024後期)', 'v8\n(2025初頭)', 'v8.5\n(2025/11/7)']
lines_of_code = [994, 846, 566, 670]
theoretical_alignment = [60, 75, 85, 100]
performance_7_players = [50, 30, 20, 4]  # ms
performance_100_players = [5000, 3000, 2000, 50]  # ms (推定含む)

# 図の作成
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('人狼ゲーム実装 - バージョン比較', fontsize=16, fontweight='bold')

# 1. コード行数
ax1 = axes[0, 0]
bars1 = ax1.bar(versions, lines_of_code, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_ylabel('コード行数', fontsize=12)
ax1.set_title('1. コード量の推移', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}行',
            ha='center', va='bottom', fontsize=10)

# 2. 理論的整合性
ax2 = axes[0, 1]
bars2 = ax2.bar(versions, theoretical_alignment, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax2.set_ylabel('理論的整合性 (%)', fontsize=12)
ax2.set_title('2. 理論的整合性の向上', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='目標: 100%')
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.legend()

# 3. パフォーマンス（7人ゲーム）
ax3 = axes[1, 0]
bars3 = ax3.bar(versions, performance_7_players, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax3.set_ylabel('実行時間 (ms)', fontsize=12)
ax3.set_title('3. パフォーマンス（7人ゲーム）', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}ms',
            ha='center', va='bottom', fontsize=9)

# 4. パフォーマンス（100人ゲーム、推定）
ax4 = axes[1, 1]
bars4 = ax4.bar(versions, performance_100_players, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax4.set_ylabel('実行時間 (ms)', fontsize=12)
ax4.set_title('4. パフォーマンス（100人ゲーム、推定）', fontsize=13, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='実用閾値: 100ms')
for i, bar in enumerate(bars4):
    height = bar.get_height()
    if height < 100:
        color = 'green'
        weight = 'bold'
    else:
        color = 'black'
        weight = 'normal'
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}ms',
            ha='center', va='bottom', fontsize=9, color=color, fontweight=weight)
ax4.legend()

plt.tight_layout()
plt.savefig('werewolf_version_comparison.png', dpi=300, bbox_inches='tight')
print("✅ グラフを保存しました: werewolf_version_comparison.png")

# 進化系統図
fig2, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# タイトル
ax.text(5, 9.5, '人狼ゲーム実装 - 進化系統樹', 
        ha='center', fontsize=16, fontweight='bold')

# v6
ax.text(2, 8, 'v6 (2024前期)', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.7))
ax.text(2, 7.3, '階層的認知モデル', ha='center', fontsize=9)
ax.text(2, 6.9, '戦略DB + 思考フェーズ', ha='center', fontsize=8)
ax.text(2, 6.5, '理論整合性: 60%', ha='center', fontsize=8, style='italic')

# 矢印とラベル
ax.annotate('', xy=(5, 7.5), xytext=(3, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(4, 7.7, '四層構造化', ha='center', fontsize=9, color='red')

# v7
ax.text(5, 8, 'v7 (2024後期)', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.7))
ax.text(5, 7.3, '四層構造の葛藤', ha='center', fontsize=9)
ax.text(5, 6.9, 'R値ベース跳躍判定', ha='center', fontsize=8)
ax.text(5, 6.5, '理論整合性: 75%', ha='center', fontsize=8, style='italic')

# 矢印とラベル
ax.annotate('', xy=(8, 7.5), xytext=(6, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(7, 7.7, '層別E・κ', ha='center', fontsize=9, color='red')

# v8
ax.text(8, 8, 'v8 (2025初頭)', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#45B7D1', alpha=0.7))
ax.text(8, 7.3, '層別E・κ独立管理', ha='center', fontsize=9)
ax.text(8, 6.9, '構造的影響力計算', ha='center', fontsize=8)
ax.text(8, 6.5, '理論整合性: 85%', ha='center', fontsize=8, style='italic')

# 矢印とラベル（v8 → v8.5）
ax.annotate('', xy=(5, 5.5), xytext=(5, 6.3),
            arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax.text(5.7, 5.9, '主観的社会\n+ Dual Engine', ha='left', fontsize=9, 
        color='green', fontweight='bold')

# v8.5
ax.text(5, 5, 'v8.5 (2025/11/7) ★', ha='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#96CEB4', alpha=0.9))
ax.text(5, 4.2, '主観的社会システム', ha='center', fontsize=10)
ax.text(5, 3.8, 'Reference実装 + Nano実装', ha='center', fontsize=9)
ax.text(5, 3.4, '理論整合性: 100%', ha='center', fontsize=9, 
        style='italic', color='red', fontweight='bold')

# 分岐: Reference vs Nano
ax.annotate('', xy=(2, 3), xytext=(4.2, 3.6),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue', linestyle='--'))
ax.text(2, 2.5, 'Reference実装', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(2, 2.0, '理論完全性', ha='center', fontsize=8)
ax.text(2, 1.6, '詳細分析', ha='center', fontsize=8)

ax.annotate('', xy=(8, 3), xytext=(5.8, 3.6),
            arrowprops=dict(arrowstyle='->', lw=2, color='orange', linestyle='--'))
ax.text(8, 2.5, 'Nano実装', ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(8, 2.0, 'パフォーマンス', ha='center', fontsize=8)
ax.text(8, 1.6, '大規模実行', ha='center', fontsize=8)

# 特徴的課題
ax.text(1, 0.8, '【v6の課題】単一Eプール', ha='left', fontsize=8, color='red')
ax.text(1, 0.4, '【v7の課題】E・κ単一値', ha='left', fontsize=8, color='red')
ax.text(5, 0.8, '【v8の課題】神の視点', ha='center', fontsize=8, color='red')
ax.text(5, 0.4, '【v8.5】全て解決！', ha='center', fontsize=9, 
        color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('werewolf_evolution_tree.png', dpi=300, bbox_inches='tight')
print("✅ 系統樹を保存しました: werewolf_evolution_tree.png")

# 機能比較マトリクス
fig3, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

features = [
    '多層構造',
    '層別エネルギー',
    '層別κ学習',
    '層別跳躍',
    '主観的社会圧力',
    '非線形層間転送',
    '神の視点廃止',
    'パフォーマンス最適化',
    '理論的整合性'
]

data = [
    ['❌ 単一E', '✅ 四層(圧力)', '✅ 四層(E・κ)', '✅ 四層完全'],
    ['❌', '❌', '✅', '✅'],
    ['❌', '❌', '✅', '✅'],
    ['❌', '⚠️ 部分的', '✅', '✅'],
    ['❌', '❌', '❌', '✅'],
    ['❌', '❌', '❌', '✅'],
    ['❌', '❌', '❌', '✅'],
    ['❌', '❌', '❌', '✅'],
    ['60%', '75%', '85%', '100%']
]

columns = ['v6', 'v7', 'v8', 'v8.5']

table = ax.table(cellText=data, rowLabels=features, colLabels=columns,
                cellLoc='center', loc='center',
                colWidths=[0.2, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# ヘッダーのスタイル
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold')

# 行ラベルのスタイル
for i in range(len(features)):
    table[(i+1, -1)].set_facecolor('#E8E8E8')

# v8.5列を強調
for i in range(len(features) + 1):
    table[(i, 3)].set_facecolor('#96CEB4' if i == 0 else '#E0F2E9')
    if i > 0:
        table[(i, 3)].set_text_props(weight='bold')

ax.set_title('機能比較マトリクス', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('werewolf_feature_matrix.png', dpi=300, bbox_inches='tight')
print("✅ 機能マトリクスを保存しました: werewolf_feature_matrix.png")

plt.show()

print("\n" + "="*60)
print("全ての可視化が完了しました！")
print("="*60)
print("\n生成されたファイル:")
print("  1. werewolf_version_comparison.png - 総合比較")
print("  2. werewolf_evolution_tree.png - 進化系統樹")
print("  3. werewolf_feature_matrix.png - 機能マトリクス")
