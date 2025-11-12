# Werewolf Game - SSD Theory Implementation

人狼ゲームのSSD理論実装バリエーション

## ファイル一覧

### 🎮 実行可能デモ

#### **werewolf_ultimate_demo.py** (21.0KB)
究極デモ版 - 最も完成度の高い実装

- 完全なゲームフロー
- SSD理論による行動決定
- 詳細なログ出力

**推奨用途:** メインデモとして使用

#### **werewolf_extended_roles.py** (22.9KB)
拡張役職版

- 複数の役職実装
- 役職別の戦略
- 高度なゲームメカニクス

**推奨用途:** 役職システムの研究

#### **werewolf_narrator.py** (20.1KB)
ナレーター付き版

- ゲーム進行の実況
- 状況説明の充実
- わかりやすい出力

**推奨用途:** プレゼンテーション・教育

### 📊 可視化・分析

#### **werewolf_visualizer.py** (23.2KB)
ゲーム状態の可視化

- プレイヤー状態の図示
- SSD内部状態の表示
- グラフィカル出力

**推奨用途:** 内部状態の理解

#### **visualize_werewolf_comparison.py** (9.4KB)
比較分析ツール

- 複数ゲームの比較
- 統計分析
- 戦略評価

**推奨用途:** 性能評価・研究

## SSD理論の適用

### 三層構造

**BASE層（生存本能）**
- 脱落回避の圧力
- 疑われることへの恐怖

**CORE層（勝利欲求）**
- 陣営勝利への意欲
- 役職遂行の動機

**UPPER層（戦略的思考）**
- 状況分析
- 投票戦略
- 発言内容の選択

### 意味圧の源泉

1. **疑い圧力**: 他者から疑われている状態
2. **情報圧力**: 役職による確定情報
3. **時間圧力**: 残りターン数による焦り
4. **社会圧力**: 多数派・少数派の影響

## クイックスタート

```powershell
# 基本デモ実行
python werewolf_ultimate_demo.py

# 可視化付き実行
python werewolf_visualizer.py

# ナレーション付き実行
python werewolf_narrator.py
```

## 旧バージョン

開発過程のバージョンは `../archive/` に保管:

- werewolf_game_v8_5.py
- werewolf_game_v9.py
- werewolf_game_v10.py
- werewolf_game_v10_integrated.py
- werewolf_game_v10_2_pressure.py
- werewolf_game_v10_2_1_causal.py

## 理論的特徴

### E/κダイナミクス

人狼ゲームでは以下のE/κバランスが重要:

- **人狼側**: 隠蔽圧（E_BASE高）vs 勝利欲（E_CORE）
- **村人側**: 推理圧（E_UPPER高）vs 生存欲（E_BASE）
- **占い師**: 情報活用圧（E_CORE/UPPER高）

### 創発的行動パターン

外部ロジックなしで以下が創発:

1. 人狼の潜伏行動（BASE圧優勢時）
2. 占い師のCO判断（CORE圧優勢時）
3. 投票先の状況依存選択（UPPER圧優勢時）

---

*Note: これらの実装は純粋なE/κ創発原理に基づいています*
