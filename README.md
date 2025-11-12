# SSD Engine Demos

Structural Subjectivity Dynamics (SSD) Theory の実用デモ集

## 概要

このリポジトリは、[ssd-engine](https://github.com/HermannDegner/ssd-engine) ライブラリを使用した実用的なデモンストレーションを提供します。

## インストール

```bash
# SSDエンジンをインストール
pip install git+https://github.com/HermannDegner/ssd-engine.git

# 高速版（推奨）
pip install git+https://github.com/HermannDegner/ssd-engine.git#egg=ssd-engine[fast]
pip install matplotlib  # 可視化用
```

## デモ一覧

### 🎱 物理シミュレーション

#### Newton's Cradle（ニュートンのゆりかご）
```bash
python newtons_cradle/newtons_cradle_animated.py
```
- SSDエンジンで物理現象をシミュレート
- 重力・衝突をSSD意味圧として表現
- リアルタイムアニメーション

### 🎮 ゲーム＆意思決定

#### Roulette（ルーレット）
```bash
python games/roulette/roulette_ssd_pure.py
```
- 認知バイアス形成のデモ
- パターン学習と慣性形成
- 100回試行で学習過程を可視化

#### Blackjack（ブラックジャック）
```bash
python games/blackjack/blackjack_ssd_pure.py
```
- 戦略学習のデモ
- リスク判断の適応
- 勝率とκ慣性の変化を観察

#### APEX Survivor（生存ゲーム）
```bash
python games/apex_survivor/apex_survivor_ssd_pure_v4.py
```
- 極限状態での意思決定
- 生存圧・社会圧・認知負荷の統合
- エージェント間の複雑な相互作用

#### Werewolf（人狼ゲーム）
```bash
python games/werewolf/werewolf_ultimate_demo.py
```
- 社会的推論と欺瞞検出
- 概念形成と記憶構造
- 拡張役職（占い師、霊媒師、狂人）
- XAI（説明可能AI）対応

### 📊 社会分析

#### Social Crisis Analysis（社会危機分析）
```bash
python social_analysis/social_crisis_analysis.py
```
- 現代社会問題の分析
- 主観的社会圧力の計算
- 集団ダイナミクスのシミュレーション

## デモの特徴

- ✅ **理論実装**: SSD理論の完全実装
- ✅ **実用的**: 実際の問題に適用可能
- ✅ **高速**: Numba加速で4-5倍高速化
- ✅ **視覚化**: リアルタイム可視化対応
- ✅ **教育的**: コードから理論を学べる

## パフォーマンス

- Numbaなし: ~100,000 steps/sec
- Numba有効: ~470,000 steps/sec（4.38倍）

詳細: [ssd-engine BENCHMARK](https://github.com/HermannDegner/ssd-engine/blob/main/BENCHMARK.md)

## 理論背景

構造主観力学（SSD）は、認知・感情・行動を統一的に記述する理論フレームワークです。

**核心概念:**
- 意味圧 (p): 構造に作用する力
- Log-Alignment: 適応的入力処理
- 整合流 (j): Ohmの法則による応答
- 未処理圧 (E): エネルギー蓄積
- 慣性 (κ): 学習痕跡
- 跳躍 (Leap): 構造的変化

## ライセンス

MIT License

## リンク

- [SSD Engine Library](https://github.com/HermannDegner/ssd-engine)
- [Theory Documentation](https://github.com/HermannDegner/ssd-engine/tree/main/docs)
