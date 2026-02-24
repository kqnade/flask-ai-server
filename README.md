# Flask 感情分析アプリ

英語テキストの感情（Positive / Negative）をリアルタイムで判定し、どの単語が予測に寄与したかをグラデーションで可視化する Web アプリ。

---

## やったこと

1. **RoBERTa のファインチューニング** — `roberta-base` を IMDB 映画レビューデータセットで追加学習
2. **Flask Web アプリの構築** — 学習済みモデルを使った感情分析 API とフロントエンド
3. **Attribution 可視化** — Gradient × Input 手法でどの単語が予測に寄与したかをグラデーション表示

---

## モデル

| 項目 | 内容 |
|------|------|
| ベースモデル | `roberta-base` |
| 学習データ | IMDB（train: 25,000件 / test: 25,000件） |
| タスク | 2クラス感情分類（Positive / Negative） |
| エポック数 | 3（best: epoch 2） |
| 最終精度（テスト） | **94.26%** |

学習は3エポック実施し、epoch 2 のチェックポイントが最良（eval_loss: 0.1612）。epoch 3 では eval_loss が悪化（0.2228）しており軽度の過学習が見られたため、`load_best_model_at_end=True` により epoch 2 のモデルが保存された。

---

## 実装

### ディレクトリ構成

```
.
├── main.py          # Flask アプリ本体
├── model_utils.py   # モデルのロード・推論・Attribution計算
├── train.py         # ファインチューニングスクリプト
├── saved_model/     # 学習済みモデル（safetensors）
├── checkpoints/     # 学習中チェックポイント
└── templates/
    └── index.html   # フロントエンド
```

### 推論（`model_utils.py`）

- アプリ起動時に `saved_model/` からモデルとトークナイザーをロード
- 入力テキストをトークナイズ → softmax → Positive / Negative ラベルと信頼度を返す

### Attribution 可視化

手法: **Gradient × Input**

```
attr_i = (∂score/∂e_i) × e_i  を隠れ次元方向に総和
score = logit[positive] - logit[negative]
```

- 各トークンの埋め込みベクトルに対して、positive - negative スコアの勾配を計算
- 勾配とembeddingの内積を取ることで、符号付きの寄与度スカラーを得る
- `[-1, 1]` に正規化し、正値（Positive 方向）→緑、負値（Negative 方向）→赤でハイライト
- RoBERTa のサブワードトークン（`Ġ` プレフィックス）は単語単位にマージして表示

Attribution はあくまで **「モデルが予測に使った根拠の可視化」** であり、各単語の独立した感情判定ではない。

---

## 苦手なパターン

### 1. 皮肉・婉曲表現

ポジティブな語彙でネガティブな評価を書くスタイルへの対応が難しい。

```
"It fails the way humans fail: completely, publicly, and with tremendous sincerity."
```

`sincerity` `tremendous` `human` といった単語が positive として拾われ、全体がネガティブな文章でも Positive と判定されることがある。IMDB の学習データでは語彙と感情が対応していることが多く、このような逆説的な表現を学習できていない。

### 2. 否定表現・二重否定

```
"This was not at all bad."    → 実際は Positive だが判定が安定しない
"I wouldn't say this is good." → good に引きずられて Positive 誤判定
```

`not` `never` `barely` などの否定語は単体では低スコアだが、後続の positive 語を打ち消す効果をモデルが学べていないケースがある。

### 3. 短文・文脈依存の表現

```
"It got better."  → 何が良くなったか文脈なしには判断困難
"I expected more." → 感情が曖昧
```

モデルはドキュメント全体で学習されているため、1文だけの短い入力では分布が訓練時と異なる。

### 4. 映画評論特有の専門的比較

```
"Better than his last film, but that's not saying much."
```

比較の文脈（前作より良い、でも前作がひどい）はモデルが明示的には学んでいない。

---

## 改善の方向性

### 短期的な対策

- **後処理による補正**: 否定語（not, never, barely, hardly 等）の直後に positive 語がある場合に信頼度を下げる簡易ルールを追加
- **入力前処理**: 極端に短い文は「判定できません」として弾く

### 中期的な対策

- **より多様なデータでの追加学習**: 皮肉・否定表現を含むデータセット（Stanford Sentiment Treebank の phrase-level など）でファインチューニングを追加
- **アンサンブル**: 複数モデルの出力を平均化して安定性を向上

### 根本的な対策

- **LLM ベースの判定に切り替え**: Claude や GPT 系モデルは文脈理解が高く、皮肉・否定への対応が大幅に改善する。ただしレイテンシとコストのトレードオフがある
- **Chain-of-Thought プロンプティング**: LLM に「なぜ positive/negative か」を説明させてから判定させることで精度が向上する傾向がある

---

## 起動方法

```bash
uv run main.py
# → http://127.0.0.1:5000
```
