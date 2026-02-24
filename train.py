import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "roberta-base"
SAVE_PATH = "./saved_model"

# 1. データセット読み込み (IMDB: pos/neg 各25000件)
print("データセットを読み込んでいます...")
dataset = load_dataset("imdb")

# 2. トークナイザーで前処理
print("トークナイズしています...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)


tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 3. モデル定義 (2クラス分類)
print("モデルを読み込んでいます...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# 4. 評価指標
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# 5. 学習設定
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
)

# 6. Trainer で学習
print("学習を開始します（GPU fine-tuning）...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)
trainer.train()

# 7. モデルとトークナイザーを保存
print(f"モデルを {SAVE_PATH} に保存しています...")
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"完了！モデルを {SAVE_PATH} に保存しました")
