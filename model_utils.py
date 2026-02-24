import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "textattack/roberta-base-SST-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# モデルによってラベル名が LABEL_0/LABEL_1 になる場合があるため正規化する
# SST-2 の慣例: 0 = negative, 1 = positive
_LABEL_NORM = {
    "label_0": "negative",
    "label_1": "positive",
}

# アプリ起動時に1回だけロード
print(f"モデルをロード中... (device: {DEVICE})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("モデルのロード完了")


def predict(text: str, debug: bool = False) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label_id = probs.argmax().item()
    raw_label = model.config.id2label[label_id].lower()
    label = _LABEL_NORM.get(raw_label, raw_label)

    result = {
        "label": label,
        "confidence": round(probs[label_id].item() * 100, 1),
    }

    if debug:
        token_ids = inputs["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        result["debug"] = {
            "device": DEVICE,
            "token_count": len(tokens),
            "tokens": tokens,
            "logits": {
                _LABEL_NORM.get(model.config.id2label[i].lower(), model.config.id2label[i].lower()): round(logits[0][i].item(), 4)
                for i in range(len(logits[0]))
            },
            "probs_all": {
                _LABEL_NORM.get(model.config.id2label[i].lower(), model.config.id2label[i].lower()): round(probs[i].item() * 100, 1)
                for i in range(len(probs))
            },
        }

    return result
