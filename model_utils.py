import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "./saved_model"
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


def get_attributions(text: str) -> list[dict]:
    """Gradient × Input によるトークンレベルのattribution。
    score > 0 → Positive方向に寄与、score < 0 → Negative方向に寄与。
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

    # embeddingsをdetachしてrequires_grad=Trueにすることで
    # embeddingレイヤーの重みには勾配を流さず、このテンソル自体の勾配だけ取得する
    embeddings = model.roberta.embeddings(inputs["input_ids"])
    embeddings = embeddings.detach().requires_grad_(True)

    outputs = model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"])
    # positive - negative の差分スコアで符号付きattributionを得る
    score = outputs.logits[0, 1] - outputs.logits[0, 0]
    score.backward()

    # Gradient × Input（隠れ次元方向に和をとって各トークンのスカラー値にする）
    attr = (embeddings.grad * embeddings).sum(dim=-1)[0].detach().cpu().numpy()
    model.zero_grad()

    # [-1, 1] に正規化
    max_abs = float(np.abs(attr).max())
    attr_list = (attr / max_abs).tolist() if max_abs > 0 else [0.0] * len(attr)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    # サブワードトークンを単語単位にマージ（RoBERTa は空白を Ġ で表現）
    words: list[dict] = []
    pending: tuple | None = None  # (word_str, scores, space_before)

    for tok, s in zip(tokens, attr_list):
        if tok in {"<s>", "</s>", "<pad>"}:
            continue
        if tok.startswith("Ġ") or pending is None:
            if pending:
                w, sc, sb = pending
                words.append({"word": w, "score": round(sum(sc) / len(sc), 3), "space_before": sb})
            pending = (tok.lstrip("Ġ"), [s], bool(words))
        else:
            w, sc, sb = pending
            pending = (w + tok, sc + [s], sb)

    if pending:
        w, sc, sb = pending
        words.append({"word": w, "score": round(sum(sc) / len(sc), 3), "space_before": sb})

    return words
