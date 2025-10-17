
import re, os, random
import torch as t
import torch.nn as nn
from typing import List, Dict

LABELS = ["happy", "angry", "uncertain", "calm"]
LABEL2ID = {k:i for i,k in enumerate(LABELS)}
ID2LABEL = {i:k for k,i in LABEL2ID.items()}

SEED_KWS = {
    "happy":      ["yay","great","awesome","love","喜欢","太棒了","开心","哈哈","lol","lmao","nice"],
    "angry":      ["angry","mad","wtf","damn","shit","生气","烦","妈的","tmd","为什么不"],
    "uncertain":  ["maybe","perhaps","not sure","不知道","可能","吗","maybe?","idk"],
}

WORD_RE = re.compile(r"[\w#@']+|[\u4e00-\u9fff]|[^\s]")
def tokenize(text: str):
    return WORD_RE.findall(text.lower())

class HashedBoW:
    def __init__(self, num_bins: int = 2048):
        self.num_bins = num_bins
    def encode(self, texts: List[str]) -> t.Tensor:
        X = t.zeros((len(texts), self.num_bins), dtype=t.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = hash(tok) % self.num_bins
                X[i, h] += 1.0
        X = t.nn.functional.normalize(X, p=2, dim=1)
        return X

class TinyBoWClassifier(nn.Module):
    def __init__(self, num_bins=2048, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(num_bins, num_classes)
    def forward(self, x):
        return self.fc(x)

def synthetic_dataset(n_per_class=120) -> Dict[str, List[str]]:
    data = {k: [] for k in LABELS}
    for label, kws in SEED_KWS.items():
        for _ in range(n_per_class):
            k = random.choice(kws)
            ctx = random.choice(["this is","i feel","感觉","其实","maybe","真的","why","bro","lol","唉"])
            punct = random.choice(["!","!!","…",".","???",""])
            noise_kw = random.choice(sum(SEED_KWS.values(), []))
            sent = f"{ctx} {k} {noise_kw}{punct}"
            data[label].append(sent.strip())
    for _ in range(n_per_class):
        sent = random.choice(["on my way.","got it.","noted.","就这样吧。","好的。","okay","嗯。","see you later.","OK"])
        data["calm"].append(sent)
    return data

def make_tensors(data: Dict[str, List[str]], fe: HashedBoW):
    Xs, ys = [], []
    for label, sents in data.items():
        y = LABEL2ID[label]
        Xs.extend(sents)
        ys.extend([y]*len(sents))
    X = fe.encode(Xs)
    y = t.tensor(ys, dtype=t.long)
    return X, y

def train_or_load(model_path="models/model.pt", bins=2048, epochs=6, lr=3e-2, device="cpu"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    fe = HashedBoW(num_bins=bins)
    model = TinyBoWClassifier(num_bins=bins, num_classes=len(LABELS)).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(t.load(model_path, map_location=device))
        model.eval()
        return model, fe
    data = synthetic_dataset(n_per_class=120)
    X, y = make_tensors(data, fe)
    model.train()
    opt = t.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    idx = t.randperm(len(y))
    X, y = X[idx], y[idx]
    for _ in range(epochs):
        logits = model(X.to(device))
        loss = loss_fn(logits, y.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    t.save(model.state_dict(), model_path)
    return model, fe

@t.inference_mode()
def predict(model: nn.Module, fe: HashedBoW, texts: List[str], device="cpu"):
    X = fe.encode(texts).to(device)
    logits = model(X)
    probs = t.softmax(logits, dim=-1)
    ids = t.argmax(probs, dim=-1).tolist()
    out = []
    for i, p in zip(ids, probs.tolist()):
        out.append({"label": ID2LABEL[i], "probs": p})
    return out

