import os
import torch as t
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ml import train_or_load, predict

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join("models", "model.pt")
app = FastAPI(title="PBJ ML API", version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_fe = None

class Item(BaseModel):
    text: str

class Batch(BaseModel):
    texts: List[str]

@app.on_event("startup")
def _load():
    global _model, _fe
    _model, _fe = train_or_load(model_path=MODEL_PATH, device=DEVICE)

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}

@app.post("/predict")
def route_predict(item: Item):
    r = predict(_model, _fe, [item.text], device=DEVICE)[0]
    return r  # {label, probs}

@app.post("/predict-batch")
def route_predict_batch(batch: Batch):
    return predict(_model, _fe, batch.texts, device=DEVICE)

if __name__ == "__main__":
    import uvicorn
    os.makedirs("models", exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)

