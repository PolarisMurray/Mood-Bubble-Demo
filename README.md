# 🌈 PBJ: Local AI Mood Bubble 💬

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange.svg?logo=pytorch)](https://pytorch.org/)
[![Chrome Extension](https://img.shields.io/badge/Chrome_Extension-MV3-yellow.svg?logo=googlechrome)](https://developer.chrome.com/docs/extensions/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

[English](#english-version) | [中文](#中文版本)

---

## English Version

A privacy-first Chrome extension that analyzes emotional tone of selected text using a **local PyTorch model** served through **FastAPI**. It visualizes tone as a **color / animation bubble** 💭, and allows users to insert tagged text (e.g., `[HAPPY] your text`) into any active input field.

---

### 🚀 Features

- **Popup interface** with live message preview 🎨  
- **Dual-mode tone analysis** 🧠  
  - *Heuristic mode*: keyword / emoji / punctuation analysis  
  - *AI mode*: PyTorch classifier (via FastAPI)  
- **Animated color bubbles** reflecting tone ✨  
- **Local inference only** — all computations stay on your device 🔒  
- **Text injection** into web inputs (`[HAPPY] your text`) 🏷️

---

### 🧠 Architecture Overview

#### 1️⃣ Chrome Extension Frontend 🌐
- Collects user-selected text or input
- Runs heuristic tone scoring in JavaScript
- Optionally calls the local API (`127.0.0.1:8000/predict`)
- Displays the tone bubble & inserts tagged text

#### 2️⃣ FastAPI + PyTorch Backend 🚀
- `api.py`: defines `/predict` endpoint
- `ml.py`: defines and trains a small neural classifier

```python
import torch.nn as nn

class ToneModel(nn.Module):
    def __init__(self, in_dim, out_dim=4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)


#### 3️⃣ Model Logic
Input text is converted to vector form using Hashed Bag-of-Words (BoW).
Each token → numeric index (hashed), accumulated counts → normalized tensor.
Model performs linear mapping and softmax classification:



#### 🔄 Data Flow
User selects text
  ↓
popup.js → FastAPI (/predict)
  ↓
PyTorch model inference (BoW → Linear → Softmax)
  ↓
Tone classification result (JSON)
  ↓
Displayed as animated bubble in popup

#### 🧮 Example Predictions
Input	Predicted Tone	Confidence
“Let's go!! This is awesome lol”	happy	0.94
“WHERE WERE YOU??”	angry	0.78
“Maybe later, not sure”	uncertain	0.83
“On my way.”	calm	0.67

#### 💡 Why It Works
FastAPI bridges the web UI with the local ML model 🌉
PyTorch handles tone classification with linear softmax layers 💡
Chrome APIs enable webpage-level interactivity 🌐
Local-only execution → zero data leakage 🛡️

#### ⚙️ Run Locally
Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Run the API server:
python api.py
Visit the interactive docs:
http://127.0.0.1:8000/docs


#### 📂 Project Structure
pbj/
├── api.py           # FastAPI server
├── ml.py            # PyTorch model + encoding
├── models/          # Trained weights
├── popup.html       # UI layout
├── popup.js         # Tone logic
├── styles.css       # Bubble animations
├── content.js       # Webpage text injector
├── manifest.json    # Chrome extension manifest
└── requirements.txt

#### 👤 Author
Murray Chen
Olin College of Engineering · Class of 2029
Focus: AI Systems · GPU Computing · Human-AI Interaction
