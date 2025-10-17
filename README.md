# 🌈 PBJ: Local AI Mood Bubble  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange.svg?logo=pytorch)](https://pytorch.org/)
[![Chrome Extension](https://img.shields.io/badge/Chrome_Extension-MV3-yellow.svg?logo=googlechrome)](https://developer.chrome.com/docs/extensions/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

[English](#english-version) | [中文](#中文版本) | [日本語](#日本語バージョン) | [한국어](#한국어-버전)

---

## English Version
A privacy-first Chrome extension that analyzes emotional tone of selected text using a **local PyTorch model** served through **FastAPI**.  
It visualizes tone as a **color / animation bubble**, and allows users to insert tagged text (e.g., `[HAPPY] your text`) into any active input field.

---

### 🚀 Features
- **Popup interface** with live message preview  
- **Dual-mode tone analysis**
  - *Heuristic mode*: keyword / emoji / punctuation analysis  
  - *AI mode*: PyTorch classifier (via FastAPI)
- **Animated color bubbles** reflecting tone  
- **Local inference only** — all computations stay on your device  
- **Text injection** into web inputs (`[HAPPY] your text`)  

---

### 🧠 Architecture Overview

#### 1️⃣ Chrome Extension Frontend  
- Collects user-selected text or input  
- Runs heuristic tone scoring in JavaScript  
- Optionally calls the local API (`127.0.0.1:8000/predict`)  
- Displays the tone bubble & inserts tagged text  

#### 2️⃣ FastAPI + PyTorch Backend  
- `api.py`: defines `/predict` endpoint  
- `ml.py`: defines and trains a small neural classifier  
  ```python
  class ToneModel(nn.Module):
      def __init__(self, in_dim, out_dim=4):
          super().__init__()
          self.linear = nn.Linear(in_dim, out_dim)
      def forward(self, x):
          return self.linear(x)
