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
- 

---
  ````markdown
# Local Tone Classifier Chrome Extension 💬

A local sentiment analysis Chrome extension built with **FastAPI** and **PyTorch**. When a user selects text, the extension analyzes the tone (happy, angry, uncertain, calm) and displays the result as a colored, animated bubble, also supporting text label insertion.

## ✨ Features Overview

* **Real-time Bubble Preview** 💭
* **Dual Analysis Modes** (Keyword + AI Model) 🧠
* **FastAPI** Local Server Invocation 🚀
* **Privacy-Friendly**, Zero Network Requests 🔒
* **Color-Animated Effects** for Enhanced Emotion Visualization 🎨
* Supports Automatic `[HAPPY]` Text Label Insertion 🏷️


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

## 3️⃣ Model Logic

Input text is converted to vector form using **Hashed Bag-of-Words (BoW)**. Each token $\rightarrow$ numeric index (hashed), accumulated counts $\rightarrow$ normalized tensor. Model performs linear mapping and softmax classification:

```python
x = fe.encode(["this is awesome!!"])
probs = torch.softmax(model(x), dim=-1)
````

## 🔄 Data Flow

User selects text $\downarrow$
`popup.js` $\rightarrow$ FastAPI (`/predict`) $\downarrow$
PyTorch model inference (BoW $\rightarrow$ Linear $\rightarrow$ Softmax) $\downarrow$
Tone classification result (JSON) $\downarrow$
Displayed as animated bubble in popup

## 🧮 Example Predictions

| Input | Predicted Tone | Confidence |
| :--- | :--- | :--- |
| “Let's go\!\! This is awesome lol” | happy | 0.94 |
| “WHERE WERE YOU??” | angry | 0.78 |
| “Maybe later, not sure” | uncertain | 0.83 |
| “On my way.” | calm | 0.67 |

## 💡 Why It Works

  * **FastAPI** bridges the web UI with the local ML model 🌉
  * **PyTorch** handles tone classification with linear softmax layers 💡
  * **Chrome APIs** enable webpage-level interactivity 🌐
  * **Local-only execution** $\rightarrow$ zero data leakage 🛡️

## ⚙️ Run Locally

1.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the API server:
    ```bash
    python api.py
    ```
4.  Then visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the API.

## 📂 Project Structure

```
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
```

## 👤 Author

**Murray Chen**
Olin College of Engineering $\cdot$ Class of 2029
Focus: AI Systems $\cdot$ GPU Computing $\cdot$ Human-AI Interaction
