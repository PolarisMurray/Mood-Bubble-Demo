# 🌈 PBJ: Local AI Mood Bubble 💬

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange.svg?logo=pytorch)](https://pytorch.org/)
[![Chrome Extension](https://img.shields.io/badge/Chrome_Extension-MV3-yellow.svg?logo=googlechrome)](https://developer.chrome.com/docs/extensions/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

[English](#english-version) | [中文](#中文版本) |**[한국어](#한국어-버전)** 🇰🇷

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


#### 3️⃣ Model Logic
- Input text is converted to vector form using Hashed Bag-of-Words (BoW).
- Each token → numeric index (hashed), accumulated counts → normalized tensor.
- Model performs linear mapping and softmax classification:



#### 🔄 Data Flow
- User selects text
  ↓
- popup.js → FastAPI (/predict)
  ↓
- PyTorch model inference (BoW → Linear → Softmax)
  ↓
- Tone classification result (JSON)
  ↓
- Displayed as animated bubble in popup

#### 🧮 Example Predictions
- Input	Predicted Tone	Confidence
- “Let's go!! This is awesome lol”	happy	0.94
- “WHERE WERE YOU??”	angry	0.78
- “Maybe later, not sure”	uncertain	0.83
- “On my way.”	calm	0.67

#### 💡 Why It Works
- FastAPI bridges the web UI with the local ML model 🌉
- PyTorch handles tone classification with linear softmax layers 💡
- Chrome APIs enable webpage-level interactivity 🌐
- Local-only execution → zero data leakage 🛡️

#### ⚙️ Run Locally
- Create and activate a virtual environment:
- 'python -m venv .venv'
- 'source .venv/bin/activate'
- Install dependencies:
- 'pip install -r requirements.txt'
- Run the API server:
- 'python api.py'
- Visit the interactive docs:
- 'http://127.0.0.1:8000/docs'


#### 📂 Project Structure
- pbj/
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
- Murray Chen
- Olin College of Engineering · Class of 2029
- Focus: Computer Science


## 한국어 버전 (Korean Version) 🇰🇷

### 🚀 기능
- **팝업 인터페이스를 통한 실시간 메시지 미리보기 🎨
- **이중 모드 감정 분석 🧠
 - *휴리스틱 모드: 키워드 / 이모지 / 구두점 분석
 - *AI 모드: PyTorch 분류기 (FastAPI를 통해)
- **감정을 반영하는 애니메이션 색상 버블 ✨
- **로컬 추론 전용 — 모든 계산은 기기 내에서 처리됩니다 🔒
- **텍스트 주입 기능 ([HAPPY] your text) 🏷️

---

### 🧠 아키텍처 개요
#### 1️⃣ Chrome 확장 프론트엔드 🌐
- 사용자가 선택한 텍스트나 입력을 수집
- JavaScript에서 휴리스틱 감정 점수 계산 수행
- 선택적으로 로컬 API(127.0.0.1:8000/predict) 호출
- 톤 버블을 표시하고 태그된 텍스트 삽입
#### 2️⃣ FastAPI + PyTorch 백엔드 🚀
- 'api.py': /predict 엔드포인트 정의
- 'ml.py': 소형 신경망 분류기 정의 및 학습

#### 3️⃣ 모델 로직
- 입력 텍스트는 Hashed Bag-of-Words (BoW)를 사용하여 벡터 형태로 변환됩니다.
- 각 토큰 → 숫자 인덱스로 해시되고, 누적 카운트 → 정규화된 텐서로 변환됩니다.
- 모델은 선형 매핑 및 소프트맥스 분류를 수행합니다.


#### 🔄 데이터 흐름
- 사용자가 텍스트 선택
↓
- popup.js → FastAPI (/predict)
↓
- PyTorch 모델 추론 (BoW → Linear → Softmax)
↓
- 감정 분류 결과 (JSON)
↓
- 팝업에서 애니메이션 버블로 표시

#### 🧮 예측 예시
- 입력 예측된 톤 신뢰도
- “Let's go!! This is awesome lol” 행복함 0.94
- “WHERE WERE YOU??” 분노 0.78
- “Maybe later, not sure” 불확실 0.83
- “On my way.” 차분함 0.67


#### 💡 작동 원리
- FastAPI는 웹 UI와 로컬 ML 모델을 연결합니다 🌉
- PyTorch는 선형 소프트맥스 계층을 통해 감정 분류를 수행합니다 💡
- Chrome API는 웹페이지 수준의 상호작용을 가능하게 합니다 🌐
- 로컬 실행만으로 데이터 유출이 전혀 없습니다 🛡️

#### ⚙️ 로컬 실행 방법
- 가상 환경 생성 및 활성화:
- 'python -m venv .venv'
- 'source .venv/bin/activate'
- 종속성 설치:
- 'pip install -r requirements.txt'
- API 서버 실행:
- 'python api.py'
- 인터랙티브 문서 방문:
- 'http://127.0.0.1:8000/docs'

#### 📂 프로젝트 구조
- pbj/
├── api.py # FastAPI 서버
├── ml.py # PyTorch 모델 + 인코딩
├── models/ # 학습된 가중치
├── popup.html # UI 레이아웃
├── popup.js # 감정 로직
├── styles.css # 버블 애니메이션
├── content.js # 웹페이지 텍스트 주입기
├── manifest.json # Chrome 확장 매니페스트
└── requirements.txt

#### 👤 작성자
머레이 첸 (Murray Chen)
올린 공과대학교 (Olin College of Engineering) · 2029학번
전공: 컴퓨터 과학 (Computer Science)


