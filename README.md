# Traffic Sign Recognizer Using CNN

**Course:** 21CSE216P – Linux and Container Technologies  
**Project:** Deep Learning–based Traffic Sign Recognition System  
**Tech Stack:** Python · Flask · TensorFlow/Keras · Podman · nginx

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Prerequisites](#prerequisites)
5. [Setup & Installation](#setup--installation)
6. [Training the Model](#training-the-model)
7. [Running with Podman](#running-with-podman)
8. [API Reference](#api-reference)
9. [Dataset](#dataset)
10. [How It Works](#how-it-works)

---

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify traffic signs from the [GTSRB (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/) dataset. The system is fully containerised using **Podman** and **podman-compose**, exposing a REST API (Flask/Gunicorn) consumed by a browser-based frontend served through nginx.

### Key Features

- CNN trained on 10 GTSRB traffic-sign classes
- REST API: `/predict`, `/history`, `/model-info`, `/health`
- SQLite persistence for prediction history (stored in a named Podman volume)
- Single-command deployment via `podman-compose`

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Podman Network: sign-net                │
│                                                         │
│  ┌──────────────────┐       ┌──────────────────────┐   │
│  │  frontend        │       │  backend             │   │
│  │  nginx:alpine    │──────▶│  python:3.10-slim    │   │
│  │  port 8080:80    │  API  │  Flask + Gunicorn    │   │
│  │                  │       │  port 5000           │   │
│  └──────────────────┘       └──────────┬───────────┘   │
│                                        │               │
│                               ┌────────▼────────┐      │
│                               │  sign-db volume │      │
│                               │  (SQLite DB)    │      │
│                               └─────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
Traffic-Sign-Recognizer/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── train.py            # CNN training script
│   ├── requirements.txt    # Python dependencies
│   ├── Containerfile       # Backend container image
│   ├── model/              # Saved model files (after training)
│   └── uploads/            # Temporary upload directory
├── frontend/
│   ├── index.html          # Main UI page
│   ├── style.css           # Stylesheet
│   ├── script.js           # Frontend logic
│   ├── nginx.conf          # nginx reverse-proxy config
│   └── Containerfile       # Frontend container image
├── dataset/                # GTSRB dataset (downloaded by train.py)
├── podman-compose.yml      # Multi-container orchestration
├── README.md
└── .gitignore
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://python.org) |
| Podman | ≥ 4.x | [podman.io](https://podman.io) |
| podman-compose | ≥ 1.x | `pip install podman-compose` |

---

## Setup & Installation

### 1. Clone / navigate to the project

```bash
cd D:\Traffic-Sign-Recognizer
```

### 2. Create a Python virtual environment (for training)

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r backend/requirements.txt
```

---

## Training the Model

The training script downloads the GTSRB dataset automatically (~300 MB).

```bash
cd backend
python train.py
```

This will:
1. Download & extract GTSRB to `../dataset/`
2. Load images for 10 classes (speed-limit signs + "No passing")
3. Train a CNN with early stopping (up to 10 epochs)
4. Save the model to `backend/model/traffic_sign_model.h5`
5. Save class names to `backend/model/class_names.json`
6. Save test accuracy to `backend/model/accuracy.txt`

> **Note:** Training typically takes 5–15 minutes depending on hardware. GPU acceleration is used automatically if available.

### Model Architecture

```
Conv2D(32) → BatchNorm → MaxPool
Conv2D(64) → BatchNorm → MaxPool
Conv2D(128) → BatchNorm
Flatten → Dense(256) → Dropout(0.5) → Dense(10, softmax)
```

Input: 32×32 RGB images  
Output: Softmax probabilities over 10 classes

---

## Running with Podman

### Build and start all services

```bash
podman-compose up --build -d
```

### Check service status

```bash
podman-compose ps
podman logs traffic-backend
podman logs traffic-frontend
```

### Open the app

Navigate to **http://localhost:8080** in your browser.

### Stop services

```bash
podman-compose down
```

### Stop and remove volumes (clears prediction history)

```bash
podman-compose down -v
```

---

## API Reference

Base URL: `http://localhost:8080/api` (proxied by nginx → backend:5000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check; returns model load status |
| GET | `/model-info` | Returns accuracy, class count, model status |
| POST | `/predict` | Upload an image; returns class + confidence |
| GET | `/history` | Last 20 predictions from DB |
| DELETE | `/history` | Clear all prediction history |

### POST /predict — Example

```bash
curl -X POST http://localhost:8080/api/predict \
  -F "file=@my_sign.jpg"
```

**Response:**
```json
{
  "class_id": 1,
  "class_name": "Speed limit (30km/h)",
  "confidence": 97.43,
  "timestamp": "2026-11-16T20:00:00.000000"
}
```

---

## Dataset

**GTSRB – German Traffic Sign Recognition Benchmark**  
Source: https://benchmark.ini.rub.de/

This project uses the first 10 classes:

| ID | Sign |
|----|------|
| 0 | Speed limit (20km/h) |
| 1 | Speed limit (30km/h) |
| 2 | Speed limit (50km/h) |
| 3 | Speed limit (60km/h) |
| 4 | Speed limit (70km/h) |
| 5 | Speed limit (80km/h) |
| 6 | End of speed limit (80km/h) |
| 7 | Speed limit (100km/h) |
| 8 | Speed limit (120km/h) |
| 9 | No passing |

---

## How It Works

1. **Training (`train.py`):** Loads PPM images from GTSRB, resizes to 32×32, normalises to [0,1], trains a CNN, and saves the model weights.
2. **Backend (`app.py`):** Flask app loaded with the saved Keras model. On `/predict`, the uploaded image is preprocessed identically to training and fed through the network. Prediction metadata is persisted to SQLite.
3. **Frontend:** Static HTML/CSS/JS page served by nginx. Communicates with the backend via `/api/` proxy path.
4. **Containers:** Backend and frontend run in separate Podman containers connected via a user-defined bridge network (`sign-net`). The SQLite database is stored in a named volume (`sign-db`) so it survives container restarts.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not loaded` error | Run `python train.py` first to generate model files |
| Download fails in `train.py` | Manually download GTSRB zip to `dataset/` |
| Port 8080 in use | Edit `podman-compose.yml` → change `"8080:80"` |
| `libgomp1` error in container | Already installed via `Containerfile` `apt-get` step |
