# Deep Multimodal Authentication Framework (DMAF)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)

An official PyTorch implementation of the **Deep Multimodal Authentication Framework (DMAF)** for detecting malicious QR codes, payment redirections, and phishing threats in mobile payment systems.

---

## 📖 Architecture Overview

DMAF integrates five complementary security modalities into a unified representation via cross-modal attention fusion, explainable risk assessment, and continual learning:

```
                      ┌───────────────────────────────────────────────┐
                      │              Multimodal Input D               │
                      └──────────────────────┬────────────────────────┘
                                             │
      ┌───────────────┬──────────────────────┼───────────────────────┬───────────────┐
      │               │                      │                       │               │
      ▼               ▼                      ▼                       ▼               ▼
 ┌─────────┐    ┌──────────┐           ┌───────────┐           ┌───────────┐   ┌───────────┐
 │ Visual  │    │ Decoded  │           │    URL    │           │ Mobile    │   │ Transact. │
 │ Image   │    │ Payload  │           │ Semantics │           │ Behavior  │   │ Metadata  │
 │  (I_p)  │    │  (T_p)   │           │    (U)    │           │   (B_p)   │   │   (M_p)   │
 └────┬────┘    └────┬─────┘           └─────┬─────┘           └─────┬─────┘   └─────┬─────┘
      │              │                       │                       │               │
  Lightweight     Contextual              Contextual             Temporal         Relational
   ViT (f_v)     Lang. (f_t)             Lang. (f_u)            Attn. (f_b)       MLP (f_m)
      │              │                       │                       │               │
      ▼              ▼                       ▼                       ▼               ▼
     F_v            F_t                     F_u                     F_b             F_m
      └──────────────┴───────────────────────┼───────────────────────┴───────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │   Cross-Modal Attention Fusion (Eqs. 13-16)   │
                     │          Mutual Attention A_ij & W_i          │
                     └───────────────────────┬───────────────────────┘
                                             │
                                      Fused Latent Z
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
                     ▼                                               ▼
         ┌────────────────────────┐                    ┌────────────────────────┐
         │ Deep Threat Classifier │                    │ Explainability (Grad)  │
         │   Y = Softmax(Wc Z+b)  │                    │    E_i = dY / dZ_i     │
         └───────────┬────────────┘                    └────────────────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │  Continual Engine &    │
         │ Decision Rule (tau)    │
         └────────────────────────┘
```

### Core Formulations:
1. **Preprocessing & Alignment**:
   $$I_p = \Psi(I), \quad T_p = \Phi(T), \quad B_p = \frac{B - \mu_B}{\sigma_B}, \quad M_p = \frac{M - \mu_M}{\sigma_M}$$
2. **Cross-Modal Attention**:
   $$A_{ij} = \frac{\exp(F_i F_j^T / \sqrt{d})}{\sum_k \exp(F_i F_k^T / \sqrt{d})}, \quad W_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$$
3. **Adaptive Fusion Embedding**:
   $$Z = \sum_{i=1}^5 W_i F_i$$
4. **Explainable Saliency & Decision**:
   $$E_i = \frac{\partial Y}{\partial Z_i}, \quad D = \begin{cases} \text{Legitimate}, & C \ge \tau \text{ and } \hat{y} = 0 \\ \text{Malicious}, & \text{otherwise} \end{cases}$$

---

## 🚀 Quickstart

### 1. Requirements & Setup
```bash
pip install -r requirements.txt
```

### 2. Verify Pipeline (End-to-End Demo)
Run the self-contained verification suite that exercises all 5 phases:
```bash
python demo.py
```

### 3. Model Training
Train the full DMAF network (automatically falls back to synthetic benchmark if no dataset path is specified):
```bash
python train.py --epochs 25 --batch_size 32 --lr 0.0001
```

To train on the Kaggle *Benign and Malicious QR Codes* dataset (200,000 samples):
```bash
python train.py --data_dir /path/to/kaggle_qr_dataset --epochs 50 --batch_size 64
```

### 4. Comprehensive Evaluation
Evaluate against the benchmark metrics (Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR, MCC, Latency, Throughput):
```bash
python evaluate.py --checkpoint checkpoints/dmaf_best.pt
```

### 5. Web UI Portal & REST API (Flask)
Launch the interactive web portal:
```bash
python app.py
```
Open your browser to: **`http://127.0.0.1:5000`**
- Upload any QR code image or paste a payment link.
- One-click presets for authentic UPI, phishing sites, and malware links.
- Interactive visualization of cross-modal attention weights and XAI risk breakdown.

REST API Example:
```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"payload": "http://phishing-site.xyz/login"}'
```

### 6. Real-Time Forensic CLI Inference
Inspect a single QR code or URL transaction via command-line:
```bash
python inference.py --image test_qr.png --payload "https://secure-checkout.merchant.com/order?id=8831" --tau 0.85
```

---

## 📊 Experimental Results (from Paper)

| Metric | CNN QR Detector | ResNet50 + URL | ViT | Multimodal CNN-BiLSTM | Hybrid Attention | **DMAF (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy (%)** | 94.12 | 95.36 | 96.18 | 97.24 | 98.01 | **99.18** |
| **F1-Score (%)** | 93.14 | 94.51 | 95.52 | 96.60 | 97.53 | **98.94** |
| **ROC-AUC** | 0.958 | 0.969 | 0.978 | 0.985 | 0.991 | **0.997** |
| **FPR (%)** | 5.74 | 4.91 | 3.94 | 2.86 | 2.11 | **1.14** |
| **FNR (%)** | 7.12 | 5.83 | 4.72 | 3.63 | 2.65 | **1.06** |
| **Latency (ms)** | 41 | 38 | 36 | 33 | 30 | **24 (8.9ms inf)** |
| **Throughput (QR/s)** | 68 | 75 | 78 | 86 | 92 | **112** |

---

## 📁 Repository Structure

```
DMAF/
├── dmaf/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters, dimensions, thresholds
│   ├── preprocessing.py          # CLAHE, deskew, tokenization, standardization
│   ├── dataset.py                # Dataset loader & synthetic benchmark
│   ├── explainability.py         # Gradient attribution E_i & modality weights
│   ├── continual_learning.py     # EWC incremental update & threshold decision
│   └── models/
│       ├── __init__.py
│       ├── vision_encoder.py     # Lightweight Vision Transformer (f_v)
│       ├── text_encoder.py       # Contextual sequence encoder (f_t, f_u)
│       ├── behavioral_encoder.py # Temporal attention telemetry encoder (f_b)
│       ├── transaction_encoder.py# Metadata dense projection (f_m)
│       ├── attention_fusion.py   # Cross-modal mutual attention (Eqs. 13-16)
│       └── dmaf_network.py       # End-to-end network architecture
├── train.py                      # Training pipeline with Adam & regularized loss
├── evaluate.py                   # Benchmark evaluation reproducing paper tables
├── inference.py                  # Real-time authentication inspector with XAI
├── demo.py                       # Self-contained end-to-end verification demo
├── requirements.txt              # Environment dependencies
└── README.md                     # Documentation
```
