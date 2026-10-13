"""
Automated Test Runner for Sample QR Codes.
Tests authentic merchant and malicious phishing QR code images through the trained DMAF pipeline.
"""

import os
import cv2
import torch
import numpy as np

from dmaf.config import ModelConfig
from dmaf.models.dmaf_network import DMAFNetwork
from dmaf.preprocessing import ImagePreprocessor, TextTokenizer
from dmaf.explainability import ExplainableRiskAssessor
from dmaf.continual_learning import ContinualLearningEngine


def test_qr(image_path: str, tau: float = 0.85):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/dmaf_best.pt"
    
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at {image_path}")
        return

    # 1. Load trained model
    config = ModelConfig()
    model = DMAFNetwork(config).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[!] Warning: Checkpoint not found, running with base weights.")

    # 2. Extract QR image and decode payload using OpenCV
    cv_img = cv2.imread(image_path)
    detector = cv2.QRCodeDetector()
    decoded_text, _, _ = detector.detectAndDecode(cv_img)
    
    from dmaf.payment_security import extract_multimodal_features, clean_qr_text
    decoded_text = clean_qr_text(decoded_text)
    if not decoded_text:
        decoded_text = "Unknown / Undecoded Payload"

    # 3. Preprocess modalities
    img_prep = ImagePreprocessor(target_size=(config.img_size, config.img_size))
    tokenizer = TextTokenizer(max_length=config.max_text_len, vocab_size=config.vocab_size)

    img_tensor = img_prep(cv_img)
    text_tokens = tokenizer.tokenize(decoded_text)
    url_tokens = tokenizer.tokenize(decoded_text)

    # Grounded Telemetry & Metadata
    behavior, meta = extract_multimodal_features(decoded_text, cv_img)

    t_img = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
    t_text = torch.from_numpy(text_tokens).unsqueeze(0).long().to(device)
    t_url = torch.from_numpy(url_tokens).unsqueeze(0).long().to(device)
    t_b = torch.from_numpy(behavior).unsqueeze(0).float().to(device)
    t_m = torch.from_numpy(meta).unsqueeze(0).float().to(device)

    # 4. Explainable Saliency & Attribution
    assessor = ExplainableRiskAssessor(model)
    explanation = assessor.explain(t_img, t_text, t_url, t_b, t_m)

    # 5. Authenticate via Continual Engine
    engine = ContinualLearningEngine(model, tau_threshold=tau)
    with torch.no_grad():
        out = model(t_img, t_text, t_url, t_b, t_m)
        probs = out["probabilities"]
        pred_class = int(torch.argmax(probs, dim=-1).item())
        decision, conf = engine.authenticate(probs, pred_class)

    # Print Clean Report
    print("=" * 65)
    print(f" TESTING FILE: {image_path}")
    print("=" * 65)
    print(f"Decoded Payload:       {decoded_text}")
    print(f"Detection Verdict:     {decision.upper()}")
    print(f"Classification:        {'Benign (Legitimate)' if pred_class == 0 else 'Malicious / Fraud'}")
    print(f"Confidence (C):        {conf * 100:.2f}% (Threshold tau: {tau})")
    print(f"Dominant Modality:     {explanation['dominant_risk_driver']}")
    print("-" * 65)
    print("Modality Contribution Breakdown:")
    for mod, w in explanation["modality_importance_weights"].items():
        bar = "#" * int(w * 25)
        print(f"  - {mod:<28} {w*100:5.1f}%  [{bar}]")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print("\n>>> TEST CASE 1: Authentic Merchant QR Code <<<")
    test_qr("samples/benign_merchant.png")

    print(">>> TEST CASE 2: Authentic EMVCo BharatQR Standard <<<")
    test_qr("samples/benign_emvco.png")

    print(">>> TEST CASE 3: Malicious Phishing QR Code <<<")
    test_qr("samples/malicious_phishing.png")

    print(">>> TEST CASE 4: Malicious Fake APK Download QR Code <<<")
    test_qr("samples/malicious_apk.png")

    print(">>> TEST CASE 5: Malicious UPI Refund Scam QR Code <<<")
    test_qr("samples/malicious_upi_scam.png")

    kaggle_b = os.path.expanduser(r"~\.cache\kagglehub\datasets\samahsadiq\benign-and-malicious-qr-codes\versions\1\QR codes\benign\benign\benign_0.png")
    if os.path.exists(kaggle_b):
        print(">>> TEST CASE 6: Real Kaggle Benign QR Code (google.com) <<<")
        test_qr(kaggle_b)

    kaggle_m = os.path.expanduser(r"~\.cache\kagglehub\datasets\samahsadiq\benign-and-malicious-qr-codes\versions\1\QR codes\malicious\malicious\malicious_316254.png")
    if os.path.exists(kaggle_m):
        print(">>> TEST CASE 7: Real Kaggle Malicious Threat QR Code <<<")
        test_qr(kaggle_m)
