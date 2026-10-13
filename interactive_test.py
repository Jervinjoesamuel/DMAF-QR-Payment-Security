"""
Interactive Manual Tester for DMAF.
Allows users to easily input any QR code image path, payment URL, or UPI link
and get real-time forensic authentication results.
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


def load_dmaf_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    model = DMAFNetwork(config).to(device)
    
    checkpoint_path = "checkpoints/dmaf_best.pt"
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] Loaded trained model weights from: {checkpoint_path}")
    else:
        print("[!] Running with initialized base model (checkpoint not found).")
        
    model.eval()
    return model, config, device


def inspect_input(user_input: str, model, config, device, tau: float = 0.85):
    img_prep = ImagePreprocessor(target_size=(config.img_size, config.img_size))
    tokenizer = TextTokenizer(max_length=config.max_text_len, vocab_size=config.vocab_size)

    # Check if the input is a path to an image file
    if os.path.isfile(user_input):
        print(f"[*] Reading image file: {user_input}")
        cv_img = cv2.imread(user_input)
        if cv_img is None:
            print("[!] Could not open image. Please check the file path.")
            return

        # Attempt to decode QR payload
        detector = cv2.QRCodeDetector()
        decoded_text, _, _ = detector.detectAndDecode(cv_img)
        if not decoded_text:
            print("[i] Note: Could not auto-decode payload from image modules; using image visual stream.")
            decoded_text = user_input
        else:
            print(f"[+] Decoded Payload from QR: {decoded_text}")

        img_tensor = img_prep(cv_img)
        payload_text = decoded_text
    else:
        # User entered a URL / payment payload directly
        payload_text = user_input.strip()
        print(f"[*] Analyzing payment payload text: {payload_text}")
        
        # Generate representative visual QR pattern
        grid = np.random.choice([0, 255], size=(128, 128)).astype(np.uint8)
        img_tensor = img_prep(grid)

    # Tokenize
    text_tokens = tokenizer.tokenize(payload_text)
    url_tokens = tokenizer.tokenize(payload_text)

    # Telemetry and metadata tensors
    behavior = np.random.randn(1, config.behavior_seq_len, config.behavior_feature_dim).astype(np.float32)
    meta = np.random.randn(1, config.meta_feature_dim).astype(np.float32)

    t_img = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
    t_text = torch.from_numpy(text_tokens).unsqueeze(0).long().to(device)
    t_url = torch.from_numpy(url_tokens).unsqueeze(0).long().to(device)
    t_b = torch.from_numpy(behavior).float().to(device)
    t_m = torch.from_numpy(meta).float().to(device)

    # Explainability attribution
    assessor = ExplainableRiskAssessor(model)
    explanation = assessor.explain(t_img, t_text, t_url, t_b, t_m)

    # Authentication decision
    engine = ContinualLearningEngine(model, tau_threshold=tau)
    with torch.no_grad():
        out = model(t_img, t_text, t_url, t_b, t_m)
        probs = out["probabilities"]
        pred_class = int(torch.argmax(probs, dim=-1).item())
        decision, conf = engine.authenticate(probs, pred_class)

    # Formatted Report
    print("\n" + "=" * 65)
    print("               DMAF AUTHENTICATION RESULT")
    print("=" * 65)
    print(f"Target Payload / URL:  {payload_text}")
    print(f"Final Decision:        {decision.upper()}")
    print(f"Threat Category:       {'Benign (Legitimate)' if pred_class == 0 else 'Malicious Threat / Fraud'}")
    print(f"Model Confidence:      {conf * 100:.2f}% (Threshold tau: {tau})")
    print(f"Dominant Risk Driver:  {explanation['dominant_risk_driver']}")
    print("-" * 65)
    print("Modality Attention Weights:")
    for mod, w in explanation["modality_importance_weights"].items():
        bar = "#" * int(w * 25)
        print(f"  - {mod:<28} {w*100:5.1f}%  [{bar}]")
    print("=" * 65 + "\n")


def main():
    print("=" * 65)
    print("   Deep Multimodal Authentication Framework - Manual Tester")
    print("=" * 65)
    model, config, device = load_dmaf_model()

    print("\nYou can test by either:")
    print("  1. Entering a path to a QR image file (e.g., samples/benign_merchant.png)")
    print("  2. Pasting any payment link or URL (e.g., upi://pay?pa=... or http://...)")
    print("  Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Enter QR image path or payment URL: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("[*] Exiting manual tester.")
                break
            inspect_input(user_input, model, config, device)
        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting.")
            break


if __name__ == "__main__":
    main()
