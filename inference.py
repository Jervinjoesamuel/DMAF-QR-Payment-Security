import os
import argparse
import json
import torch
import numpy as np

from dmaf.config import ModelConfig
from dmaf.models.dmaf_network import DMAFNetwork
from dmaf.explainability import ExplainableRiskAssessor
from dmaf.continual_learning import ContinualLearningEngine
from dmaf.preprocessing import ImagePreprocessor, TextTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="DMAF Real-Time Authentication Inference")
    parser.add_argument("--image", type=str, default="", help="Path to QR code image")
    parser.add_argument("--payload", type=str, default="https://secure-checkout.merchant-pay.com/order?id=8831", help="Decoded text payload")
    parser.add_argument("--url", type=str, default="", help="Target URL (defaults to payload)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dmaf_best.pt", help="Path to trained checkpoint")
    parser.add_argument("--tau", type=float, default=0.85, help="Adaptive authentication threshold tau")
    return parser.parse_args()


def run_inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] DMAF Real-Time QR Code Authentication Inspector [{device}]")

    # Load Model
    config = ModelConfig()
    model = DMAFNetwork(config).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] Loaded trained weights from {args.checkpoint}")
    else:
        print("[!] Note: Running with initialized model weights (checkpoint not found).")

    # Prepare inputs
    img_prep = ImagePreprocessor(target_size=(config.img_size, config.img_size))
    tokenizer = TextTokenizer(max_length=config.max_text_len, vocab_size=config.vocab_size)

    if args.image and os.path.exists(args.image):
        img_tensor = img_prep(args.image)
    else:
        # Create a representative QR mock image if none provided
        grid = np.random.choice([0, 255], size=(128, 128)).astype(np.uint8)
        img_tensor = img_prep(grid)

    target_url = args.url if args.url else args.payload
    text_tokens = tokenizer.tokenize(args.payload)
    url_tokens = tokenizer.tokenize(target_url)

    # Standardized telemetry & metadata tensors
    behavior_tensor = np.random.randn(1, config.behavior_seq_len, config.behavior_feature_dim).astype(np.float32)
    meta_tensor = np.random.randn(1, config.meta_feature_dim).astype(np.float32)

    t_img = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
    t_text = torch.from_numpy(text_tokens).unsqueeze(0).long().to(device)
    t_url = torch.from_numpy(url_tokens).unsqueeze(0).long().to(device)
    t_b = torch.from_numpy(behavior_tensor).float().to(device)
    t_m = torch.from_numpy(meta_tensor).float().to(device)

    # Perform Explainable Risk Assessment
    assessor = ExplainableRiskAssessor(model)
    explanation = assessor.explain(t_img, t_text, t_url, t_b, t_m)

    # Authentication Decision (Eqs. 22-23)
    engine = ContinualLearningEngine(model, tau_threshold=args.tau)
    with torch.no_grad():
        out = model(t_img, t_text, t_url, t_b, t_m)
        probs = out["probabilities"]
        pred_class = int(torch.argmax(probs, dim=-1).item())
        decision, confidence = engine.authenticate(probs, pred_class)

    print("\n" + "=" * 60)
    print("           DMAF AUTHENTICATION VERDICT REPORT")
    print("=" * 60)
    print(f"Decoded Payload:       {args.payload}")
    print(f"Authentication Result: {decision.upper()}")
    print(f"Predicted Class:       {'Benign (Legitimate)' if pred_class == 0 else 'Malicious / Fraud'}")
    print(f"Model Confidence (C):  {confidence * 100:.2f}% (Threshold tau: {args.tau})")
    print(f"Dominant Risk Driver:  {explanation['dominant_risk_driver']}")
    print("-" * 60)
    print("Modality Attention Importance Weights:")
    for mod, w in explanation["modality_importance_weights"].items():
        bar = "#" * int(w * 30)
        print(f"  - {mod:<30} {w*100:5.1f}%  [{bar}]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_inference()
