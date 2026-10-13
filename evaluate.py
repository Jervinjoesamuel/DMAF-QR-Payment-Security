import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
)

from dmaf.config import ModelConfig
from dmaf.models.dmaf_network import DMAFNetwork
from dmaf.dataset import QRMultimodalDataset, generate_synthetic_benchmark, scan_kaggle_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DMAF Framework")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dmaf_best.pt", help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="", help="Path to Kaggle data")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--samples", type=int, default=1000, help="Number of test samples")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] DMAF Comprehensive Evaluation Benchmark on {device}")

    # Load data
    if args.data_dir and os.path.exists(args.data_dir):
        raw_samples = scan_kaggle_directory(args.data_dir)
        if not raw_samples:
            raw_samples = generate_synthetic_benchmark(args.samples)
    else:
        raw_samples = generate_synthetic_benchmark(args.samples)

    dataset = QRMultimodalDataset(raw_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    config = ModelConfig()
    model = DMAFNetwork(config).to(device)

    if os.path.exists(args.checkpoint):
        print(f"[*] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[!] No checkpoint found at specified path. Evaluating initialized model architecture.")

    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    # Latency and throughput profiling
    sample_latencies = []
    
    # Warmup
    dummy_img = torch.zeros((1, 3, 128, 128), device=device)
    dummy_text = torch.zeros((1, 128), dtype=torch.long, device=device)
    dummy_url = torch.zeros((1, 128), dtype=torch.long, device=device)
    dummy_b = torch.zeros((1, 10, 8), device=device)
    dummy_m = torch.zeros((1, 16), device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_img, dummy_text, dummy_url, dummy_b, dummy_m)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_eval_start = time.time()
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            text = batch["text"].to(device)
            url = batch["url"].to(device)
            behavior = batch["behavior"].to(device)
            meta = batch["meta"].to(device)
            labels = batch["label"].to(device)

            # Per-sample latency check
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(img, text, url, behavior, meta)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            batch_time_ms = (t1 - t0) * 1000
            sample_latencies.append(batch_time_ms / len(labels))

            probs = outputs["probabilities"]
            preds = torch.argmax(probs, dim=-1)

            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    total_eval_time = time.time() - t_eval_start

    # Metrics computation (Paper Tables 1 & 2)
    acc = accuracy_score(all_targets, all_preds) * 100
    prec = precision_score(all_targets, all_preds, zero_division=0) * 100
    rec = recall_score(all_targets, all_preds, zero_division=0) * 100
    f1 = f1_score(all_targets, all_preds, zero_division=0) * 100
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds, labels=[0, 1]).ravel()
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
    mcc = matthews_corrcoef(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds) * 100

    avg_latency_ms = float(np.mean(sample_latencies))
    throughput = len(all_targets) / total_eval_time
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    gpu_mem = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    print("\n" + "=" * 65)
    print("      TABLE 1: CLASSIFICATION PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"Accuracy (%):             {acc:.2f}%")
    print(f"Precision (%):            {prec:.2f}%")
    print(f"Recall (%):               {rec:.2f}%")
    print(f"F1-Score (%):             {f1:.2f}%")
    print(f"ROC-AUC:                  {auc:.4f}")

    print("\n" + "=" * 65)
    print("      TABLE 2: SECURITY & RELIABILITY PERFORMANCE")
    print("=" * 65)
    print(f"False Positive Rate (FPR): {fpr:.2f}%")
    print(f"False Negative Rate (FNR): {fnr:.2f}%")
    print(f"Detection Time / Latency:  {avg_latency_ms:.2f} ms/sample")
    print(f"Matthews Corr. (MCC):      {mcc:.4f}")
    print(f"Balanced Accuracy (%):     {bal_acc:.2f}%")

    print("\n" + "=" * 65)
    print("      TABLE 3: COMPUTATIONAL EFFICIENCY & DEPLOYMENT")
    print("=" * 65)
    print(f"Model Parameters:          {num_params_m:.2f} Million")
    print(f"Inference Latency:         {avg_latency_ms:.2f} ms/sample")
    print(f"Peak GPU Memory:           {gpu_mem:.2f} GB")
    print(f"Throughput:                {throughput:.1f} QR Codes/second")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    evaluate()
