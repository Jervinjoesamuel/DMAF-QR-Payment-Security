import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dmaf.config import ModelConfig
from dmaf.models.dmaf_network import DMAFNetwork
from dmaf.dataset import (
    QRMultimodalDataset,
    scan_kaggle_directory,
    generate_synthetic_benchmark,
    generate_payment_benchmark_samples,
)

DEFAULT_KAGGLE_PATH = os.path.expanduser(
    r"~\.cache\kagglehub\datasets\samahsadiq\benign-and-malicious-qr-codes\versions\1\QR codes"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DMAF Multimodal Framework on Kaggle and Payment QR Datasets")
    parser.add_argument("--data_dir", type=str, default="", help="Path to Kaggle dataset directory")
    parser.add_argument("--max_samples_per_class", type=int, default=2500, help="Max samples per class from Kaggle dataset (0 for all)")
    parser.add_argument("--include_payment_samples", type=int, default=1000, help="Number of authentic & malicious payment QR samples to include")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Adam)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Regularization lambda")
    parser.add_argument("--save_path", type=str, default="checkpoints/dmaf_best.pt", help="Checkpoint save path")
    parser.add_argument("--synthetic_samples", type=int, default=2000, help="Fallback synthetic samples if data_dir not found")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing DMAF Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    target_data_dir = args.data_dir
    if not target_data_dir or not os.path.exists(target_data_dir):
        if os.path.exists(DEFAULT_KAGGLE_PATH):
            target_data_dir = DEFAULT_KAGGLE_PATH
            print(f"[*] Using local Kaggle dataset at: {target_data_dir}")

    samples = []
    # 1. Load Kaggle Dataset
    if target_data_dir and os.path.exists(target_data_dir):
        max_samples = None if args.max_samples_per_class == 0 else args.max_samples_per_class
        print(f"[*] Scanning Kaggle dataset (max per class: {max_samples or 'All'})...")
        kaggle_samples = scan_kaggle_directory(target_data_dir, max_samples_per_class=max_samples)
        samples.extend(kaggle_samples)
        print(f"[*] Loaded {len(kaggle_samples)} samples from Kaggle dataset.")
    else:
        print("[!] Kaggle dataset directory not found, using synthetic fallback.")
        samples.extend(generate_synthetic_benchmark(args.synthetic_samples))

    # 2. Augment with High-Fidelity Payment QR samples
    if args.include_payment_samples > 0:
        print(f"[*] Augmenting training set with {args.include_payment_samples} authentic and malicious payment QR samples...")
        payment_samples = generate_payment_benchmark_samples(num_per_category=args.include_payment_samples // 2)
        samples.extend(payment_samples)

    print(f"[*] Total combined dataset size: {len(samples)} samples (Balanced Benign & Malicious).")

    dataset = QRMultimodalDataset(samples)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model_config = ModelConfig()
    model = DMAFNetwork(model_config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_val_f1 = 0.0

    print(f"[*] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"[*] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            img = batch["img"].to(device)
            text = batch["text"].to(device)
            url = batch["url"].to(device)
            behavior = batch["behavior"].to(device)
            meta = batch["meta"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(img, text, url, behavior, meta)
            loss = criterion(outputs["logits"], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs["logits"], dim=-1)
            train_preds.extend(preds.cpu().numpy().tolist())
            train_targets.extend(labels.cpu().numpy().tolist())

        train_acc = accuracy_score(train_targets, train_preds)
        train_loss = total_loss / len(train_set)

        # Validation
        model.eval()
        val_preds, val_probs, val_targets = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["img"].to(device)
                text = batch["text"].to(device)
                url = batch["url"].to(device)
                behavior = batch["behavior"].to(device)
                meta = batch["meta"].to(device)
                labels = batch["label"].to(device)

                outputs = model(img, text, url, behavior, meta)
                probs = outputs["probabilities"]
                preds = torch.argmax(probs, dim=-1)

                val_probs.extend(probs[:, 1].cpu().numpy().tolist())
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_targets, val_probs)
        except Exception:
            val_auc = 0.5
        scheduler.step(val_f1)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:4.1f}s) | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Val AUC: {val_auc:.4f}")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, args.save_path)
            print(f"  --> [*] Saved checkpoint: {args.save_path} (F1: {val_f1*100:.2f}%)")

    print("=" * 75)
    print(f"[+] Training complete. Peak Validation F1: {best_val_f1 * 100:.2f}%\n")


if __name__ == "__main__":
    train()
