import os
import cv2
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from dmaf.preprocessing import ImagePreprocessor, TextTokenizer, TabularStandardizer
from dmaf.payment_security import clean_qr_text, extract_multimodal_features


class QRMultimodalDataset(Dataset):
    """
    Multimodal Dataset for DMAF:
    Loads QR images and extracts:
    1. QR code image tensor (I_p)
    2. Decoded payload tokens (T_p)
    3. URL tokens (U)
    4. Device behavioral telemetry sequence (B_p)
    5. Transaction metadata vector (M_p)
    6. Ground truth label (0: Benign / Legitimate, 1: Malicious / Fraud)
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        img_preprocessor: Optional[ImagePreprocessor] = None,
        tokenizer: Optional[TextTokenizer] = None,
        meta_standardizer: Optional[TabularStandardizer] = None,
        behavior_standardizer: Optional[TabularStandardizer] = None,
        img_size: int = 128,
        seq_len: int = 10,
        meta_dim: int = 16,
        behavior_dim: int = 8,
    ):
        self.samples = samples
        self.img_preprocessor = img_preprocessor or ImagePreprocessor(target_size=(img_size, img_size))
        self.tokenizer = tokenizer or TextTokenizer()
        self.meta_standardizer = meta_standardizer
        self.behavior_standardizer = behavior_standardizer
        self.seq_len = seq_len
        self.meta_dim = meta_dim
        self.behavior_dim = behavior_dim
        self.qr_detector = cv2.QRCodeDetector()

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_qr_payload(self, img_path: str) -> str:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                decoded_info, _, _ = self.qr_detector.detectAndDecode(img)
                if decoded_info:
                    return clean_qr_text(decoded_info)
        except Exception:
            pass
        return ""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        
        # 1. Image
        img_path = item.get("img_path", "")
        cv_img = None
        if img_path and os.path.exists(img_path):
            cv_img = cv2.imread(img_path)
            if cv_img is not None:
                img_tensor = self.img_preprocessor(cv_img)
            else:
                img_tensor = np.zeros((3, 128, 128), dtype=np.float32)
        elif "img_data" in item:
            img_tensor = self.img_preprocessor(item["img_data"])
        else:
            img_tensor = np.zeros((3, 128, 128), dtype=np.float32)

        # 2. Decoded Text Payload & 3. URL
        text_raw = item.get("text", "")
        if not text_raw and img_path and os.path.exists(img_path):
            text_raw = self._extract_qr_payload(img_path)
            item["text"] = text_raw  # cache in dict
            
        text_clean = clean_qr_text(text_raw)
        url_raw = item.get("url", text_clean)
        
        text_tokens = self.tokenizer.tokenize(text_clean)
        url_tokens = self.tokenizer.tokenize(url_raw)

        # 4. Behavioral Telemetry & 5. Transaction Metadata
        behavior_raw = item.get("behavior")
        meta_raw = item.get("meta")

        if behavior_raw is None or meta_raw is None:
            # Extract grounded multimodal features based on payload & image
            b_feat, m_feat = extract_multimodal_features(
                text_clean,
                cv_img=cv_img,
                behavior_seq_len=self.seq_len,
                behavior_dim=self.behavior_dim,
                meta_dim=self.meta_dim,
            )
            if behavior_raw is None:
                behavior_raw = b_feat
            if meta_raw is None:
                meta_raw = m_feat

        behavior_raw = np.asarray(behavior_raw, dtype=np.float32)
        if self.behavior_standardizer:
            behavior_raw = self.behavior_standardizer.transform(behavior_raw)

        meta_raw = np.asarray(meta_raw, dtype=np.float32)
        if self.meta_standardizer:
            meta_raw = self.meta_standardizer.transform(meta_raw)

        # Ground truth label
        label = int(item.get("label", 0))

        return {
            "img": torch.from_numpy(img_tensor).float(),
            "text": torch.from_numpy(text_tokens).long(),
            "url": torch.from_numpy(url_tokens).long(),
            "behavior": torch.from_numpy(behavior_raw).float(),
            "meta": torch.from_numpy(meta_raw).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def scan_kaggle_directory(
    data_dir: str,
    max_samples_per_class: Optional[int] = None,
    cache_path: str = "data/kaggle_decoded_cache.json",
    predecode_threads: int = 8,
) -> List[Dict[str, Any]]:
    """
    Recursively scans the Kaggle dataset directory (handles nested folders like benign/benign),
    balances classes, and decodes QR payloads with multithreading and disk caching.
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    
    # Load existing cache if present
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"[*] Loaded {len(cache)} pre-decoded entries from cache: {cache_path}")
        except Exception:
            cache = {}

    samples = []
    detector = cv2.QRCodeDetector()

    for category, label in [("benign", 0), ("malicious", 1)]:
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            continue

        found_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            found_files.extend(glob.glob(os.path.join(cat_dir, "**", ext), recursive=True))

        found_files = sorted(found_files)
        if max_samples_per_class:
            found_files = found_files[:max_samples_per_class]

        print(f"[*] Found {len(found_files)} images for category '{category}' (label {label}).")

        # Multi-threaded decoding for items not yet in cache
        to_decode = [f for f in found_files if f not in cache]
        if to_decode:
            print(f"[*] Pre-decoding {len(to_decode)} {category} QR codes using {predecode_threads} threads...")
            def decode_file(fpath):
                try:
                    im = cv2.imread(fpath)
                    if im is not None:
                        txt, _, _ = detector.detectAndDecode(im)
                        return fpath, clean_qr_text(txt)
                except Exception:
                    pass
                return fpath, ""

            with ThreadPoolExecutor(max_workers=predecode_threads) as pool:
                results = list(pool.map(decode_file, to_decode))
                for fpath, cleaned in results:
                    cache[fpath] = cleaned

            # Persist updated cache
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            except Exception as e:
                print(f"[!] Warning: Could not save cache: {e}")

        for f in found_files:
            text = cache.get(f, "")
            samples.append({
                "img_path": f,
                "label": label,
                "text": text,
                "url": text,
            })

    return samples


def generate_payment_benchmark_samples(num_per_category: int = 500) -> List[Dict[str, Any]]:
    """
    Generates high-fidelity payment QR samples across legitimate payment schemes
    (UPI, EMVCo, Banking Gateways) and malicious fraud schemes (Phishing, Fake APKs, Scam VPAs).
    """
    benign_payments = [
        "upi://pay?pa=freshmart@icici&pn=FreshMartGrocery&am=85.50&cu=INR&mc=5411",
        "upi://pay?pa=starbucks@okhdfcbank&pn=StarbucksCoffee&am=350.00&cu=INR",
        "upi://pay?pa=shell.fuel@okaxis&pn=ShellGasStation&am=1200.00&cu=INR",
        "upi://pay?pa=metro.rail@paytm&pn=CityMetroTransit&am=40.00&cu=INR",
        "upi://pay?pa=amazon.seller@upi&pn=AmazonPayMerchant&am=899.00&cu=INR",
        "00020101021226580014A00000052701200106999999520441115303356540510.005802IN5913MerchantName6009Bangalore",
        "00020101021126400014A0000005270110011499999999995204541153033565802IN5911CornerStore6006Mumbai",
        "00020101021126580014A000000527012001068888885204581253033565406150.005802IN5915CoffeeHouseCafe6005Delhi",
        "00020101021226480014A000000527011801121234567890125204599953033565802IN5909SuperMart6007Chennai",
        "00020101021126600014A000000527012201087777777752045311530335654072500.005802IN5912ApparelStore6009Kolkata",
        "https://checkout.stripe.com/pay/cs_live_a1b2c3d4e5f6g7h8",
        "https://rzp.io/l/merchant_checkout_session_49204",
        "https://paypal.me/verifiedstore/45.00",
        "https://netbanking.hdfcbank.com/netbanking/entry",
        "https://pay.google.com/gp/p/ui/pay?id=tx_992819",
        "https://p.paytm.me/xP5t/storecheckout",
    ]

    malicious_payments = [
        "http://secure-update-bank-login.xyz/auth?session=expired",
        "http://sbi-reward-points-claim.top/redeem?user=984920",
        "http://paytm-kyc-verification.online/login.php?urgent=true",
        "http://free-cashback-reward.top/download/paytm_update.apk",
        "http://192.168.1.105:8080/download/gpay_bonus_v2.apk",
        "http://hdfc-netbanking-portal.tk/secure/login",
        "upi://pay?pa=lottery.refund.desk@ybl&pn=LotteryClaimDesk&am=5000.00&cu=INR",
        "upi://pay?pa=tax.refund.helpdesk@upi&pn=IncomeTaxRefundClaim&am=12500.00&cu=INR",
        "http://icici-card-protection.click/verify/cvv",
        "http://pay-merchant-gateway.cc/checkout.php?fake=1",
        "http://fast-loan-approval.buzz/download/loan_agent.apk",
        "http://10.0.0.1:8888/phish/bank_account_verification",
    ]

    import qrcode

    # Pre-render distinct QR images for speed
    all_urls = benign_payments + malicious_payments
    qr_images = {}
    for u in all_urls:
        try:
            qr = qrcode.QRCode(version=2, box_size=6, border=2)
            qr.add_data(u)
            qr.make(fit=True)
            qr_images[u] = np.array(qr.make_image(fill_color="black", back_color="white").convert("RGB"))
        except Exception:
            qr_images[u] = np.random.choice([0, 255], size=(128, 128, 3)).astype(np.uint8)

    samples = []
    for i in range(num_per_category):
        # Benign Payment
        url_b = benign_payments[i % len(benign_payments)]
        samples.append({
            "img_path": "",
            "img_data": qr_images[url_b],
            "text": url_b,
            "url": url_b,
            "label": 0,
        })

        # Malicious Threat Payment
        url_m = malicious_payments[i % len(malicious_payments)]
        samples.append({
            "img_path": "",
            "img_data": qr_images[url_m],
            "text": url_m,
            "url": url_m,
            "label": 1,
        })

    return samples


def generate_synthetic_benchmark(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Combines general web benchmark with payment security samples.
    """
    return generate_payment_benchmark_samples(num_per_category=num_samples // 2)
