import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple


class ImagePreprocessor:
    """
    Image enhancement pipeline (Eq. 2: I_p = Psi(I)):
    - Bilateral / Gaussian denoising
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Geometric alignment / perspective deskewing
    """
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _deskew_qr(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return image
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)
                
        if not angles:
            return image
            
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return image
            
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    def __call__(self, img_input: Any) -> np.ndarray:
        if isinstance(img_input, str):
            image = cv2.imread(img_input)
            if image is None:
                raise FileNotFoundError(f"Cannot load image at path: {img_input}")
        elif isinstance(img_input, np.ndarray):
            image = img_input.copy()
        else:
            image = np.array(img_input)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 1. Bilateral filter for edge-preserving denoising
        denoised = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

        # 2. Geometric alignment
        deskewed = self._deskew_qr(denoised)

        # 3. Adaptive contrast normalization (LAB space CLAHE)
        lab = cv2.cvtColor(deskewed, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_clahe = self.clahe.apply(l_chan)
        enhanced_lab = cv2.merge((l_clahe, a_chan, b_chan))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 4. Resize and normalize to [0, 1] range float32 (CHW layout)
        resized = cv2.resize(enhanced, self.target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))


class TextTokenizer:
    """
    Subword / character-level tokenizer for payload and URL representation (Eq. 3: T_p = Phi(T)).
    Handles URI schemes, query parameters, hex encodings, and common payment tokens.
    """
    def __init__(self, max_length: int = 128, vocab_size: int = 5000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.vocab = self._build_default_vocab()

    def _build_default_vocab(self) -> Dict[str, int]:
        vocab = {"<PAD>": 0, "<UNK>": 1}
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;=%"
        for c in chars:
            vocab[c] = len(vocab)
        keywords = [
            "http", "https", "upi", "pay", "alipay", "wechat", "wallet", "merchant", 
            "token", "amount", "redirect", "login", "secure", "auth", "session", "bank",
            "icici", "hdfc", "sbi", "axis", "paytm", "gpay", "phonepe", "paypal", "stripe", "razorpay",
            "bharatqr", "emvco", "qr", "checkout", "gateway", "invoice", "transfer",
            "fraud", "phish", "malware", "apk", "update", "verify", "account", "bonus", "reward",
            "cashback", "claim", "free", "lottery", "support", "refund", "kyc", "vpa", "inr"
        ]
        for kw in keywords:
            if kw not in vocab and len(vocab) < self.vocab_size:
                vocab[kw] = len(vocab)
        return vocab

    def tokenize(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.max_length, dtype=np.int64)
        text = str(text).strip().lower()
        tokens = []
        i = 0
        while i < len(text) and len(tokens) < self.max_length:
            matched = False
            for length in (8, 6, 4):
                sub = text[i:i+length]
                if sub in self.vocab:
                    tokens.append(self.vocab[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                c = text[i]
                tokens.append(self.vocab.get(c, self.unk_id))
                i += 1
                
        if len(tokens) < self.max_length:
            tokens += [self.pad_id] * (self.max_length - len(tokens))
        return np.array(tokens[:self.max_length], dtype=np.int64)


class TabularStandardizer:
    """
    Standardization for transaction metadata M and behavioral features B (Eq. 4).
    """
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.mean = np.zeros(feature_dim, dtype=np.float32)
        self.std = np.ones(feature_dim, dtype=np.float32)
        self.is_fitted = False

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0).astype(np.float32)
        self.std = np.std(data, axis=0).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if not self.is_fitted:
            return data
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
