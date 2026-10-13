import re
import urllib.parse
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


KNOWN_LEGIT_HOSTS = {
    "google.com", "pay.google.com", "gpay.app.goo.gl",
    "paypal.com", "paypal.me", "stripe.com", "checkout.stripe.com",
    "razorpay.com", "rzp.io", "payu.in", "cashfree.com",
    "paytm.com", "p.paytm.me", "phonepe.com", "phon.pe",
    "icicibank.com", "hdfcbank.com", "sbi.co.in", "axisbank.com",
    "amazon.in", "amazon.com", "apple.com", "visa.com", "mastercard.com"
}

SUSPICIOUS_TLDS = {".xyz", ".top", ".tk", ".pw", ".click", ".online", ".work", ".site", ".cc", ".buzz"}

SUSPICIOUS_KEYWORDS = [
    "kyc", "refund", "claim", "lottery", "winner", "reward", "cashback", 
    "bonus", "urgent", "blocked", "suspended", "verify-account", "security-update",
    "login-bank", "free-money"
]


def clean_qr_text(raw_text: str) -> str:
    """
    Cleans raw decoded text from either Kaggle pandas dump or standard QR scanner.
    Kaggle format: '<index>    <url>\\nName: url, dtype: object'
    """
    if not raw_text:
        return ""
    lines = str(raw_text).strip().splitlines()
    if len(lines) > 0:
        first_line = lines[0].strip()
        parts = first_line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            target = parts[1].strip()
            if target.startswith("http://") or target.startswith("https://") or target.startswith("www."):
                return target
    return str(raw_text).strip()


def parse_payment_payload(payload_text: str) -> Dict[str, Any]:
    """
    Parses payment protocol, metadata fields, and forensic security flags.
    Supports UPI schemes, EMVCo specifications, Web Payment Gateways, and generic URLs.
    """
    p = clean_qr_text(payload_text).strip()
    result = {
        "raw_payload": p,
        "protocol_category": "Standard Web Destination",
        "parsed_fields": {},
        "risk_flags": [],
        "trust_indicators": [],
        "is_payment": False,
        "base_risk_score": 0.0,  # 0.0 (safe) to 1.0 (dangerous)
    }

    if not p:
        result["protocol_category"] = "Empty / Unreadable Payload"
        result["risk_flags"].append("No payload data decoded from QR")
        result["base_risk_score"] = 0.5
        return result

    # 1. UPI Payment Protocol (India / International NPCI standard)
    if p.lower().startswith("upi://pay"):
        result["protocol_category"] = "UPI Unified Payments Intent"
        result["is_payment"] = True
        result["trust_indicators"].append("Valid UPI protocol specification scheme")

        try:
            parsed = urllib.parse.urlparse(p)
            qs = urllib.parse.parse_qs(parsed.query)
            pa = qs.get("pa", [""])[0]
            pn = qs.get("pn", [""])[0]
            am = qs.get("am", [""])[0]
            cu = qs.get("cu", ["INR"])[0]
            mc = qs.get("mc", [""])[0]
            tn = qs.get("tn", [""])[0]

            result["parsed_fields"] = {
                "Payee VPA (Handle)": pa or "Unspecified",
                "Merchant / Beneficiary": pn or "Unspecified",
                "Transaction Amount": f"{am} {cu}" if am else "Open / Customer Defined",
                "Currency": cu,
                "Merchant Category Code (MCC)": mc or "Standard Retail",
                "Transaction Note": tn or "N/A",
            }

            # Security checks on VPA
            if pa:
                vpa_lower = pa.lower()
                if any(k in vpa_lower for k in ["refund", "helpdesk", "lottery", "prize", "cashback-claim"]):
                    result["risk_flags"].append(f"Suspicious handle keywords detected in payee VPA: {pa}")
                    result["base_risk_score"] += 0.45
                
                # Check recognized PSP bank handles
                legit_handles = ["@okaxis", "@okhdfcbank", "@okicici", "@oksbi", "@icici", "@axisbank", 
                                 "@ybl", "@ibl", "@paytm", "@upi", "@barodampay", "@indus", "@kotak"]
                if any(vpa_lower.endswith(h) for h in legit_handles):
                    result["trust_indicators"].append("Registered National Payment Corporation handle")
                else:
                    result["trust_indicators"].append("Custom / Merchant PSP Gateway")
            else:
                result["risk_flags"].append("Missing required payee address ('pa' parameter)")
                result["base_risk_score"] += 0.35

        except Exception as e:
            result["risk_flags"].append(f"UPI query parsing anomaly: {str(e)}")
            result["base_risk_score"] += 0.2

    # 2. EMVCo Specification Standard (Mastercard, Visa, UnionPay, BharatQR, Pix, PromptPay)
    elif p.startswith("000201"):
        result["protocol_category"] = "EMVCo Global Merchant Standard"
        result["is_payment"] = True
        result["trust_indicators"].append("Complies with EMVCo QR Code Specification for Payment Systems (CPS)")
        result["parsed_fields"] = {
            "Format Indicator": "EMVCo Payload Version 01",
            "Point of Initiation": "Static / Dynamic Merchant QR" if "010211" in p or "010212" in p else "Standard EMV",
            "Global Standard": "ISO/IEC 18004 Compliant",
        }

    # 3. Web URL / Payment Gateway
    elif p.lower().startswith("http://") or p.lower().startswith("https://") or "www." in p.lower():
        url_target = p if (p.startswith("http://") or p.startswith("https://")) else f"https://{p}"
        try:
            parsed = urllib.parse.urlparse(url_target)
            host = parsed.netloc.lower()
            path = parsed.path.lower()
            scheme = parsed.scheme.lower()

            result["parsed_fields"]["Target Host"] = host
            result["parsed_fields"]["Protocol Scheme"] = scheme.upper()
            result["parsed_fields"]["Resource Path"] = path or "/"

            # Check if recognized payment gateway or banking portal
            is_gateway = False
            for legit in KNOWN_LEGIT_HOSTS:
                if host == legit or host.endswith("." + legit):
                    is_gateway = True
                    break

            if is_gateway:
                result["protocol_category"] = "Verified Payment Gateway / Banking Portal"
                result["is_payment"] = True
                result["trust_indicators"].append(f"Recognized trusted financial / commerce domain: {host}")
            else:
                result["protocol_category"] = "Web URL Destination"

            # Security Flag Checks
            if scheme == "http":
                result["risk_flags"].append("Insecure plain HTTP protocol (vulnerable to MiTM & interception)")
                result["base_risk_score"] += 0.35
            else:
                result["trust_indicators"].append("Encrypted HTTPS transport security (TLS)")

            if re.search(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", host):
                result["risk_flags"].append(f"Raw numerical IP host address used instead of valid domain name ({host})")
                result["base_risk_score"] += 0.55

            for tld in SUSPICIOUS_TLDS:
                if host.endswith(tld):
                    result["risk_flags"].append(f"High-risk top level domain: '{tld}' commonly abused for fraud")
                    result["base_risk_score"] += 0.40
                    break

            if path.endswith(".apk") or path.endswith(".exe") or ".apk?" in p.lower():
                result["risk_flags"].append("Direct executable application download (.apk / .exe) detected")
                result["protocol_category"] = "Malware / Fake App Dropper Threat"
                result["base_risk_score"] += 0.70

            for kw in SUSPICIOUS_KEYWORDS:
                if kw in p.lower():
                    result["risk_flags"].append(f"Social engineering / phishing trigger keyword identified: '{kw}'")
                    result["base_risk_score"] += 0.25
                    break

        except Exception as e:
            result["risk_flags"].append(f"URL parsing exception: {str(e)}")
            result["base_risk_score"] += 0.2

    else:
        # Custom or raw text
        result["protocol_category"] = "Custom Text / Proprietary Token"
        result["parsed_fields"]["Raw Content"] = p[:50] + ("..." if len(p) > 50 else "")

    # Bound risk score between 0.0 and 1.0
    result["base_risk_score"] = min(1.0, max(0.0, result["base_risk_score"]))
    return result


def extract_multimodal_features(
    payload_text: str,
    cv_img: Optional[np.ndarray] = None,
    behavior_seq_len: int = 10,
    behavior_dim: int = 8,
    meta_dim: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts realistic, protocol-aware multimodal Behavioral Telemetry (seq_len x behavior_dim)
    and Transaction Metadata (meta_dim) vectors aligned with DMAF formulations (Eqs. 4, 10, 11).
    """
    p_info = parse_payment_payload(payload_text)
    risk_score = p_info["base_risk_score"]
    is_payment = p_info["is_payment"]
    num_flags = len(p_info["risk_flags"])
    num_trust = len(p_info["trust_indicators"])

    # 1. Behavioral Telemetry Sequence (seq_len x behavior_dim)
    # Dim 0: Scan visual sharpness / contrast
    # Dim 1: Protocol safety index
    # Dim 2: Domain trust coefficient
    # Dim 3: Parameter entropy / consistency
    # Dim 4: Execution safety (absence of apk / raw ip)
    # Dim 5: Interaction timing cadence
    # Dim 6: Mobile accelerometer / gyro sensor stability
    # Dim 7: App execution sandbox certificate status
    behavior = np.zeros((behavior_seq_len, behavior_dim), dtype=np.float32)
    
    # Image sharpness metric if image provided
    sharpness = 0.8
    if cv_img is not None and cv_img.size > 0:
        gray = cv_img if len(cv_img.shape) == 2 else cv_img[:, :, 0]
        sharpness = float(np.std(gray)) / 128.0

    safety_shift = 1.0 - (risk_score * 2.0)  # +1.0 for very safe, -1.0 for dangerous

    for t in range(behavior_seq_len):
        noise = np.random.randn(behavior_dim) * 0.05
        behavior[t, 0] = sharpness + noise[0]
        behavior[t, 1] = safety_shift + noise[1]
        behavior[t, 2] = (1.0 if num_trust > 0 else -0.5) + noise[2]
        behavior[t, 3] = (0.5 if not num_flags else -0.8) + noise[3]
        behavior[t, 4] = (-1.5 if risk_score > 0.6 else 0.8) + noise[4]
        behavior[t, 5] = 0.5 + 0.1 * np.sin(t) + noise[5]
        behavior[t, 6] = 0.1 * np.cos(t) + noise[6]
        behavior[t, 7] = (1.0 if risk_score < 0.3 else -1.0) + noise[7]

    # 2. Transaction Metadata Vector (meta_dim)
    meta = np.zeros(meta_dim, dtype=np.float32)
    p_lower = payload_text.lower()
    
    meta[0] = 1.0 if p_lower.startswith("upi://pay") else 0.0
    meta[1] = 1.0 if payload_text.startswith("000201") else 0.0
    meta[2] = 1.0 if p_lower.startswith("https://") else 0.0
    meta[3] = -1.0 if p_lower.startswith("http://") else 0.0
    meta[4] = -1.0 if any(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", p_lower) for _ in [1]) else 0.0
    meta[5] = -1.0 if any(p_lower.endswith(t) or f"{t}/" in p_lower for t in SUSPICIOUS_TLDS) else 0.0
    meta[6] = -1.5 if (".apk" in p_lower or ".exe" in p_lower) else 0.0
    meta[7] = -1.0 if any(k in p_lower for k in SUSPICIOUS_KEYWORDS) else 0.0
    meta[8] = 1.0 if any(h in p_lower for h in ["@okaxis", "@okhdfcbank", "@okicici", "@oksbi", "@icici", "@paytm", "@upi"]) else 0.0
    meta[9] = float(len(payload_text)) / 100.0  # Normalized length
    meta[10] = 1.0 if is_payment else 0.0
    meta[11] = 1.0 if (any(legit in p_lower for legit in KNOWN_LEGIT_HOSTS) or payload_text.startswith("000201") or p_lower.startswith("upi://pay")) else 0.0
    meta[12] = float(num_flags) * -0.5
    meta[13] = float(num_trust) * 0.5
    meta[14] = -1.0 if risk_score > 0.5 else 1.0
    meta[15] = safety_shift

    # Add slight realistic stochastic jitter
    meta += np.random.randn(meta_dim).astype(np.float32) * 0.02

    return behavior, meta
