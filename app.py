import os
import time
import random
import glob
import cv2
import qrcode
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify

from dmaf.config import ModelConfig
from dmaf.models.dmaf_network import DMAFNetwork
from dmaf.preprocessing import ImagePreprocessor, TextTokenizer
from dmaf.explainability import ExplainableRiskAssessor
from dmaf.continual_learning import ContinualLearningEngine
from dmaf.payment_security import parse_payment_payload, extract_multimodal_features, clean_qr_text

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model at Startup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = ModelConfig()
MODEL = DMAFNetwork(CONFIG).to(DEVICE)

CHECKPOINT_PATH = os.path.join("checkpoints", "dmaf_best.pt")
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    val_acc = ckpt.get("val_acc", 0.0)
    print(f"[*] DMAF Server: Loaded checkpoint from {CHECKPOINT_PATH} (Trained Val Acc: {val_acc*100:.2f}%) on {DEVICE}")
else:
    print("[!] DMAF Server: Checkpoint not found; running with default weights.")

MODEL.eval()

IMG_PREP = ImagePreprocessor(target_size=(CONFIG.img_size, CONFIG.img_size))
TOKENIZER = TextTokenizer(max_length=CONFIG.max_text_len, vocab_size=CONFIG.vocab_size)
ASSESSOR = ExplainableRiskAssessor(MODEL)
CONTINUAL = ContinualLearningEngine(MODEL, tau_threshold=0.85)

KAGGLE_DIR = os.path.expanduser(
    r"~\.cache\kagglehub\datasets\samahsadiq\benign-and-malicious-qr-codes\versions\1\QR codes"
)


def process_and_authenticate(cv_img, payload_text, tau=0.85):
    payload_text = clean_qr_text(payload_text)
    if not payload_text and cv_img is None:
        payload_text = "upi://pay?pa=freshmart@icici&pn=FreshMartGrocery&am=85.50&cu=INR"

    # If image is missing but text exists, generate authentic QR code image
    generated_img_path = None
    if cv_img is None:
        qr = qrcode.QRCode(version=2, box_size=6, border=2)
        qr.add_data(payload_text)
        qr.make(fit=True)
        pil_img = qr.make_image(fill_color="black", back_color="white")
        cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Save generated QR preview
        gen_filename = f"gen_qr_{int(time.time()*1000)}.png"
        gen_full_path = os.path.join(app.config["UPLOAD_FOLDER"], gen_filename)
        cv2.imwrite(gen_full_path, cv_img)
        generated_img_path = f"/static/uploads/{gen_filename}"

    # Preprocess image tensor
    img_tensor = IMG_PREP(cv_img)

    # Tokenize payload and URL
    text_tokens = TOKENIZER.tokenize(payload_text)
    url_tokens = TOKENIZER.tokenize(payload_text)

    # Extract protocol-grounded multimodal features (Behavior B_p and Metadata M_p)
    behavior, meta = extract_multimodal_features(
        payload_text,
        cv_img=cv_img,
        behavior_seq_len=CONFIG.behavior_seq_len,
        behavior_dim=CONFIG.behavior_feature_dim,
        meta_dim=CONFIG.meta_feature_dim,
    )

    t_img = torch.from_numpy(img_tensor).unsqueeze(0).float().to(DEVICE)
    t_text = torch.from_numpy(text_tokens).unsqueeze(0).long().to(DEVICE)
    t_url = torch.from_numpy(url_tokens).unsqueeze(0).long().to(DEVICE)
    t_b = torch.from_numpy(behavior).unsqueeze(0).float().to(DEVICE)
    t_m = torch.from_numpy(meta).unsqueeze(0).float().to(DEVICE)

    # 1. Forensic Explainability (Saliency Gradient Eq. 20)
    explanation = ASSESSOR.explain(t_img, t_text, t_url, t_b, t_m)

    # 2. Decision rule (Eq. 22-23)
    with torch.no_grad():
        out = MODEL(t_img, t_text, t_url, t_b, t_m)
        probs = out["probabilities"]
        pred_class = int(torch.argmax(probs, dim=-1).item())
        decision, conf = CONTINUAL.authenticate(probs, pred_class)

    # 3. Payment Protocol Inspection & Risk Flags
    payment_info = parse_payment_payload(payload_text)

    return {
        "payload": payload_text,
        "decision": decision,
        "is_legitimate": (decision == "Legitimate"),
        "threat_category": "Benign / Authentic Transaction" if pred_class == 0 else "Malicious Threat / Fraud Attempt",
        "confidence": float(conf),
        "tau": tau,
        "dominant_driver": explanation["dominant_risk_driver"].replace("_", " "),
        "modality_weights": {
            k.replace("_", " "): round(float(v) * 100, 1)
            for k, v in explanation["modality_importance_weights"].items()
        },
        "payment_info": payment_info,
        "generated_img_path": generated_img_path,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, current_payload="")


@app.route("/analyze", methods=["POST"])
def analyze():
    payload_text = request.form.get("payload_text", "").strip()
    file = request.files.get("qr_image")
    
    cv_img = None
    saved_rel_path = None

    if file and file.filename != "":
        filename = f"qr_{int(time.time()*1000)}_{file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        saved_rel_path = f"/static/uploads/{filename}"

        cv_img = cv2.imread(filepath)
        if cv_img is not None:
            detector = cv2.QRCodeDetector()
            decoded_text, _, _ = detector.detectAndDecode(cv_img)
            if decoded_text:
                payload_text = clean_qr_text(decoded_text)

    result = process_and_authenticate(cv_img, payload_text)
    if saved_rel_path:
        result["image_path"] = saved_rel_path
    elif result.get("generated_img_path"):
        result["image_path"] = result["generated_img_path"]

    return render_template("index.html", result=result, current_payload=result["payload"])


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    payload = data.get("payload", "").strip()
    if not payload:
        return jsonify({"error": "Missing payload parameter"}), 400

    result = process_and_authenticate(None, payload)
    return jsonify(result)


@app.route("/api/random_kaggle_sample", methods=["GET"])
def random_kaggle_sample():
    category = request.args.get("category", "benign")
    cat_dir = os.path.join(KAGGLE_DIR, category)
    if not os.path.isdir(cat_dir):
        return jsonify({"error": f"Kaggle category {category} not found"}), 404

    files = glob.glob(os.path.join(cat_dir, "**", "*.png"), recursive=True)
    if not files:
        return jsonify({"error": "No images found in category"}), 404

    chosen_file = random.choice(files)
    im = cv2.imread(chosen_file)
    detector = cv2.QRCodeDetector()
    decoded, _, _ = detector.detectAndDecode(im)
    clean_text = clean_qr_text(decoded)

    # Copy image to static/uploads for display
    rel_name = f"sample_{int(time.time()*1000)}.png"
    dest_path = os.path.join(app.config["UPLOAD_FOLDER"], rel_name)
    cv2.imwrite(dest_path, im)

    return jsonify({
        "category": category,
        "payload": clean_text,
        "image_url": f"/static/uploads/{rel_name}",
        "source": "Kaggle 200,000 QR Dataset"
    })


if __name__ == "__main__":
    print("[*] Starting DMAF Mobile Payment Security Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
