import os
import qrcode
from PIL import Image

def generate_sample_qr_codes():
    os.makedirs("samples", exist_ok=True)
    
    samples_data = [
        {
            "filename": "samples/benign_merchant.png",
            "payload": "upi://pay?pa=freshmart@icici&pn=FreshMartGrocery&am=85.50&cu=INR",
            "label": "Legitimate Merchant Payment"
        },
        {
            "filename": "samples/benign_emvco.png",
            "payload": "00020101021226580014A00000052701200106999999520441115303356540510.005802IN5913MerchantName6009Bangalore",
            "label": "Legitimate EMVCo BharatQR Payment"
        },
        {
            "filename": "samples/malicious_phishing.png",
            "payload": "http://secure-update-bank-login.xyz/auth?session=expired",
            "label": "Malicious Phishing Portal"
        },
        {
            "filename": "samples/malicious_apk.png",
            "payload": "http://free-cashback-reward.top/download/paytm_update.apk",
            "label": "Malware / Fake APK Redirection"
        },
        {
            "filename": "samples/malicious_upi_scam.png",
            "payload": "upi://pay?pa=lottery.refund.desk@ybl&pn=LotteryClaimDesk&am=5000.00&cu=INR",
            "label": "Malicious UPI Refund Scam"
        }
    ]
    
    for item in samples_data:
        qr = qrcode.QRCode(
            version=2,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(item["payload"])
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(item["filename"])
        print(f"[+] Generated: {item['filename']} -> [{item['label']}]")

if __name__ == "__main__":
    generate_sample_qr_codes()
