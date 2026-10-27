import os
import sys
from pathlib import Path

# Setup project root and load .env
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
from config import load_config
try:
    load_config()
except Exception as e:
    print(f"[!] Warning loading config: {e}")

from email_service import _send, SALES_NOTIFICATION_EMAIL

def test_mail():
    recipient = os.getenv("SALES_NOTIFICATION_EMAIL", SALES_NOTIFICATION_EMAIL).strip()
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_port = os.getenv("SMTP_PORT", "587")
    email_from = os.getenv("EMAIL_FROM", "")

    print(f"\n--- Checking Email Configuration ---")
    print(f"SMTP Host:    {smtp_host}")
    print(f"SMTP Port:    {smtp_port}")
    print(f"SMTP User:    {smtp_user}")
    print(f"From Address: {email_from}")
    print(f"Recipient:    {recipient}")
    print(f"------------------------------------\n")

    print(f"Attempting to send test email to {recipient}...")
    try:
        success = _send(
            to=recipient,
            subject="OrionSoft Email Integration Test",
            html="<h2>Success!</h2><p>Your Gmail SMTP integration is working perfectly on the Droplet.</p>"
        )
        if success:
            print(f"[SUCCESS] Email sent successfully to {recipient}!")
        else:
            print(f"[FAILED] Email sending returned False.")
    except Exception as e:
        print(f"[ERROR] Exception during send: {e}")

if __name__ == "__main__":
    test_mail()
