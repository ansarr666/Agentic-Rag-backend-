"""Email service using Resend API.

Handles:
- OTP generation and sending (lead capture + admin login)
- Discovery call confirmation emails
- Graceful fallback to logging when RESEND_API_KEY is not set
"""

import logging
import os
import random
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()
EMAIL_FROM = os.getenv("EMAIL_FROM", "OrionSoft Consultations <consultations@orionsofttechnologies.com>").strip()
EMAIL_REPLY_TO = os.getenv("EMAIL_REPLY_TO", "consultations@orionsofttechnologies.com").strip()
OTP_EXPIRE_MINUTES = int(os.getenv("OTP_EXPIRE_MINUTES", "10"))

# ---------------------------------------------------------------------------
# Core send helper
# ---------------------------------------------------------------------------

def _send(to: str, subject: str, html: str) -> bool:
    """Send an email via Resend API, SMTP, or graceful logging fallback."""
    from_addr = os.getenv("EMAIL_FROM", EMAIL_FROM).strip()
    reply_to = os.getenv("EMAIL_REPLY_TO", EMAIL_REPLY_TO).strip()
    resend_key = os.getenv("RESEND_API_KEY", RESEND_API_KEY).strip()
    smtp_host = os.getenv("SMTP_HOST", "").strip()

    # 1. Resend API
    if resend_key:
        try:
            import resend
            resend.api_key = resend_key
            payload = {
                "from": from_addr,
                "to": [to],
                "subject": subject,
                "html": html,
            }
            if reply_to:
                payload["reply_to"] = reply_to
            resend.Emails.send(payload)
            logger.info(f"Email sent via Resend to {to} from {from_addr}")
            return True
        except Exception as e:
            logger.warning(f"Resend send failed for {to}: {e}")

    # 2. Standard SMTP (Gmail, AWS SES, SendGrid, etc.)
    if smtp_host:
        try:
            import smtplib
            import re
            from email.message import EmailMessage

            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = to
            if reply_to:
                msg["Reply-To"] = reply_to

            # Plain text fallback + HTML alternative
            plain_text = re.sub(r"<[^>]+>", " ", html)
            plain_text = re.sub(r"\s+", " ", plain_text).strip()
            msg.set_content(plain_text)
            msg.add_alternative(html, subtype="html")

            port = int(os.getenv("SMTP_PORT", "587"))
            user = os.getenv("SMTP_USER", "").strip()
            password = os.getenv("SMTP_PASSWORD", "").strip()
            use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

            with smtplib.SMTP(smtp_host, port, timeout=10) as server:
                if use_tls:
                    server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(msg)

            logger.info(f"Email sent via SMTP ({smtp_host}) to {to} from {from_addr}")
            return True
        except Exception as e:
            logger.warning(f"SMTP delivery failed for {to}: {e}")

    # 3. Development / Sandbox fallback: Log to stdout/logger
    logger.info(f"[EMAIL LOG] From: {from_addr} | Reply-To: {reply_to} | To: {to} | Subject: {subject}")
    return True


# ---------------------------------------------------------------------------
# OTP utilities
# ---------------------------------------------------------------------------

def generate_otp(length: int = 6) -> str:
    """Generate a cryptographically secure numeric OTP."""
    return "".join(secrets.choice(string.digits) for _ in range(length))


def otp_expiry() -> datetime:
    """Return UTC expiry time for a new OTP."""
    return datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)


# ---------------------------------------------------------------------------
# Email templates
# ---------------------------------------------------------------------------

def _otp_html(otp: str, purpose: str, expire_minutes: int) -> str:
    purpose_label = {
        "lead": "verify your email address",
        "admin": "log in to your admin dashboard",
    }.get(purpose, "verify your identity")

    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; margin: 0; padding: 40px 20px;">
      <div style="max-width: 480px; margin: 0 auto; background: #ffffff; border-radius: 16px;
                  border: 1px solid #e2e8f0; padding: 40px; box-shadow: 0 4px 16px rgba(0,0,0,0.06);">
        <div style="text-align: center; margin-bottom: 32px;">
          <div style="display: inline-flex; width: 48px; height: 48px; border-radius: 12px;
                      background: linear-gradient(135deg, #2563eb, #3b82f6);
                      align-items: center; justify-content: center;
                      font-size: 24px; font-weight: 800; color: white; margin-bottom: 12px;">O</div>
          <h2 style="margin: 0; color: #1e293b; font-size: 22px; font-weight: 700;">OrionSoft Technologies</h2>
        </div>

        <p style="color: #475569; font-size: 15px; margin-bottom: 8px;">
          Use the code below to <strong>{purpose_label}</strong>:
        </p>

        <div style="text-align: center; margin: 28px 0;">
          <div style="display: inline-block; background: #eff6ff; border: 2px solid #bfdbfe;
                      border-radius: 12px; padding: 20px 40px;">
            <span style="font-size: 40px; font-weight: 800; letter-spacing: 10px;
                         color: #2563eb; font-family: monospace;">{otp}</span>
          </div>
        </div>

        <p style="color: #94a3b8; font-size: 13px; text-align: center; margin-top: 0;">
          This code expires in <strong>{expire_minutes} minutes</strong>.
          If you did not request this, you can safely ignore this email.
        </p>

        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 28px 0;">
        <p style="color: #cbd5e1; font-size: 12px; text-align: center; margin: 0;">
          &copy; 2026 OrionSoft Technologies Inc.
        </p>
      </div>
    </body>
    </html>
    """


def _confirmation_html(name: str, phone: str, summary: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; margin: 0; padding: 40px 20px;">
      <div style="max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 16px;
                  border: 1px solid #e2e8f0; padding: 40px; box-shadow: 0 4px 16px rgba(0,0,0,0.06);">
        <div style="text-align: center; margin-bottom: 32px;">
          <div style="display: inline-block; width: 48px; height: 48px; border-radius: 12px;
                      background: linear-gradient(135deg, #2563eb, #3b82f6);
                      line-height: 48px; text-align: center;
                      font-size: 24px; font-weight: 800; color: white; margin-bottom: 12px;">O</div>
          <h2 style="margin: 0; color: #1e293b; font-size: 22px; font-weight: 700;">Discovery Call Confirmed</h2>
        </div>

        <p style="color: #475569; font-size: 15px;">Hi <strong>{name or 'there'}</strong>,</p>
        <p style="color: #475569; font-size: 15px;">
          Thank you for reaching out to <strong>OrionSoft Technologies</strong>.
          Your discovery call request has been received and confirmed.
        </p>

        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
                    padding: 20px 24px; margin: 24px 0;">
          <p style="margin: 0 0 8px; color: #166534; font-size: 14px; font-weight: 600;">Your Details</p>
          <p style="margin: 4px 0; color: #15803d; font-size: 14px;">📞 Phone: {phone}</p>
          <p style="margin: 4px 0; color: #15803d; font-size: 14px;">💼 Topic: {summary or 'Enterprise Digital Solutions'}</p>
          <p style="margin: 4px 0; color: #15803d; font-size: 14px;">✅ Status: Confirmed &amp; Queued</p>
        </div>

        <p style="color: #475569; font-size: 14px;">
          A member of our solutions engineering team will reach out to you at
          <strong>{phone}</strong> shortly. Feel free to reply to this email
          with any additional details.
        </p>

        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 28px 0;">
        <p style="color: #cbd5e1; font-size: 12px; text-align: center; margin: 0;">
          &copy; 2026 OrionSoft Technologies Inc. &bull; All rights reserved.
        </p>
      </div>
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_otp(to: str, otp: str, purpose: str = "lead") -> bool:
    """Send an OTP email. purpose: 'lead' or 'admin'."""
    subject_map = {
        "lead": "Your OrionSoft verification code",
        "admin": "Your OrionSoft admin login code",
    }
    subject = subject_map.get(purpose, "Your OrionSoft verification code")
    html = _otp_html(otp, purpose, OTP_EXPIRE_MINUTES)
    return _send(to, subject, html)


def send_confirmation(to: str, name: str, phone: str, summary: str = "") -> bool:
    """Send a branded discovery call confirmation email."""
    if not to:
        return False
    html = _confirmation_html(name, phone, summary)
    return _send(to, "Discovery Call Confirmed - OrionSoft Technologies", html)
