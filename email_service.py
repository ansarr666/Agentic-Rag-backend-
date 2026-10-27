"""Email service for transactional emails, calendar invites, and real-time sales alerts.

Handles:
- 1-click Google Calendar & Outlook links + .ics invite generation
- Real-time instant lead notifications dispatched to Rahul / sales inbox
- Visitor discovery call confirmation emails
- Resend API delivery with SMTP and graceful logging fallback
"""

import logging
import os
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

EMAIL_FROM = os.getenv("EMAIL_FROM", "OrionSoft AI <consultations@orionsofttechnologies.com>").strip()
EMAIL_REPLY_TO = os.getenv("EMAIL_REPLY_TO", "consultations@orionsofttechnologies.com").strip()
SALES_NOTIFICATION_EMAIL = os.getenv("SALES_NOTIFICATION_EMAIL", "rahul@orionsofttechnologies.com").strip()
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()


def _calendar_start(start_time: Optional[datetime]) -> datetime:
    if start_time is None:
        base = datetime.now(timezone.utc) + timedelta(days=1)
        start_time = base.replace(hour=14, minute=0, second=0, microsecond=0)
    elif start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    return start_time.astimezone(timezone.utc)


def generate_calendar_links(
    title: str = "OrionSoft Discovery Call",
    summary: str = "",
    start_time: Optional[datetime] = None,
    duration_minutes: int = 30,
) -> Dict[str, Any]:
    """Generate Google Calendar and Outlook links for a discovery call."""
    start_time = _calendar_start(start_time)
    end_time = start_time + timedelta(minutes=duration_minutes)
    details = (
        "Discovery call with OrionSoft Technologies solutions engineering team.\n\n"
        f"Topic: {summary or 'Enterprise Digital Solutions'}\n\n"
        "Our team will connect with you at your designated contact number."
    )
    location = "Phone Call / Online"
    start_iso = start_time.strftime("%Y%m%dT%H%M%SZ")
    end_iso = end_time.strftime("%Y%m%dT%H%M%SZ")

    google_url = "https://calendar.google.com/calendar/render?" + urllib.parse.urlencode({
        "action": "TEMPLATE",
        "text": title,
        "dates": f"{start_iso}/{end_iso}",
        "details": details,
        "location": location,
    })
    outlook_url = "https://outlook.live.com/calendar/0/action/compose?" + urllib.parse.urlencode({
        "path": "/calendar/action/compose",
        "rru": "addevent",
        "subject": title,
        "body": details,
        "startdt": start_time.isoformat(),
        "enddt": end_time.isoformat(),
        "location": location,
    })

    return {
        "google": google_url,
        "outlook": outlook_url,
        "booking_url": os.getenv("CALENDAR_BOOKING_URL", "").strip() or None,
        "start_iso": start_time.isoformat(),
        "end_iso": end_time.isoformat(),
        "formatted_date": start_time.strftime("%A, %B %d, %Y at %I:%M %p UTC"),
    }


def generate_ics(
    title: str = "OrionSoft Discovery Call",
    summary: str = "",
    start_time: Optional[datetime] = None,
    duration_minutes: int = 30,
) -> str:
    """Generate an RFC 5545 iCalendar invite."""
    start_time = _calendar_start(start_time)
    end_time = start_time + timedelta(minutes=duration_minutes)
    dtstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def escape(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\r\n", "\\n")
            .replace("\r", "\\n")
            .replace("\n", "\\n")
        )

    desc = summary or "Discovery call with OrionSoft Technologies"
    return (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//OrionSoft Technologies//Discovery Call//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:REQUEST\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:discovery-{secrets.token_hex(8)}@orionsofttechnologies.com\r\n"
        f"DTSTAMP:{dtstamp}\r\n"
        f"DTSTART:{start_time.strftime('%Y%m%dT%H%M%SZ')}\r\n"
        f"DTEND:{end_time.strftime('%Y%m%dT%H%M%SZ')}\r\n"
        f"SUMMARY:{escape(title)}\r\n"
        f"DESCRIPTION:{escape(desc)}\r\n"
        "LOCATION:Phone Call / Online\r\n"
        "STATUS:CONFIRMED\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )


# ---------------------------------------------------------------------------
# Core Send Helper (Resend / SMTP / Graceful Log)
# ---------------------------------------------------------------------------

def _send(to: str, subject: str, html: str) -> bool:
    """Send an email via Resend API, SMTP, or log gracefully."""
    from_addr = os.getenv("EMAIL_FROM", EMAIL_FROM).strip()
    reply_to = os.getenv("EMAIL_REPLY_TO", EMAIL_REPLY_TO).strip()
    resend_key = os.getenv("RESEND_API_KEY", RESEND_API_KEY).strip()
    smtp_host = os.getenv("SMTP_HOST", "").strip()

    # 1. Resend API
    if resend_key:
        try:
            import resend
            resend.api_key = resend_key
            to_list = [addr.strip() for addr in to.split(",") if addr.strip()] if isinstance(to, str) else [to]
            payload = {
                "from": from_addr,
                "to": to_list,
                "subject": subject,
                "html": html,
            }
            if reply_to:
                payload["reply_to"] = reply_to
            resend.Emails.send(payload)
            logger.info("Email sent via Resend to %s: %s", to, subject)
            return True
        except Exception as e:
            logger.warning("Resend send failed for %s: %s", to, e)
            if "not verified" in str(e).lower() and "onboarding@resend.dev" not in from_addr:
                try:
                    payload["from"] = "OrionSoft AI <onboarding@resend.dev>"
                    resend.Emails.send(payload)
                    logger.info("Email sent via Resend fallback (onboarding@resend.dev) to %s", to)
                    return True
                except Exception as fallback_err:
                    logger.warning("Resend fallback failed for %s: %s", to, fallback_err)

    # 2. SMTP fallback
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

            logger.info("Email sent via SMTP to %s: %s", to, subject)
            return True
        except Exception as e:
            logger.warning("SMTP send failed for %s: %s", to, e)

    # 3. Development / Sandbox fallback
    logger.info("[EMAIL LOG] To: %s | Subject: %s", to, subject)
    return True


# ---------------------------------------------------------------------------
# Sales Team Real-Time Alert
# ---------------------------------------------------------------------------

def send_sales_alert(lead: Dict[str, Any], event_type: str = "new_lead") -> bool:
    """Send an instant high-priority lead notification directly to Rahul / sales inbox."""
    recipient = os.getenv("SALES_NOTIFICATION_EMAIL", SALES_NOTIFICATION_EMAIL).strip()
    if not recipient:
        return False

    name = lead.get("name") or "Website Visitor"
    email = lead.get("email") or "Not provided"
    phone = lead.get("booking_phone") or lead.get("phone") or "Not provided"
    company = lead.get("company") or "Not specified"
    requirement = lead.get("project_requirement") or lead.get("message") or ""
    summary = lead.get("conversation_summary") or ""
    intent = (lead.get("intent_level") or "high").upper()
    action = lead.get("requested_action") or "Consultation Inquiry"

    is_booking = event_type == "discovery_call" or bool(lead.get("booking_confirmed"))
    badge_color = "#16a34a" if is_booking else "#2563eb"
    badge_label = "DISCOVERY CALL BOOKED" if is_booking else "NEW QUALIFIED LEAD"
    subject = f"{'📅 Call Booked' if is_booking else '🔥 New Lead'}: {name} ({company if company != 'Not specified' else email})"

    requirement_block = ""
    if requirement:
        requirement_block = f"""
        <div style="background: #f8fafc; border-left: 4px solid #2563eb; padding: 14px 16px; border-radius: 4px; margin-bottom: 20px;">
          <p style="margin: 0 0 6px; font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">Project Requirement / Inquiry:</p>
          <p style="margin: 0; font-size: 14px; color: #1e293b; line-height: 1.5;">{requirement}</p>
        </div>
        """

    summary_block = ""
    if summary:
        summary_block = f"""
        <div style="background: #f0fdf4; border-left: 4px solid #16a34a; padding: 14px 16px; border-radius: 4px; margin-bottom: 20px;">
          <p style="margin: 0 0 6px; font-size: 11px; font-weight: 700; color: #166534; text-transform: uppercase;">AI Chat Summary:</p>
          <p style="margin: 0; font-size: 13px; color: #1e293b; line-height: 1.5;">{summary}</p>
        </div>
        """

    call_button = ""
    if phone and phone != "Not provided":
        call_button = f"""
        <a href="tel:{phone}" style="display: inline-block; background: #0f172a; color: #ffffff; text-decoration: none; padding: 10px 18px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-left: 8px;">
          📞 Call {phone}
        </a>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 24px;">
      <div style="max-width: 580px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.06);">
        <div style="background: {badge_color}; padding: 20px 24px; color: #ffffff;">
          <span style="font-size: 11px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; background: rgba(255,255,255,0.25); padding: 4px 8px; border-radius: 4px;">{badge_label}</span>
          <h2 style="margin: 10px 0 0; font-size: 22px; font-weight: 700; color: #ffffff;">{name}</h2>
          <p style="margin: 4px 0 0; font-size: 14px; opacity: 0.95;">Company: <strong>{company}</strong></p>
        </div>

        <div style="padding: 24px;">
          <table style="width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 20px;">
            <tr>
              <td style="padding: 8px 0; color: #64748b; width: 130px; font-weight: 600;">Email:</td>
              <td style="padding: 8px 0; color: #0f172a;"><a href="mailto:{email}" style="color: #2563eb; text-decoration: none; font-weight: 600;">{email}</a></td>
            </tr>
            <tr>
              <td style="padding: 8px 0; color: #64748b; font-weight: 600;">Phone:</td>
              <td style="padding: 8px 0; color: #0f172a;"><strong style="color: #0f172a;">{phone}</strong></td>
            </tr>
            <tr>
              <td style="padding: 8px 0; color: #64748b; font-weight: 600;">Intent Level:</td>
              <td style="padding: 8px 0; font-weight: 700; color: {'#dc2626' if intent == 'HIGH' else '#2563eb'};">{intent}</td>
            </tr>
            <tr>
              <td style="padding: 8px 0; color: #64748b; font-weight: 600;">Requested Action:</td>
              <td style="padding: 8px 0; color: #0f172a;">{action}</td>
            </tr>
          </table>

          {requirement_block}
          {summary_block}

          <div style="margin-top: 24px; padding-top: 18px; border-top: 1px solid #e2e8f0;">
            <a href="mailto:{email}?subject=Regarding%20your%20consultation%20with%20OrionSoft" 
               style="display: inline-block; background: #2563eb; color: #ffffff; text-decoration: none; padding: 10px 18px; border-radius: 6px; font-size: 13px; font-weight: 600;">
               ✉️ Reply via Email
            </a>
            {call_button}
          </div>
        </div>

        <div style="background: #f8fafc; padding: 12px 24px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #94a3b8; text-align: center;">
          OrionSoft AI Real-time Lead Dispatcher &bull; Delivered to {recipient}
        </div>
      </div>
    </body>
    </html>
    """
    return _send(recipient, subject, html)


# ---------------------------------------------------------------------------
# Visitor Confirmation (with Calendar Buttons)
# ---------------------------------------------------------------------------

def send_confirmation(
    to: str,
    name: str,
    phone: str,
    summary: str = "",
    calendar_links: Optional[Dict[str, Any]] = None,
) -> bool:
    """Send branded confirmation email to visitor with 1-click calendar links."""
    if not to:
        return False
    cal = calendar_links or generate_calendar_links(summary=summary)
    google_link = cal.get("google", "#")
    outlook_link = cal.get("outlook", "#")
    formatted_date = cal.get("formatted_date", "Next Business Day at 2:00 PM UTC")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; margin: 0; padding: 40px 20px;">
      <div style="max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 16px; border: 1px solid #e2e8f0; padding: 40px; box-shadow: 0 4px 16px rgba(0,0,0,0.06);">
        <div style="text-align: center; margin-bottom: 28px;">
          <div style="display: inline-block; width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, #2563eb, #3b82f6); line-height: 48px; text-align: center; font-size: 24px; font-weight: 800; color: white; margin-bottom: 12px;">O</div>
          <h2 style="margin: 0; color: #1e293b; font-size: 22px; font-weight: 700;">Discovery Call Confirmed</h2>
        </div>
        <p style="color: #475569; font-size: 15px;">Hi <strong>{name or 'there'}</strong>,</p>
        <p style="color: #475569; font-size: 15px;">
          Thank you for reaching out to <strong>OrionSoft Technologies</strong>. Your discovery call request has been confirmed.
        </p>
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 18px 22px; margin: 20px 0;">
          <p style="margin: 0 0 6px; color: #166534; font-size: 13px; font-weight: 700;">Your Booking Details</p>
          <p style="margin: 4px 0; color: #15803d; font-size: 14px;">📞 Phone: {phone}</p>
          <p style="margin: 4px 0; color: #15803d; font-size: 14px;">💼 Topic: {summary or 'Enterprise Digital Solutions'}</p>
          <p style="margin: 4px 0; color: #15803d; font-size: 14px;">📅 Window: {formatted_date}</p>
        </div>
        <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 20px; text-align: center; margin: 20px 0;">
          <p style="margin: 0 0 12px; color: #1e40af; font-size: 13px; font-weight: 700;">Add this call to your calendar:</p>
          <a href="{google_link}" target="_blank" style="display: inline-block; background: #2563eb; color: #ffffff; text-decoration: none; padding: 9px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; margin: 3px;">+ Google Calendar</a>
          <a href="{outlook_link}" target="_blank" style="display: inline-block; background: #0284c7; color: #ffffff; text-decoration: none; padding: 9px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; margin: 3px;">+ Outlook / 365</a>
        </div>
        <p style="color: #475569; font-size: 14px;">
          A member of our solutions engineering team will reach out to you at <strong>{phone}</strong>.
        </p>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;">
        <p style="color: #94a3b8; font-size: 12px; text-align: center; margin: 0;">&copy; 2026 OrionSoft Technologies Inc.</p>
      </div>
    </body>
    </html>
    """
    return _send(to, "Discovery Call Confirmed - OrionSoft Technologies", html)
