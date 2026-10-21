"""Email service placeholder; outbound email and OTP generation are disabled."""

import logging
import os
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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
