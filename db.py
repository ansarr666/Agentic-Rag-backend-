"""Persistence layer for users, conversations, and messages.

Supports both PostgreSQL (when DATABASE_URL is set) and SQLite (fallback).
PostgreSQL is used in production; SQLite is used for local dev without a DB.
"""

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DATABASE_URL = os.getenv("DATABASE_URL", "")

# ---------------------------------------------------------------------------
# SQLite fallback setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "app.db"
DB_PATH = Path(os.getenv("RAG_DB_PATH", str(DEFAULT_DB_PATH)))

_lock = threading.Lock()

USE_POSTGRES = bool(DATABASE_URL)


def configure_from_env() -> None:
    """Select the database after application configuration has loaded .env."""
    global DATABASE_URL, USE_POSTGRES, DB_PATH

    DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
    USE_POSTGRES = bool(DATABASE_URL)
    DB_PATH = Path(os.getenv("RAG_DB_PATH", str(DEFAULT_DB_PATH)))


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _sqlite_connect():
    import sqlite3
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _pg_connect():
    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


@contextmanager
def _get_conn():
    """Yield a connection and commit/close it automatically."""
    if USE_POSTGRES:
        conn = _pg_connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        with _lock, _sqlite_connect() as conn:
            yield conn


def _fetchall(conn, query: str, params=()) -> List[Dict[str, Any]]:
    """Run a SELECT and return list of dicts regardless of DB backend."""
    if USE_POSTGRES:
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
    else:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def _fetchone(conn, query: str, params=()) -> Optional[Dict[str, Any]]:
    if USE_POSTGRES:
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return dict(row) if row else None
    else:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None


def _execute(conn, query: str, params=()):
    if USE_POSTGRES:
        with conn.cursor() as cur:
            cur.execute(query, params)
    else:
        conn.execute(query, params)


def _placeholder() -> str:
    """Return the correct query placeholder for the active DB."""
    return "%s" if USE_POSTGRES else "?"


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables and indexes if they do not exist."""
    if USE_POSTGRES:
        _init_postgres()
    else:
        _init_sqlite()


def _init_sqlite() -> None:
    with _lock, _sqlite_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT,
                role TEXT NOT NULL DEFAULT 'employee',
                provider TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS otp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                code TEXT NOT NULL,
                purpose TEXT NOT NULL DEFAULT 'lead',
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS leads (
                lead_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                name TEXT,
                email TEXT,
                phone TEXT,
                company TEXT,
                industry TEXT,
                project_requirement TEXT,
                desired_solution TEXT,
                existing_technology TEXT,
                timeline TEXT,
                budget TEXT,
                intent_level TEXT NOT NULL DEFAULT 'medium',
                conversation_summary TEXT,
                requested_action TEXT,
                booking_phone TEXT,
                booking_confirmed INTEGER NOT NULL DEFAULT 0,
                confirmation_email_sent INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_otp_email ON otp_codes(email);
            CREATE INDEX IF NOT EXISTS idx_leads_email ON leads(email);
            """
        )


def _init_postgres() -> None:
    with _get_conn() as conn:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT,
                role TEXT NOT NULL DEFAULT 'employee',
                provider TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS otp_codes (
                id SERIAL PRIMARY KEY,
                email TEXT NOT NULL,
                code TEXT NOT NULL,
                purpose TEXT NOT NULL DEFAULT 'lead',
                expires_at TEXT NOT NULL,
                used BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS leads (
                lead_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                name TEXT,
                email TEXT,
                phone TEXT,
                company TEXT,
                industry TEXT,
                project_requirement TEXT,
                desired_solution TEXT,
                existing_technology TEXT,
                timeline TEXT,
                budget TEXT,
                intent_level TEXT NOT NULL DEFAULT 'medium',
                conversation_summary TEXT,
                requested_action TEXT,
                booking_phone TEXT,
                booking_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
                confirmation_email_sent BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_otp_email ON otp_codes(email)",
            "CREATE INDEX IF NOT EXISTS idx_leads_email ON leads(email)",
        ]
        for stmt in statements:
            _execute(conn, stmt)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API — identical interface regardless of backend
# ---------------------------------------------------------------------------

def upsert_user(user_id: str, email: Optional[str], role: str, provider: str) -> None:
    p = _placeholder()
    if USE_POSTGRES:
        query = f"""
            INSERT INTO users (id, email, role, provider, created_at)
            VALUES ({p}, {p}, {p}, {p}, {p})
            ON CONFLICT(id) DO UPDATE SET email=EXCLUDED.email, role=EXCLUDED.role
        """
    else:
        query = f"""
            INSERT INTO users (id, email, role, provider, created_at)
            VALUES ({p}, {p}, {p}, {p}, {p})
            ON CONFLICT(id) DO UPDATE SET email=excluded.email, role=excluded.role
        """
    with _get_conn() as conn:
        _execute(conn, query, (user_id, email, role, provider, _now()))


def create_conversation(user_id: str, title: Optional[str] = None) -> str:
    cid = uuid.uuid4().hex
    now = _now()
    p = _placeholder()
    query = f"INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES ({p}, {p}, {p}, {p}, {p})"
    with _get_conn() as conn:
        _execute(conn, query, (cid, user_id, title or "New Conversation", now, now))
    return cid


def ensure_conversation(conversation_id: str, user_id: str, title: Optional[str] = None) -> None:
    p = _placeholder()
    with _get_conn() as conn:
        row = _fetchone(conn, f"SELECT id FROM conversations WHERE id={p}", (conversation_id,))
        if row is None:
            now = _now()
            _execute(
                conn,
                f"INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES ({p}, {p}, {p}, {p}, {p})",
                (conversation_id, user_id, title or "New Conversation", now, now),
            )


def list_conversations(user_id: str) -> List[Dict[str, Any]]:
    p = _placeholder()
    with _get_conn() as conn:
        return _fetchall(
            conn,
            f"SELECT id, title, created_at, updated_at FROM conversations WHERE user_id={p} ORDER BY updated_at DESC",
            (user_id,),
        )


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    p = _placeholder()
    with _get_conn() as conn:
        return _fetchone(conn, f"SELECT * FROM conversations WHERE id={p}", (conversation_id,))


def add_message(conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    now = _now()
    p = _placeholder()
    with _get_conn() as conn:
        _execute(
            conn,
            f"INSERT INTO messages (conversation_id, role, content, metadata, created_at) VALUES ({p}, {p}, {p}, {p}, {p})",
            (conversation_id, role, content, json.dumps(metadata or {}), now),
        )
        _execute(conn, f"UPDATE conversations SET updated_at={p} WHERE id={p}", (now, conversation_id))


def get_messages(conversation_id: str) -> List[Dict[str, Any]]:
    p = _placeholder()
    with _get_conn() as conn:
        rows = _fetchall(
            conn,
            f"SELECT id, role, content, metadata, created_at FROM messages WHERE conversation_id={p} ORDER BY id ASC",
            (conversation_id,),
        )
    out: List[Dict[str, Any]] = []
    for d in rows:
        try:
            d["metadata"] = json.loads(d["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# OTP functions
# ---------------------------------------------------------------------------

def save_otp(email: str, code: str, purpose: str, expires_at: str) -> None:
    """Store a new OTP, invalidating any previous unused codes for the same email+purpose."""
    p = _placeholder()
    with _get_conn() as conn:
        # Invalidate previous OTPs for this email+purpose
        _execute(
            conn,
            f"UPDATE otp_codes SET used={'TRUE' if USE_POSTGRES else '1'} WHERE email={p} AND purpose={p} AND used={'FALSE' if USE_POSTGRES else '0'}",
            (email, purpose),
        )
        _execute(
            conn,
            f"INSERT INTO otp_codes (email, code, purpose, expires_at, used, created_at) VALUES ({p}, {p}, {p}, {p}, {'FALSE' if USE_POSTGRES else '0'}, {p})",
            (email, code, purpose, expires_at, _now()),
        )


def verify_otp(email: str, code: str, purpose: str) -> tuple:
    """
    Verify an OTP code. Returns (success: bool, error: str).
    Marks the code as used on success.
    """
    p = _placeholder()
    with _get_conn() as conn:
        row = _fetchone(
            conn,
            f"SELECT id, expires_at, used FROM otp_codes WHERE email={p} AND code={p} AND purpose={p} ORDER BY id DESC LIMIT 1",
            (email, code, purpose),
        )
        if not row:
            return False, "Invalid code. Please check and try again."

        used = row["used"]
        # PostgreSQL returns bool, SQLite returns int
        if used is True or used == 1:
            return False, "This code has already been used. Please request a new one."

        expires_at = row["expires_at"]
        now = datetime.now(timezone.utc).isoformat()
        if now > expires_at:
            return False, "This code has expired. Please request a new one."

        # Mark as used
        _execute(
            conn,
            f"UPDATE otp_codes SET used={'TRUE' if USE_POSTGRES else '1'} WHERE id={p}",
            (row["id"],),
        )
        return True, ""


def cleanup_expired_otps() -> None:
    """Delete expired OTP records (call periodically to keep the table clean)."""
    p = _placeholder()
    now = _now()
    with _get_conn() as conn:
        _execute(conn, f"DELETE FROM otp_codes WHERE expires_at < {p}", (now,))


# ---------------------------------------------------------------------------
# Leads functions
# ---------------------------------------------------------------------------

def save_lead(lead: Dict[str, Any]) -> None:
    """Insert or refresh a lead record in the leads table."""
    p = _placeholder()
    now = _now()
    with _get_conn() as conn:
        _execute(conn, f"""
            INSERT INTO leads (
                lead_id, conversation_id, name, email, phone, company,
                industry, project_requirement, desired_solution, existing_technology,
                timeline, budget, intent_level, conversation_summary,
                requested_action, booking_phone, booking_confirmed,
                confirmation_email_sent, created_at, updated_at
            ) VALUES (
                {p},{p},{p},{p},{p},{p},{p},{p},{p},{p},
                {p},{p},{p},{p},{p},{p},
                {'FALSE' if USE_POSTGRES else '0'},
                {'FALSE' if USE_POSTGRES else '0'},
                {p},{p}
            )
            ON CONFLICT(lead_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                name=excluded.name,
                email=excluded.email,
                phone=excluded.phone,
                company=excluded.company,
                industry=excluded.industry,
                project_requirement=excluded.project_requirement,
                desired_solution=excluded.desired_solution,
                existing_technology=excluded.existing_technology,
                timeline=excluded.timeline,
                budget=excluded.budget,
                intent_level=excluded.intent_level,
                conversation_summary=excluded.conversation_summary,
                requested_action=excluded.requested_action,
                updated_at=excluded.updated_at
        """, (
            lead.get("lead_id", ""),
            lead.get("conversation_id", ""),
            lead.get("name", ""),
            lead.get("email", ""),
            lead.get("phone", ""),
            lead.get("company", ""),
            lead.get("industry", ""),
            lead.get("project_requirement", ""),
            lead.get("desired_solution", ""),
            lead.get("existing_technology", ""),
            lead.get("timeline", ""),
            lead.get("budget", ""),
            lead.get("intent_level", "medium"),
            lead.get("conversation_summary", ""),
            lead.get("requested_action", "Talk to an OrionSoft expert"),
            lead.get("phone", ""),
            now,
            now,
        ))


def update_lead_booking(lead_id: str, phone: str, summary: str, requested_action: str) -> None:
    """Update a lead record when a discovery call is booked."""
    p = _placeholder()
    now = _now()
    with _get_conn() as conn:
        _execute(conn, f"""
            UPDATE leads SET
                booking_phone={p},
                conversation_summary={p},
                requested_action={p},
                intent_level='high',
                booking_confirmed={'TRUE' if USE_POSTGRES else '1'},
                confirmation_email_sent={'TRUE' if USE_POSTGRES else '1'},
                updated_at={p}
            WHERE lead_id={p}
        """, (phone, summary, requested_action, now, lead_id))


def get_all_leads() -> List[Dict[str, Any]]:
    """Return all leads ordered by most recent first."""
    with _get_conn() as conn:
        return _fetchall(conn, "SELECT * FROM leads ORDER BY created_at DESC")
