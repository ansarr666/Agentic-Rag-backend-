import os
import sys
import json
import time
import uuid
import logging
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock
from flask import Flask, request, jsonify, g, has_request_context, redirect
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

# Setup project root and ensure UTF-8 encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from pipeline import RAGPipeline, IngestionPipeline
from evaluation.run_eval import run_decision_evaluation, run_retrieval_evaluation
import db

# ── Logging (one JSON object per line on stdout) ─────────
class JsonFormatter(logging.Formatter):
    """Adds request_id/endpoint inside a request, plus any method/status/latency_ms passed via extra."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if has_request_context():
            entry["request_id"] = g.get("request_id")
            entry["endpoint"] = request.path
        for key in ("method", "status", "latency_ms"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        if record.exc_info:
            entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(JsonFormatter())
# force=True: modules imported above may already have configured the root logger
logging.basicConfig(level=logging.INFO, handlers=[_log_handler], force=True)
logger = logging.getLogger(__name__)

# ── Flask App ────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
START_TIME = time.monotonic()

# ── Load RAG Pipeline (once at startup) ──────────────────
config = load_config()
db.configure_from_env()
rag = RAGPipeline(config, project_root=PROJECT_ROOT)
logger.info("Agentic RAG Pipeline initialized and ready for queries.")

try:
    db.init_db()
    logger.info("%s persistence initialized.", "PostgreSQL" if db.USE_POSTGRES else "SQLite")
except Exception as e:
    logger.warning("Could not initialize database: %s", e)

# ── Production hardening (env is read after load_config(), which loads .env) ──
SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=os.getenv("FLASK_ENV", "production"),
        release=os.getenv("APP_VERSION", "1.0.0"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE") or "0"),
        send_default_pii=False,
    )
    logger.info("Sentry error reporting enabled.")

# Behind a load balancer, trust X-Forwarded-For from this many proxies so limits see real client IPs
TRUSTED_PROXY_COUNT = int(os.getenv("TRUSTED_PROXY_COUNT") or "0")
if TRUSTED_PROXY_COUNT:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUSTED_PROXY_COUNT, x_proto=TRUSTED_PROXY_COUNT)

# CORS: only the listed origins; for /api/public/* allow all origins by default if unset
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})
else:
    CORS(app, resources={r"/api/public/*": {"origins": "*"}})


@app.before_request
def _start_request():
    g.request_id = (request.headers.get("X-Request-ID") or uuid.uuid4().hex)[:64]
    g.start_time = time.perf_counter()


@app.after_request
def _log_request(response):
    response.headers["X-Request-ID"] = g.get("request_id", "")
    if request.path.startswith("/api/") and request.path != "/api/health":
        latency_ms = round((time.perf_counter() - g.get("start_time", time.perf_counter())) * 1000, 2)
        logger.info("request", extra={"method": request.method, "status": response.status_code, "latency_ms": latency_ms})
    return response


# Rate limits apply to /api/* only (pages, assets and the health check are never limited).
# memory:// counts per gunicorn worker; set RATE_LIMIT_STORAGE_URI=redis://... to share counts.
limiter_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() not in ("false", "0", "no")
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[os.getenv("RATE_LIMIT_DEFAULT", "100/hour")],
    default_limits_exempt_when=lambda: not request.path.startswith("/api/") or request.path == "/api/health",
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
    headers_enabled=True,
    enabled=limiter_enabled,
)


@app.errorhandler(429)
def _rate_limited(e):
    return jsonify({"error": "Too many requests. Please slow down and try again shortly.",
                    "limit": str(e.description)}), 429


@app.errorhandler(Exception)
def _unhandled_error(e):
    if isinstance(e, HTTPException):
        return e
    logger.exception("Unhandled server error")
    return _server_error()


def _server_error():
    """Generic 500 body: details stay in the logs, the client gets a request_id to quote."""
    return jsonify({"error": "Internal server error.", "request_id": g.get("request_id")}), 500

# ── API Key Security (Centralized via auth.py) ───────────
from auth import check_auth, require_admin_key, get_api_key


# In-memory per-IP rate limiter for public endpoints
_public_rate_limits = defaultdict(list)
_leads_lock = Lock()
_conversations_lock = Lock()

def public_rate_limited(limit: int = 10, window: int = 60) -> bool:
    ip = request.remote_addr or "unknown"
    now = time.time()
    _public_rate_limits[ip] = [t for t in _public_rate_limits[ip] if now - t < window]
    if len(_public_rate_limits[ip]) >= limit:
        return True
    _public_rate_limits[ip].append(now)
    return False

# Registered tool definitions in memory
REGISTERED_TOOLS = [
    {
        "id": "tool-knowledge-search",
        "name": "Internal Knowledge Search",
        "code": "knowledge_search",
        "category": "Retrieval",
        "description": "Combines dense vectors and sparse BM25 with Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking.",
        "enabled": True,
        "accessScope": "All Employees",
        "executions24h": 1420,
        "avgLatencyMs": 48,
        "status": "Online"
    },
    {
        "id": "tool-google-drive",
        "name": "Google Drive Knowledge Connector",
        "code": "google_drive",
        "category": "Integration",
        "description": "Securely discovers, incrementally synchronizes, and searches approved company Google Drive documents.",
        "enabled": True,
        "accessScope": "All Employees",
        "executions24h": 312,
        "avgLatencyMs": 62,
        "status": "Online"
    },
    {
        "id": "tool-calculator",
        "name": "Deterministic AST Calculator",
        "code": "calculator",
        "category": "Analysis",
        "description": "Safe AST-evaluated arithmetic, percentage reductions, and financial calculations with zero hallucination.",
        "enabled": True,
        "accessScope": "All Employees",
        "executions24h": 184,
        "avgLatencyMs": 4,
        "status": "Online"
    },
    {
        "id": "tool-web-search",
        "name": "Authoritative Web Search",
        "code": "web_search",
        "category": "Retrieval",
        "description": "Controlled fallback for public technical releases and external official documentation with domain authority gating.",
        "enabled": True,
        "accessScope": "Elevated Roles",
        "executions24h": 68,
        "avgLatencyMs": 195,
        "status": "Online"
    },
    {
        "id": "tool-database-query",
        "name": "Enterprise PostgreSQL Connector",
        "code": "sql_read_only_agent",
        "category": "Data",
        "description": "Executes parameterized read-only queries with SQL injection prevention and data masking.",
        "enabled": True,
        "accessScope": "Elevated Roles",
        "executions24h": 94,
        "avgLatencyMs": 56,
        "status": "Online"
    }
]



# ── Public Prospect Endpoints ───────────────────────────

@app.route("/api/public/chat", methods=["POST"])
def public_chat():
    if public_rate_limited():
        return jsonify({"error": "Too many requests. Please try again later."}), 429

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        resp = rag.query(question, user_role="employee", persona="prospect")
        return jsonify({
            "answer": resp.answer,
            "suggested_queries": resp.suggested_queries or []
        })
    except Exception as e:
        logger.error(f"Public chat failed: {e}", exc_info=True)
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# ── Email Validation & Notification Services ─────────────────────

DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com", "tempmail.com", "guerrillamail.com", "10minutemail.com",
    "sharklasers.com", "yopmail.com", "trashmail.com", "getairmail.com",
    "dispostable.com", "mytemp.email", "throwawaymail.com", "fakeinbox.com",
    "temp-mail.org", "mohmal.com", "generator.email", "emailondeck.com",
    "inboxkitten.com", "crazymailing.com", "tempail.com", "armyspy.com",
    "cuvox.de", "dayrep.com", "fleckens.hu", "gustr.com", "jourrapide.com",
    "rhyta.com", "superrito.com", "teleworm.us", "burnermail.io", "trashmail.net",
    "mail7.io", "nada.ltd", "dropmail.me", "tempmailaddress.com", "tempmail.plus",
    "fake-box.com", "temp-inbox.com", "10mail.org", "fakemailgenerator.com", "mohmal.in",
    "trashmail.org", "maildrop.cc", "disposablemail.com", "throwawayemailaddress.com",
    "discard.email", "spambog.com", "spam4.me", "trash-mail.at", "mytempmail.com",
    "getnada.com", "inboxbear.com", "mailnesia.com", "guerrillamailblock.com"
}

# Personal email domains that are explicitly allowed (fast-pathed without DNS check)
ALLOWED_PERSONAL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "yahoo.fr", "ymail.com",
    "hotmail.com", "hotmail.co.uk", "outlook.com", "outlook.de", "live.com", "msn.com",
    "icloud.com", "me.com", "mac.com",
    "aol.com", "zoho.com", "zohomail.com",
    "proton.me", "protonmail.com", "pm.me",
    "mail.com", "gmx.com", "gmx.net", "gmx.de", "web.de", "fastmail.com", "t-online.de", "rediffmail.com"
}

# Common domain typos mapped to their intended domain
COMMON_DOMAIN_TYPOS = {
    "gmai.com": "gmail.com",
    "gmaill.com": "gmail.com",
    "gamil.com": "gmail.com",
    "gmial.com": "gmail.com",
    "gmal.com": "gmail.com",
    "yaho.com": "yahoo.com",
    "yahooo.com": "yahoo.com",
    "hotmial.com": "hotmail.com",
    "hotmai.com": "hotmail.com",
    "outloo.com": "outlook.com",
    "outlok.com": "outlook.com",
    "iclou.com": "icloud.com",
    "protn.me": "proton.me",
    "protomail.com": "protonmail.com",
}


def check_domain_mail_servers(domain: str) -> tuple[bool, str]:
    """Verify that domain has valid MX or A DNS records to receive mail."""
    if domain in ALLOWED_PERSONAL_DOMAINS:
        return True, ""
    try:
        import dns.resolver
        try:
            records = dns.resolver.resolve(domain, "MX", lifetime=3.0)
            if records and len(records) > 0:
                return True, ""
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            try:
                dns.resolver.resolve(domain, "A", lifetime=2.0)
                return True, ""
            except Exception:
                return False, f"The domain '@{domain}' does not exist or has no active mail server."
        except (dns.resolver.LifetimeTimeout, dns.resolver.NoNameservers):
            pass
    except ImportError:
        try:
            import socket
            socket.gethostbyname(domain)
            return True, ""
        except (socket.gaierror, socket.herror, OSError):
            return False, f"The domain '@{domain}' could not be reached. Please check for typos."
    except Exception:
        pass
    return True, ""


def validate_work_email(email: str) -> tuple[bool, str]:
    """
    Validates email format, checks for typos, blocks disposable fake domains,
    and checks domain mail server existence.
    Returns (is_valid, error_message).
    """
    if not email or len(email) > 254:
        return False, "Please enter a valid email address."

    # 1. Syntax check
    if not re.fullmatch(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
        return False, "Please provide a valid email address (e.g. name@company.com)."

    parts = email.split("@")
    if len(parts) != 2:
        return False, "Please provide a valid email address."
    prefix = parts[0].lower()
    domain = parts[1].lower()

    # 2. Obvious typo check
    if domain in COMMON_DOMAIN_TYPOS:
        suggested = COMMON_DOMAIN_TYPOS[domain]
        return False, f"Did you mean @{suggested}? Please check your email for typos."

    # 3. Disposable / temporary email check
    if domain in DISPOSABLE_EMAIL_DOMAINS:
        return False, "Temporary/disposable email addresses are not accepted."

    # 4. Reject obvious dummy test inputs
    if prefix in {"test", "asdf", "fake", "none", "nobody", "dummy"} and domain in {"test.com", "example.com", "fake.com"}:
        return False, "Please provide a real email address."

    # 5. Domain mail server check
    return check_domain_mail_servers(domain)




@app.route("/api/public/lead", methods=["POST"])
def public_lead():
    if public_rate_limited(limit=10, window=60):
        return jsonify({"error": "Too many requests."}), 429

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Please provide valid contact details."}), 400

    def clean(field, limit=500):
        return str(data.get(field) or "").strip()[:limit]

    name = clean("name", 120)
    email = clean("email", 254).lower()
    phone = clean("phone", 40)
    country_code = clean("country_code", 5)
    company = clean("company", 200)

    if not name:
        return jsonify({"error": "Please provide your full name."}), 400
    is_valid, err_msg = validate_work_email(email)
    if not is_valid:
        return jsonify({"error": err_msg}), 400
    if "country_code" in data:
        if not re.fullmatch(r"\+[1-9]\d{0,3}", country_code) or not re.fullmatch(r"\d{10}", phone):
            return jsonify({"error": "Please select a country code and provide a 10-digit phone number."}), 400
        phone = f"{country_code}{phone}"
    elif phone and len(re.sub(r"\D", "", phone)) < 7:
        return jsonify({"error": "Please provide a valid phone number."}), 400

    existing_lead_id = clean("lead_id", 64)
    lead_id = existing_lead_id or uuid.uuid4().hex
    conversation_id = clean("conversation_id", 64) or uuid.uuid4().hex

    leads_path = PROJECT_ROOT / "data" / "leads.json"
    now = datetime.now(timezone.utc).isoformat()

    with _leads_lock:
        leads = []
        if leads_path.exists():
            try:
                loaded = json.loads(leads_path.read_text(encoding="utf-8"))
                leads = loaded if isinstance(loaded, list) else []
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read existing lead data; starting a new lead list.")

        # Update an existing lead (e.g., handoff request) rather than duplicating it
        if existing_lead_id:
            for lead in leads:
                if lead.get("lead_id") == existing_lead_id:
                    lead["name"] = name
                    lead["email"] = email
                    lead["phone"] = phone
                    lead["company"] = company or lead.get("company", "")
                    lead["project_requirement"] = clean("project_requirement", 3000) or lead.get("project_requirement", "")
                    lead["conversation_summary"] = clean("conversation_summary", 5000) or lead.get("conversation_summary", "")
                    lead["requested_action"] = clean("requested_action", 200) or lead.get("requested_action", "Talk to an OrionSoft expert")
                    lead["intent_level"] = clean("intent_level", 40) or lead.get("intent_level", "medium")
                    lead["updated_at"] = now
                    leads_path.write_text(json.dumps(leads, indent=2, ensure_ascii=False), encoding="utf-8")
                    # Sync updated lead to Supabase
                    try:
                        db.save_lead(lead)
                    except Exception as db_err:
                        logger.warning(f"Supabase lead update failed for {existing_lead_id}: {db_err}")
                    return jsonify({
                        "success": True,
                        "lead_id": existing_lead_id,
                        "conversation_id": lead.get("conversation_id", conversation_id)
                    })

        lead_record = {
            "lead_id": lead_id,
            "conversation_id": conversation_id,
            "name": name,
            "email": email,
            "phone": phone,
            "company": company,
            "industry": clean("industry", 200),
            "project_requirement": clean("project_requirement", 3000),
            "desired_solution": clean("desired_solution", 500),
            "existing_technology": clean("existing_technology", 1000),
            "timeline": clean("timeline", 120),
            "budget": clean("budget", 120),
            "intent_level": clean("intent_level", 40) or "medium",
            "conversation_summary": clean("conversation_summary", 5000),
            "requested_action": clean("requested_action", 200) or "Talk to an OrionSoft expert",
            "timestamp": now,
            "message": clean("project_requirement", 3000)
        }
        leads.append(lead_record)
        leads_path.write_text(json.dumps(leads, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save new lead to Supabase
    try:
        db.save_lead(lead_record)
    except Exception as db_err:
        logger.warning(f"Supabase lead save failed for {lead_id}: {db_err}")

    return jsonify({"success": True, "lead_id": lead_id, "conversation_id": conversation_id})


INTENT_SIGNALS = [
    "i want to build", "i want to automate", "i want to create", "i want to develop",
    "i want a", "we want to", "we need", "can you develop", "can you build",
    "can you integrate", "can you help us build", "can you help me build",
    "how much would", "how much does", "how much will",
    "we are looking for", "we're looking for", "we are looking to", "we're looking to",
    "i'm looking to", "i am looking to", "looking to build", "looking to automate",
    "we need this for", "can someone from your team", "i want a quote",
    "i'd like a consultation", "i would like a consultation", "get a quote",
    "talk to", "speak with", "contact me", "book a", "schedule a", "hire",
    "build me", "proposal", "estimate", "starting a project", "our company needs",
    "we are a company",
]

def detect_project_intent(question: str) -> str:
    """Lightweight high-intent signal detection for the public conversation."""
    q = question.lower()
    if any(sig in q for sig in INTENT_SIGNALS):
        return "high"
    return "normal"


def _persist_conversation(lead_id: str, conversation_id: str, name: str, question: str, answer: str) -> None:
    """Append a visitor/assistant exchange to the conversation store keyed by conversation_id."""
    if not conversation_id:
        return
    path = PROJECT_ROOT / "data" / "conversations.json"
    now = datetime.now(timezone.utc).isoformat()
    with _conversations_lock:
        store = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                store = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read existing conversation data; starting fresh.")
        entry = store.get(conversation_id, {"lead_id": lead_id, "name": name, "messages": []})
        entry["lead_id"] = lead_id or entry.get("lead_id", "")
        entry["name"] = name or entry.get("name", "")
        entry["messages"].append({"role": "user", "content": question, "timestamp": now})
        entry["messages"].append({"role": "assistant", "content": answer, "timestamp": now})
        store[conversation_id] = entry
        path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")

    # Dual-write to the configured database (Supabase PostgreSQL or local SQLite).
    try:
        user_id = lead_id or f"anon_{conversation_id[:12]}"
        db.upsert_user(user_id=user_id, email=None, role="guest", provider="lead" if lead_id else "anonymous")
        title = f"Chat with {name}" if name else "Visitor Chat"
        db.ensure_conversation(conversation_id=conversation_id, user_id=user_id, title=title)
        db.add_message(conversation_id, role="user", content=question, metadata={"lead_id": lead_id, "name": name})
        db.add_message(conversation_id, role="assistant", content=answer, metadata={"lead_id": lead_id, "name": name})
    except Exception as db_err:
        logger.warning(f"Database write failed for conversation {conversation_id}: {db_err}")


@app.route("/api/public/conversation", methods=["POST"])
def public_conversation():
    if public_rate_limited(limit=30, window=60):
        return jsonify({"error": "Too many requests. Please try again in a moment."}), 429

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a message."}), 400
    if len(question) > 2000:
        question = question[:2000]

    raw_history = data.get("history") or []
    history = []
    if isinstance(raw_history, list):
        for turn in raw_history:
            if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
                history.append({
                    "role": turn.get("role"),
                    "content": (turn.get("content") or "")[:2000]
                })
    history = history[-12:]

    lead_id = str(data.get("lead_id") or "").strip()[:64]
    conversation_id = str(data.get("conversation_id") or "").strip()[:64]
    name = str(data.get("name") or "").strip()[:120]

    try:
        result = rag.conversational_query(question, history=history, visitor_name=name or None)
    except Exception as e:
        logger.error(f"Conversational query failed: {e}", exc_info=True)
        return jsonify({"error": "Something went wrong. Please try again."}), 500

    answer = result.get("answer") or ""
    suggestions = (result.get("suggested_questions") or [])[:4]
    intent = detect_project_intent(question)

    _persist_conversation(lead_id, conversation_id, name, question, answer)

    return jsonify({
        "answer": answer,
        "suggested_questions": suggestions,
        "intent": intent,
    })


@app.route("/api/public/intent", methods=["POST"])
def public_intent():
    """Record a high-intent handoff request against an existing lead."""
    if public_rate_limited(limit=10, window=60):
        return jsonify({"error": "Too many requests."}), 429

    data = request.get_json(silent=True) or {}
    lead_id = str(data.get("lead_id") or "").strip()[:64]
    conversation_id = str(data.get("conversation_id") or "").strip()[:64]
    summary = str(data.get("summary") or "").strip()[:5000]
    action = str(data.get("requested_action") or "Talk to an OrionSoft expert").strip()[:200]
    phone = str(data.get("phone") or "").strip()[:40]

    if not lead_id:
        return jsonify({"error": "Missing lead identifier."}), 400
    if phone and len(re.sub(r"\D", "", phone)) < 7:
        return jsonify({"error": "Please provide a valid phone number with at least 7 digits."}), 400

    leads_path = PROJECT_ROOT / "data" / "leads.json"
    with _leads_lock:
        leads = []
        if leads_path.exists():
            try:
                loaded = json.loads(leads_path.read_text(encoding="utf-8"))
                leads = loaded if isinstance(loaded, list) else []
            except (OSError, json.JSONDecodeError):
                leads = []
        updated = False
        for lead in leads:
            if lead.get("lead_id") == lead_id:
                lead["intent_level"] = "high"
                lead["requested_action"] = action
                if phone:
                    lead["phone"] = phone
                if summary:
                    lead["conversation_summary"] = summary
                if conversation_id:
                    lead["conversation_id"] = conversation_id
                lead["updated_at"] = datetime.now(timezone.utc).isoformat()
                updated = True
                break
        if updated:
            leads_path.write_text(json.dumps(leads, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update lead booking record in Supabase
    if updated:
        try:
            db.update_lead_booking(
                lead_id=lead_id,
                phone=phone,
                summary=summary,
                requested_action=action,
            )
        except Exception as db_err:
            logger.warning(f"Supabase booking update failed for lead {lead_id}: {db_err}")

    return jsonify({
        "success": True,
        "message": "Discovery call request confirmed. Our team will reach out shortly."
    })





@app.route("/api/admin/leads", methods=["GET"])
def admin_leads():
    if not require_admin_key():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401

    leads_path = PROJECT_ROOT / "data" / "leads.json"
    if not leads_path.exists():
        return jsonify({"leads": []})
    try:
        return jsonify({"leads": json.loads(leads_path.read_text(encoding="utf-8"))})
    except Exception:
        logger.exception("Reading leads failed")
        return _server_error()


# ── Core Agentic Query Endpoint ──────────────────────────

@app.route("/api/query", methods=["POST"])
@limiter.limit(lambda: os.getenv("RATE_LIMIT_QUERY", "60/minute"))
def api_query():
    """
    Handle a user query with agentic routing, evidence gating,
    tool execution, and strict structured output validation.
    Secured with API Key authentication for external integrations.
    """
    if not check_auth():
        return jsonify({
            "error": "Unauthorized: Missing or invalid API Key. Please include the 'X-API-Key' header in your request."
        }), 401

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    user_role = data.get("user_role", "employee")
    user_groups = data.get("user_groups", [])

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        response = rag.query(question, user_role=user_role, user_groups=user_groups)
        return jsonify(response.model_dump())
    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        return _server_error()


# ── Document Upload & Ingestion Endpoint ────────────────

SUPPORTED_UPLOAD_EXTENSIONS = {".txt", ".pdf", ".md", ".docx", ".html", ".htm"}

@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    """Accept document uploads, persist them, and re-index the knowledge base."""
    if not check_auth():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    files = request.files.getlist("file")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No file selected."}), 400

    docs_dir = PROJECT_ROOT / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for f in files:
        filename = secure_filename(f.filename)
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))}"
            }), 400
        f.save(str(docs_dir / filename))
        uploaded.append(filename)

    try:
        pipeline = IngestionPipeline(config, project_root=PROJECT_ROOT)
        summary = pipeline.run()
        new_rag = RAGPipeline(config, project_root=PROJECT_ROOT)
        global rag
        rag = new_rag
    except Exception as e:
        logger.error(f"Ingestion after upload failed: {e}", exc_info=True)
        return jsonify({"error": "File saved but indexing failed.", "request_id": g.get("request_id")}), 500

    return jsonify({
        "success": True,
        "uploaded": uploaded,
        "total_documents": summary.total_documents,
        "total_chunks": summary.total_chunks
    })

# ── Google Drive Integration Endpoints ───────────────────

@app.route("/api/integrations/google-drive/status", methods=["GET"])
def gdrive_status():
    """Get high-level Google Drive connection and sync status."""
    if not check_auth():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    return jsonify(rag.google_drive.get_status())


@app.route("/api/integrations/google-drive/authorize", methods=["GET"])
def gdrive_authorize():
    """Initiate Google OAuth 2.0 flow for Google Drive."""
    try:
        auth_url = rag.google_drive.build_authorization_url()
        return redirect(auth_url)
    except ValueError as e:
        logger.warning(f"Google Drive OAuth configuration missing: {e}")
        return jsonify({
            "error": "Google Drive OAuth not configured.",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Google Drive authorization initiation failed: {e}", exc_info=True)
        return _server_error()


@app.route("/api/integrations/google-drive/callback", methods=["GET"])
def gdrive_callback():
    """Handle Google OAuth 2.0 redirect callback."""
    error = request.args.get("error")
    if error:
        logger.warning(f"Google Drive OAuth denied by user: {error}")
        return jsonify({"error": "Google Drive authorization denied.", "detail": error}), 400

    code = request.args.get("code")
    if not code:
        logger.warning("Google Drive OAuth callback missing authorization code.")
        return jsonify({"error": "Google Drive authorization code missing."}), 400

    try:
        rag.google_drive.exchange_code(code)
        return jsonify({"connected": True})
    except Exception as e:
        logger.error(f"Google Drive OAuth code exchange failed: {e}", exc_info=True)
        return jsonify({"error": "Google Drive authorization failed."}), 500


@app.route("/api/integrations/google-drive/connect", methods=["POST"])
def gdrive_connect():
    """Connect company Google Drive workspace."""
    if not require_admin_key():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    data = request.get_json(silent=True) or {}
    account = data.get("account", "admin@orionsofttechnologies.com")
    workspace = data.get("workspace", "OrionSoft Corporate Drive")
    status = rag.google_drive.connect(account=account, workspace=workspace)
    return jsonify(status)


@app.route("/api/integrations/google-drive/disconnect", methods=["POST"])
def gdrive_disconnect():
    """Disconnect company Google Drive workspace."""
    if not require_admin_key():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    status = rag.google_drive.disconnect()
    return jsonify(status)


@app.route("/api/integrations/google-drive/sources", methods=["GET", "POST"])
def gdrive_sources():
    """Inspect or update explicit source folder/file selection."""
    if not require_admin_key():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        selected = data.get("selected_sources", {})
        status = rag.google_drive.update_sources(selected)
        return jsonify(status)
    else:
        return jsonify(rag.google_drive.get_available_sources())


@app.route("/api/integrations/google-drive/sync", methods=["POST"])
def gdrive_sync():
    """Trigger incremental sync pipeline for connected Drive documents."""
    if not require_admin_key():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    try:
        summary = rag.google_drive.sync_now()
        from dataclasses import asdict
        return jsonify(asdict(summary))
    except Exception:
        logger.exception("Google Drive sync failed")
        return jsonify({"error": "Internal server error.", "status": "Failed", "request_id": g.get("request_id")}), 500


# ── Agents & Tools Hub Endpoints ─────────────────────────

@app.route("/api/tools", methods=["GET"])
def list_tools():
    """List registered agent tools and operational metrics."""
    if not check_auth():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    return jsonify(REGISTERED_TOOLS)


@app.route("/api/tools/<tool_id>/toggle", methods=["POST"])
def toggle_tool(tool_id):
    """Enable or disable an agent tool."""
    if not require_admin_key():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    for t in REGISTERED_TOOLS:
        if t["id"] == tool_id:
            t["enabled"] = not t["enabled"]
            return jsonify({"success": True, "tool": t})
    new_tool = {
        "id": tool_id,
        "name": tool_id.replace("tool-", "").replace("-", " ").title(),
        "enabled": True
    }
    REGISTERED_TOOLS.append(new_tool)
    return jsonify({"success": True, "tool": new_tool})


# ── Decision-Quality Evaluation Endpoints ────────────────

@app.route("/api/evaluations/run", methods=["POST"])
def run_eval():
    """Execute the automated decision-quality benchmark suite."""
    if not check_auth():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    try:
        report = run_decision_evaluation(rag=rag)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return _server_error()


@app.route("/api/evaluations/latest", methods=["GET"])
def get_latest_eval():
    """Retrieve the latest benchmark evaluation results and confusion matrix."""
    if not check_auth():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    report_path = PROJECT_ROOT / "evaluation" / "results" / "decision_benchmark.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            logger.exception("Reading evaluation report failed")
            return _server_error()

    # If decision_benchmark doesn't exist yet, run it
    report = run_decision_evaluation(rag=rag)
    return jsonify(report)


@app.route("/api/evaluations/retrieval", methods=["GET", "POST"])
def get_retrieval_eval():
    """Retrieve or execute the automated retrieval quality benchmark suite."""
    if not check_auth():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    report_path = PROJECT_ROOT / "evaluation" / "results" / "retrieval_benchmark.json"
    if request.method == "GET" and report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            logger.exception("Reading retrieval report failed")
            return _server_error()

    try:
        report = run_retrieval_evaluation(rag=rag)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}", exc_info=True)
        return _server_error()


# ── Pipeline Deep Trace Inspector Endpoint ───────────────

@app.route("/api/trace/query", methods=["POST"])
def trace_query():
    """Execute query and return step-by-step technical execution trace."""
    if not check_auth():
        return jsonify({"error": "Unauthorized: Missing or invalid API Key."}), 401
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    user_role = data.get("user_role", "employee")

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    resp = rag.query(question, user_role=user_role)
    return jsonify({
        "query": question,
        "decision": resp.decision.value,
        "evidence_state": resp.evidence_state.value,
        "tools_used": resp.tools_used,
        "confidence_label": resp.confidence_label,
        "confidence_score": resp.confidence_score,
        "latency_ms": resp.latency_ms,
        "execution_trace": [t.model_dump() for t in resp.execution_trace],
        "evidence_signals": resp.evidence_signals.model_dump() if resp.evidence_signals else None,
        "sources": [s.model_dump() for s in resp.sources],
        "answer": resp.answer
    })


# ── System Health & Auth Status ──────────────────────────

@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Check API key authentication status and instructions."""
    api_key = get_api_key()
    client_key = request.headers.get("X-API-Key", "").strip()
    if not client_key:
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header.startswith("Bearer "):
            client_key = auth_header[7:].strip()
    key_valid = (not api_key) or (client_key == api_key)
    return jsonify({
        "auth_required": bool(api_key),
        "header_name": "X-API-Key",
        "configuration_status": "configured" if api_key else "missing",
        "is_authenticated": check_auth(),
        "key_valid": key_valid
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "uptime_s": int(time.monotonic() - START_TIME),
        "provider": config.get("models", {}).get("llm", {}).get("provider", "unknown"),
        "google_drive_connected": rag.google_drive.state.get("connected", False),
        "api_key_secured": bool(get_api_key())
    })



if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    print(f"\n{'='*60}")
    print(f"  Agentic RAG Server is ready!")
    print(f"  Health Check:      http://localhost:{port}/api/health")
    print(f"{'='*60}\n", flush=True)
    # use_reloader=False prevents double-loading heavy neural models & Windows stat-scanning freezes
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)
