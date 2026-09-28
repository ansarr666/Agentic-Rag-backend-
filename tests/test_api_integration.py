"""Comprehensive API and Persistence integration test suite."""

import json
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def local_database(monkeypatch):
    """Keep API integration tests isolated from the configured Supabase database."""
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")


def test_lead_country_code_and_ten_digit_phone(tmp_path, monkeypatch):
    """The visitor form stores a combined number and rejects malformed input."""
    import app as app_module

    (tmp_path / "data").mkdir()
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "validate_work_email", lambda email: (True, ""))
    saved_leads = []
    monkeypatch.setattr(app_module.db, "save_lead", saved_leads.append)
    client = app_module.app.test_client()
    lead = {"name": "Jane Doe", "email": "jane@enterprise.com", "country_code": "+91"}

    for phone in ("123456789", "12345678901", "12345abc90"):
        response = client.post("/api/public/lead", json={**lead, "phone": phone})
        assert response.status_code == 400

    response = client.post("/api/public/lead", json={**lead, "phone": "9876543210"})
    assert response.status_code == 200
    assert saved_leads[0]["phone"] == "+919876543210"
    stored_leads = json.loads((tmp_path / "data" / "leads.json").read_text(encoding="utf-8"))
    assert stored_leads[0]["phone"] == "+919876543210"


def test_db_persistence(tmp_path, monkeypatch):
    """Verify SQLite persistence layer for users, conversations, and messages."""
    import db
    db_file = tmp_path / "test_app.db"
    monkeypatch.setattr(db, "DB_PATH", db_file)

    db.init_db()
    assert db_file.exists()

    # User upsert
    db.upsert_user("usr_001", "dev@orionsoft.com", "admin", "oauth")
    
    # Conversation creation and retrieval
    cid = db.create_conversation("usr_001", "Sprint Planning Chat")
    assert cid is not None
    conv = db.get_conversation(cid)
    assert conv is not None
    assert conv["title"] == "Sprint Planning Chat"
    assert conv["user_id"] == "usr_001"

    # Message append and retrieval
    db.add_message(cid, "user", "What is the sprint velocity?", {"source": "web"})
    db.add_message(cid, "assistant", "Sprint velocity is 42 points.", {"confidence": 0.95})
    messages = db.get_messages(cid)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is the sprint velocity?"
    assert messages[0]["metadata"]["source"] == "web"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["metadata"]["confidence"] == 0.95

    db.save_lead({"lead_id": "lead_001", "conversation_id": cid, "name": "Jane", "email": "jane@orionsoft.com"})
    db.save_lead({"lead_id": "lead_001", "conversation_id": cid, "name": "Jane Doe", "email": "jane@orionsoft.com"})
    saved_leads = db.get_all_leads()
    assert len(saved_leads) == 1
    assert saved_leads[0]["name"] == "Jane Doe"


def test_auth_check_helpers(monkeypatch):
    """Verify auth helper functions in auth.py."""
    import auth
    from app import app

    # Unauthenticated when key is empty
    monkeypatch.setenv("RAG_API_KEY", "")
    with app.test_request_context():
        assert auth.check_auth() is True
        assert auth.require_admin_key() is False

    # Authenticated when key is set
    monkeypatch.setenv("RAG_API_KEY", "secret_123")
    with app.test_request_context():
        assert auth.check_auth() is False

    with app.test_request_context(headers={"X-API-Key": "wrong_key"}):
        assert auth.check_auth() is False

    with app.test_request_context(headers={"X-API-Key": "secret_123"}):
        assert auth.check_auth() is True

    with app.test_request_context(headers={"Authorization": "Bearer secret_123"}):
        assert auth.check_auth() is True

    with app.test_request_context(headers={"Origin": "http://localhost"}):
        assert auth.check_auth() is True
        assert auth.require_admin_key() is False

    with app.test_request_context(headers={"X-API-Key": "secret_123"}):
        assert auth.require_admin_key() is True


def test_api_routes(monkeypatch):
    """Verify core API endpoints using Flask test client."""
    monkeypatch.setenv("RAG_API_KEY", "test-admin-key")
    from app import app
    client = app.test_client()

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "version" in data

    # Auth status
    res = client.get("/api/auth/status")
    assert res.status_code == 200
    data = res.get_json()
    assert "auth_required" in data
    assert "is_authenticated" in data

    # Tools list and toggle
    res = client.get("/api/tools", headers={"Origin": "http://localhost:5000"})
    assert res.status_code == 200
    tools = res.get_json()
    assert isinstance(tools, list)
    assert len(tools) > 0

    res = client.post("/api/tools/tool-calculator/toggle", headers={"X-API-Key": "test-admin-key"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True

    # Google Drive status & sources
    res = client.get("/api/integrations/google-drive/status", headers={"Origin": "http://localhost:5000"})
    assert res.status_code == 200
    data = res.get_json()
    assert "connected" in data

    res = client.post(
        "/api/integrations/google-drive/sources",
        json={"selected_sources": {"my_drive": True}},
        headers={"X-API-Key": "test-admin-key"}
    )
    assert res.status_code == 200

    # Public Lead submission
    res = client.post("/api/public/lead", json={
        "name": "Jane Doe",
        "email": "jane@enterprise.com",
        "phone": "+1 555 123 4567",
        "company": "Acme Corp",
        "project_requirement": "Custom Agentic RAG integration"
    })
    assert res.status_code == 200
    lead_data = res.get_json()
    assert lead_data["success"] is True
    assert "lead_id" in lead_data

    # Admin leads retrieval
    assert client.get("/api/admin/leads", headers={"Origin": "http://localhost:5000"}).status_code == 401
    res = client.get("/api/admin/leads", headers={"X-API-Key": "test-admin-key"})
    assert res.status_code == 200
    leads = res.get_json()
    assert "leads" in leads

    # Query endpoint empty question validation
    res = client.post(
        "/api/query",
        json={"question": ""},
        headers={"Origin": "http://localhost:5000"}
    )
    assert res.status_code == 400
    assert "error" in res.get_json()

    # The backend serves API responses only; the frontend serves its own pages.
    res = client.get("/")
    assert res.status_code == 404
    assert client.get("/admin").status_code == 404
    res = client.get("/widget")
    assert res.status_code == 404

    preflight = client.options(
        "/api/public/conversation",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"

    # Retrieval evaluations endpoint
    res = client.get("/api/evaluations/retrieval", headers={"Origin": "http://localhost:5000"})
    assert res.status_code == 200
    ret_data = res.get_json()
    assert "hit_rate_at_k" in ret_data
    assert "mrr" in ret_data


def test_conversation_dual_write(tmp_path, monkeypatch):
    """Verify visitor exchanges are dual-written to SQLite and JSON."""
    import db
    from app import app
    client = app.test_client()

    cid = "test_conv_dual_write_123"
    res = client.post("/api/public/conversation", json={
        "question": "What services do you provide?",
        "conversation_id": cid,
        "name": "Alex",
        "lead_id": "lead_test_001"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "answer" in data

    # Verify message written to SQLite
    sqlite_messages = db.get_messages(cid)
    assert len(sqlite_messages) >= 2
    assert sqlite_messages[0]["role"] == "user"
    assert sqlite_messages[0]["content"] == "What services do you provide?"
    assert sqlite_messages[1]["role"] == "assistant"


def test_google_drive_oauth_endpoints(monkeypatch):
    """Verify Google Drive OAuth 2.0 authorization and callback endpoints."""
    from app import app
    from pipeline.tools.google_drive import GoogleDriveConnector
    client = app.test_client()

    # 1. Test build_authorization_url with mock env vars
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id-12345.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:5000/api/integrations/google-drive/callback")

    url = GoogleDriveConnector.build_authorization_url()
    assert "https://accounts.google.com/o/oauth2/v2/auth" in url
    assert "test-client-id-12345" in url
    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.readonly" in url or "drive.readonly" in url

    # 2. Test GET /api/integrations/google-drive/authorize with configured env vars
    res = client.get("/api/integrations/google-drive/authorize")
    assert res.status_code == 302
    assert "accounts.google.com" in res.headers.get("Location", "")

    # 3. Test GET /api/integrations/google-drive/authorize when env vars are unconfigured
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    res_unconfigured = client.get("/api/integrations/google-drive/authorize")
    assert res_unconfigured.status_code == 400
    assert "Google Drive OAuth not configured" in res_unconfigured.get_json()["error"]

    # 4. Test GET /api/integrations/google-drive/callback error handling
    res_err = client.get("/api/integrations/google-drive/callback?error=access_denied")
    assert res_err.status_code == 400
    assert res_err.get_json()["detail"] == "access_denied"

    # 5. Test GET /api/integrations/google-drive/callback missing code
    res_nocode = client.get("/api/integrations/google-drive/callback")
    assert res_nocode.status_code == 400
    assert "missing" in res_nocode.get_json()["error"].lower()
