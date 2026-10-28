# OrionSoft Agentic RAG Platform — Engineering & Operational Handoff Document

**Document Version:** 1.0.0  
**Target Environment:** Production (DigitalOcean Droplet + Nginx + Gunicorn)  
**Public API Endpoint:** `https://api.orionsofttechnologies.com`  
**Frontend Widget Portal:** `https://orionagenticrag.netlify.app`  

---

## 1. System Overview & Architecture

The OrionSoft Agentic RAG Platform is a production-grade enterprise conversational engine delivering **zero-hallucination knowledge retrieval**, **deterministic arithmetic evaluation**, and **automated lead qualification**.

### Architectural Core
1. **Autonomous Multi-Path Router**: Categorizes incoming queries into five distinct paths (`internal_rag`, `google_drive`, `calculator`, `web_search`, and `abstain`).
2. **Anti-Exfiltration Gate**: Enforces strict air-gapping on confidential terms (salaries, internal policies, strategic roadmaps), preventing prompt-injection leaks to external tools.
3. **Hybrid Dense + Sparse Retrieval**: Dual-indexes company knowledge using `all-MiniLM-L6-v2` dense vectors (FAISS) and `BM25Okapi` lexical tokens.
4. **Cross-Encoder Reranker**: Employs `cross-encoder/ms-marco-MiniLM-L-6-v2` to score query-chunk cross-attention, eliminating false positives before LLM generation.
5. **Deterministic AST Math Engine**: Compiles numerical questions into isolated Python Abstract Syntax Trees (`ast.AST`) with `eval()` and `exec()` banned.
6. **Rule 6 Negative Abstention**: Automatically declines to speculate when document evidence is insufficient or conflicting.
7. **Automated Lead Dispatch & Scheduling**: Captures prospect intent, runs syntax and disposable-domain filters, dispatches real-time email alerts via Resend, and creates 1-click Google Calendar, Outlook, and `.ics` meeting invites.

---

## 2. Production Infrastructure Specification

| Component | Specification | Location / Command |
| :--- | :--- | :--- |
| **Server Host** | DigitalOcean Droplet (Ubuntu 22.04 LTS) | IP: `142.93.217.14` (Bengaluru, IN) |
| **App Directory** | `/home/deploy/Agentic-Rag-backend-` | Deploy User: `deploy` / `root` |
| **Python Runtime** | Python 3.11 Virtual Environment | `/home/deploy/Agentic-Rag-backend-/.venv` |
| **App Server (WSGI)** | Gunicorn (1 worker, 4 threads, 180s timeout) | Bound to `127.0.0.1:5000` |
| **Process Manager** | Systemd Service (`orion-rag.service`) | `/etc/systemd/system/orion-rag.service` |
| **Reverse Proxy** | Nginx with SSL (Let's Encrypt / Certbot) | `/etc/nginx/sites-available/api.orionsofttechnologies.com` |
| **Public API Domain** | `https://api.orionsofttechnologies.com` | DNS A Record pointing to `142.93.217.14` |
| **Swap Memory** | 2.0 GB Swapfile (`/swapfile`) | Prevents OOM-killer termination during heavy model loads |

---

## 3. Configuration & Environment Variables (`.env`)

Configuration is stored in `/home/deploy/Agentic-Rag-backend-/.env`. This file is intentionally excluded from Git for security.

```ini
# ── Server & Runtime ──
PORT=5000
APP_ENV=production
LOG_LEVEL=INFO
APP_VERSION=1.0.0
ALLOWED_ORIGINS=https://orionagenticrag.netlify.app,https://orionsofttechnologies.com

# ── API Security ──
RAG_API_KEY=rag_sec_8f7b2a9c1d4e

# ── Email & Sales Lead Alerts ──
# Resend API Key:
RESEND_API_KEY=re_your_active_resend_api_key_here
# Primary recipient for real-time lead and booking alerts:
SALES_NOTIFICATION_EMAIL=orionsofttechnologiespvt.ltd@gmail.com
# From & Reply-To addresses:
EMAIL_FROM=OrionSoft AI <onboarding@resend.dev>
EMAIL_REPLY_TO=orionsofttechnologiespvt.ltd@gmail.com

# ── Secondary SMTP Fallback (Optional) ──
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=orionsofttechnologies@gmail.com
SMTP_PASSWORD=
SMTP_USE_TLS=true

# ── Neural Generation & LLM Providers ──
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key

# ── Persistence ──
DATABASE_URL=
RAG_DB_PATH=data/app.db
```

---

## 4. CI/CD & Automated Deployment

The repository includes a GitHub Actions workflow in [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml).

### Automated Pipeline Workflow
Every `git push` to `main`:
1. Connects to the DigitalOcean Droplet via SSH (`appleboy/ssh-action`).
2. Stashes any local edits and runs `git pull origin main`.
3. Updates dependencies via `pip install -r requirements.txt`.
4. Gracefully reboots Gunicorn via `orion-rag.service`.
5. Pings `http://127.0.0.1:5000/api/health` to verify zero-downtime recovery.

### Required GitHub Secrets
Configure under **GitHub Repository → Settings → Secrets and variables → Actions**:
* `DROPLET_HOST`: `142.93.217.14`
* `DROPLET_USER`: `root` or `deploy`
* `DROPLET_PASSWORD` (or `DROPLET_SSH_KEY`): Server authentication credential.

---

## 5. Operational Runbook & Maintenance

### Daily Service Control
```bash
# Check service health & uptime:
systemctl status orion-rag --no-pager

# Restart the backend service:
systemctl restart orion-rag

# Stream live backend execution logs:
journalctl -u orion-rag -f -n 50

# Restart the Nginx web server:
systemctl restart nginx
```

### One-Command Full Stack Recovery Script
If Nginx or Gunicorn ever needs to be reconfigured from scratch, run the automated setup script:
```bash
cd /home/deploy/Agentic-Rag-backend-
bash setup_nginx.sh
```

### Health Check Verification
```bash
# Internal loopback test:
curl -i http://127.0.0.1:5000/api/health

# Public HTTPS test:
curl -i https://api.orionsofttechnologies.com/api/health
```

Expected output:
```json
{"api_key_secured":true,"google_drive_connected":false,"provider":"gemini","status":"ok","uptime_s":1452,"version":"1.0.0"}
```

---

## 6. Troubleshooting & Failure Recovery

### 1. Chatbot shows "Failed to fetch" in browser
* **Root Cause**: Gunicorn is down on port 5000, causing Nginx to return `502 Bad Gateway`.
* **Fix**:
  ```bash
  # Check if port 5000 is occupied by a stale zombie process:
  fuser -k 5000/tcp || true
  # Restart service:
  systemctl restart orion-rag
  ```

### 2. "Out of Memory" (OOM) or Process Killed
* **Root Cause**: Loading Sentence-Transformers and FAISS requires ~0.8 GB of RAM at initialization.
* **Fix**: Ensure swap memory is active:
  ```bash
  swapon --show
  # If inactive, allocate 2GB swap:
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  ```

### 3. Lead Emails Not Arriving
* **Root Cause**: Resend API key mismatch or recipient restrictions.
* **Verification**: Run the built-in email test suite:
  ```bash
  cd /home/deploy/Agentic-Rag-backend-
  .venv/bin/python test_email.py
  ```
* Ensure `RESEND_API_KEY` in `.env` matches the active key generated from `https://resend.com/api-keys`.

---

## 7. Website Integration & Embed Options

### Option A: Standard 1-Line Script Tag
Paste right before the closing `</body>` tag on any CMS (WordPress, Webflow, Shopify, custom HTML):
```html
<script src="https://orionagenticrag.netlify.app/widget-loader.js" async></script>
```

### Option B: Isolated iFrame Embed
```html
<iframe 
  src="https://orionagenticrag.netlify.app" 
  style="width: 100%; height: 700px; border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);"
  title="OrionSoft AI Assistant">
</iframe>
```

---

## 8. Handoff Checklist & Sign-Off

- [x] Backend source code pushed and synchronized on branch `main`
- [x] Continuous deployment configured via GitHub Actions (`.github/workflows/deploy.yml`)
- [x] Nginx reverse proxy configured with valid Let's Encrypt SSL (`api.orionsofttechnologies.com`)
- [x] Systemd persistent watchdog service active (`orion-rag.service`)
- [x] 2GB Swapfile allocated to prevent memory termination
- [x] Email notification service verified and delivering live to `orionsofttechnologiespvt.ltd@gmail.com`
- [x] Public health endpoint returning HTTP 200 OK
