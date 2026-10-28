# OrionSoft Agentic RAG Platform — Engineering & Operational Handoff Document

**Document Version:** 1.1.0  
**Last Updated:** October 28, 2026  
**Target Environment:** Production (DigitalOcean Droplet + Nginx + Gunicorn)  
**Public API Endpoint:** `https://api.orionsofttechnologies.com`  
**Frontend Widget Portal:** `https://orionagenticrag.netlify.app`  

---

## 1. System Overview & Architecture

The OrionSoft Agentic RAG Platform is a production-grade enterprise conversational engine delivering **zero-hallucination knowledge retrieval**, **deterministic arithmetic evaluation**, and **automated lead qualification**.

### Architectural Core
1. **Autonomous Multi-Path Router**: Categorizes incoming queries into five distinct paths (`internal_rag`, `google_drive`, `calculator`, `web_search`, and `abstain`).
2. **Smart Conversational & Booking Router**: Tolerates colloquial phrasing and common typos (e.g. `"ook me call"`, `"book me call"`, `"call me"`, `"talk to sales"`), immediately triggering high-intent discovery call flows without pushy rephrase loops.
3. **Pure English Guardrail Enforcement**: Rigorously sanitizes output and suggested follow-up pills, stripping non-English/Hinglish tokens (`"kaam"`, `"karta"`, `"sakta"`, `"bana"`, `"dete"`, etc.).
4. **Anti-Exfiltration Gate**: Enforces strict air-gapping on confidential terms (salaries, internal policies, strategic roadmaps), preventing prompt-injection leaks to external tools.
5. **Hybrid Dense + Sparse Retrieval**: Dual-indexes company knowledge using `all-MiniLM-L6-v2` dense vectors (FAISS) and `BM25Okapi` lexical tokens.
6. **Cross-Encoder Reranker**: Employs `cross-encoder/ms-marco-MiniLM-L-6-v2` to score query-chunk cross-attention, eliminating false positives before LLM generation.
7. **Deterministic AST Math Engine**: Compiles numerical questions into isolated Python Abstract Syntax Trees (`ast.AST`) with `eval()` and `exec()` banned.
8. **Rule 6 Negative Abstention**: Automatically declines to speculate when document evidence is insufficient or conflicting.
9. **Reliable Lead Dispatch & Calendar Invites**: Delivers real-time email notifications on new leads, lead updates, and discovery calls to all 4 sales team inboxes via direct Google SMTP with self-healing credentials, generating 1-click Google Calendar, Outlook, and `.ics` meeting invites.

---

## 2. Production Infrastructure Specification

| Component | Specification | Location / Command |
| :--- | :--- | :--- |
| **Server Host** | DigitalOcean Droplet (Ubuntu 22.04 LTS) | IP: `142.93.217.14` (Bengaluru, IN) |
| **App Directory** | `/home/deploy/Agentic-Rag-backend-` | Deploy User: `deploy` / `root` |
| **Python VirtualEnv** | Python 3.11 Virtual Environment | `/home/deploy/Agentic-Rag-backend-/.venv` |
| **App Server (WSGI)** | Gunicorn (1 worker, 2 threads, 180s timeout) | Bound to `127.0.0.1:5000` |
| **Process Manager** | Systemd Service (`orion-rag.service`) | `/etc/systemd/system/orion-rag.service` |
| **Reverse Proxy** | Nginx with SSL (Let's Encrypt / Certbot) | `/etc/nginx/sites-available/api.orionsofttechnologies.com` |
| **Public API Domain** | `https://api.orionsofttechnologies.com` | DNS A Record pointing to `142.93.217.14` |
| **Swap Memory** | 2.0 GB Swapfile (`/swapfile`) | Prevents OOM-killer termination during model loads |

---

## 3. Configuration & Environment Variables (`.env`)

Configuration is stored in `/home/deploy/Agentic-Rag-backend-/.env`. This file is intentionally excluded from Git for security.

```ini
# ── Server & Runtime ──
PORT=5000
APP_ENV=production
LOG_LEVEL=INFO
APP_VERSION=1.1.0
ALLOWED_ORIGINS=https://orionagenticrag.netlify.app,https://orionsofttechnologies.com

# ── API Security ──
RAG_API_KEY=rag_sec_8f7b2a9c1d4e

# ── Email & Notification Configuration ──
# Exact display name & reply address prospects will see:
EMAIL_FROM=OrionSoft AI <info@orionsofttechnologies.com>
EMAIL_REPLY_TO=orionsofttechnologiespvt.ltd@gmail.com

# Dispatched in real-time to all 4 team inboxes:
SALES_NOTIFICATION_EMAIL=orionsofttechnologiespvt.ltd@gmail.com, anam@orionsofttechnologies.com, rahul@orionsofttechnologies.com, sakib@orionsofttechnologies.com

# Standard Google SMTP Configuration:
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=orionsofttechnologiespvt.ltd@gmail.com
SMTP_PASSWORD=aapi usnk zndq aqmm
SMTP_USE_TLS=true

# Resend API Key (Optional secondary fallback):
RESEND_API_KEY=

# ── Neural Generation & LLM Providers ──
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key

# ── Persistence ──
DATABASE_URL=postgresql://...supabase.co...
RAG_DB_PATH=data/app.db
```

> [!NOTE]
> `email_service.py` features **self-healing credentials**: it automatically strips spaces from `SMTP_PASSWORD` (`aapi usnk zndq aqmm` → `aapiusnkzndqaqmm`), auto-corrects `orionsofttechnologies@gmail.com` to `orionsofttechnologiespvt.ltd@gmail.com`, and falls back to all 4 team addresses if environment variables are missing.

---

## 4. Deployment & Operations Runbook

### Method 1: Safe One-Line Deploy Command (Recommended)
Paste this single chained command in the Droplet Console:
```bash
cd /home/deploy/Agentic-Rag-backend- && git pull origin main && sudo systemctl restart orion-rag.service
```

### Method 2: Using the Makefile Target
```bash
cd /home/deploy/Agentic-Rag-backend- && make deploy
```

### Method 3: Automated Update Script
```bash
cd /home/deploy/Agentic-Rag-backend- && ./update.sh
```

---

## 5. Verification & Service Health Checks

### Check Systemd Service Status
```bash
systemctl status orion-rag.service --no-pager
```
*Expected: `active (running)` in green.*

### Check Internal HTTP Loopback
```bash
curl -I http://127.0.0.1:5000/api/health
```
*Expected: `HTTP/1.1 200 OK`.*

### Check Public HTTPS Production Endpoint
```bash
curl -s https://api.orionsofttechnologies.com/api/health
```
*Expected output:*
```json
{"api_key_secured":true,"google_drive_connected":true,"provider":"gemini","status":"ok","uptime_s":120,"version":"1.0.0"}
```

---

## 6. Daily Service Control Commands

```bash
# Restart backend service:
sudo systemctl restart orion-rag.service

# Stream live backend execution logs:
journalctl -u orion-rag.service -f -n 50

# Restart Nginx web server:
sudo systemctl reload nginx

# Check listening ports:
ss -tulpn | grep 5000
```

---

## 7. Troubleshooting & Failure Recovery

### 1. Chatbot shows "Failed to fetch" in browser
* **Root Cause**: Gunicorn service stopped or crashed on port 5000, causing Nginx to return `502 Bad Gateway`.
* **Fix**:
  ```bash
  fuser -k 5000/tcp || true
  sudo systemctl restart orion-rag.service
  ```

### 2. "Out of Memory" (OOM) Termination
* **Root Cause**: Sentence-Transformers and FAISS require ~0.8 GB of RAM during model weight loading.
* **Fix**: Ensure swapfile is active:
  ```bash
  swapon --show
  # If inactive, allocate 2GB swap:
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  ```

### 3. Lead Notifications & Booking Alerts
* **Recipients**: Real-time alerts dispatch concurrently to all 4 team inboxes:
  1. `orionsofttechnologiespvt.ltd@gmail.com`
  2. `anam@orionsofttechnologies.com`
  3. `rahul@orionsofttechnologies.com`
  4. `sakib@orionsofttechnologies.com`
* **Triggers**:
  - `POST /api/public/lead` (New prospect submission)
  - `POST /api/public/lead` (Lead update from chat handoff)
  - `POST /api/public/intent` (Discovery call booking request)

---

## 8. Website Integration & Embed Options

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

## 9. Handoff Checklist & Sign-Off

- [x] Backend source code pushed and synchronized on branch `main` (`cf76b2c`)
- [x] Production server restarted and operational with sub-second health checks (`uptime_s` verified)
- [x] Smart booking intent handles typos (`"ook me call"`, `"book me call"`, `"call me"`) with zero rephrase loops
- [x] Pure English enforcement active on all responses and suggested pill questions
- [x] Real-time sales alerts verified and delivering live to all 4 team inboxes via Google SMTP
- [x] One-line deploy script, `Makefile`, and `update.sh` committed and verified
- [x] Nginx reverse proxy configured with valid Let's Encrypt SSL (`api.orionsofttechnologies.com`)
- [x] Systemd persistent watchdog service active (`orion-rag.service`)
- [x] Public health endpoint returning HTTP 200 OK
