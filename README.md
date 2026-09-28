# Agentic RAG backend

Flask API for visitor chat, the admin workspace, document ingestion, hybrid retrieval, agent routing, LLM generation, evaluation, Google Drive integration, and persistence. This repository does not build or serve React files.

## Run locally (Windows PowerShell)

```powershell
cd C:\Users\ANAM\Agentic-RAG-backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env: set ALLOWED_ORIGINS=http://localhost:5173 and required provider credentials
.\.venv\Scripts\python.exe main.py --mode ingest
.\.venv\Scripts\python.exe start.py
```

The API listens on `PORT` (default 5000). Check `http://localhost:5000/api/health`. On Linux, use `python3 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, and `.venv/bin/python start.py`.

`main.py --mode ingest` builds the BM25 and vector indexes from `data/public`, `data/raw`, and `documents`. The checked-in public source is `data/public/Category.pdf`. Local indexes are ignored by Git; ingest after a fresh clone. Other CLI modes: `--mode query --question "..." `, `--mode interactive`, and `--mode eval`.

## Configuration

Copy `.env.example` and fill only the values needed for your deployment. The template contains names only. `config/config.yaml` selects model providers, retrieval, and paths. Paths resolve relative to this repository.

- `ALLOWED_ORIGINS`: comma-separated exact frontend origins, for example `http://localhost:5173`. Include every deployed admin/widget origin. The API does not use browser cookies.
- `RAG_API_KEY`: admin/API key. Set this for the admin workspace; never put it in frontend build configuration. A human admin enters it in the admin page.
- `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`: server-side provider keys according to `config/config.yaml`. Without the chosen key, the existing grounded generator fallback applies.
- `DATABASE_URL`: PostgreSQL connection string. Without it, SQLite uses `data/app.db`; `RAG_DB_PATH` overrides that path. Leads and conversations are also written to JSON files under `data/`.
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`: Google Drive OAuth. The redirect URI must be the backend URL ending in `/api/integrations/google-drive/callback` and must match the provider registration.
- `SMTP_*`, `RESEND_API_KEY`, `EMAIL_FROM`, `EMAIL_REPLY_TO`, `OTP_EXPIRE_MINUTES`: email and OTP.
- `RATE_LIMIT_*`, `TRUSTED_PROXY_COUNT`, `SENTRY_*`, `PORT`, `WEB_CONCURRENCY`, `GUNICORN_THREADS`: runtime operations.

Keep `data/`, `documents/`, and `evaluation/results/` on persistent storage. Existing local visitor records and OAuth state were deliberately not copied from the original repository into Git; migrate them privately if needed.

## API and authentication

See [API_CONTRACT.md](API_CONTRACT.md). The visitor endpoints are public and rate limited. Admin endpoints require `X-API-Key` or the matching bearer key. `/api/query` and related staff endpoints use the existing `check_auth` behavior. `auth.py` also contains JWT/introspection identity helpers; the current Flask routes do not invoke those helpers. Google Drive OAuth redirects from the backend to Google and back to the backend callback; it is separate from admin sign-in and does not issue a browser session.

## Test and deploy

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`Dockerfile` builds this API alone and starts `python start.py`. Pass environment variables at deployment, expose port 5000 or `PORT`, configure `ALLOWED_ORIGINS`, and mount persistent data. The repository has no frontend build dependency.
