"""Production WSGI entrypoint using Waitress (cross-platform, no Gunicorn on Windows)."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    try:
        from waitress import serve
        print(f"Serving Agentic RAG on http://0.0.0.0:{port} with Waitress")
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        print(f"Serving Agentic RAG on http://0.0.0.0:{port} with Flask development server")
        app.run(host="0.0.0.0", port=port, debug=False)
