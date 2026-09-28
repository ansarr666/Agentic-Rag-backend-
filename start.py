"""Production entrypoint: launch gunicorn on $PORT (default 5000).

Workers and threads can be tuned with WEB_CONCURRENCY and GUNICORN_THREADS.
Each worker loads its own copy of the embedding and reranker models (~0.5 GB each).
"""

import os
import sys


from pathlib import Path


def main() -> None:
    # Speed up startup by disabling online HuggingFace update checks if cached
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    port = os.getenv("PORT", "5000")
    workers = os.getenv("WEB_CONCURRENCY", "2")
    threads = os.getenv("GUNICORN_THREADS", "4")

    print(f"Starting Agentic RAG: http://localhost:{port} "
          f"(bind 0.0.0.0:{port}, workers={workers}, threads={threads}, timeout=120s)", flush=True)

    if sys.platform == "win32":
        try:
            from waitress import serve
            from app import app
            serve(app, host="0.0.0.0", port=int(port), threads=int(threads))
        except ImportError:
            from app import app
            app.run(host="0.0.0.0", port=int(port), debug=False)
        return

    # Replace this process so gunicorn receives container stop signals directly.
    # No gunicorn access log: app.py logs each API request as JSON with latency_ms.
    os.execvp("gunicorn", [
        "gunicorn",
        "--bind", f"0.0.0.0:{port}",
        "--workers", workers,
        "--threads", threads,
        "--timeout", "120",
        "app:app",
    ])


if __name__ == "__main__":
    main()
