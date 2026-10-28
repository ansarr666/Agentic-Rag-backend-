#!/usr/bin/env bash
set -e

DIR="/home/deploy/Agentic-Rag-backend-"
cd "$DIR"

echo "==> Pulling latest changes from main branch..."
git pull origin main

echo "==> Restarting orion-rag.service..."
if command -v systemctl &> /dev/null; then
    sudo systemctl restart orion-rag.service 2>/dev/null || systemctl restart orion-rag.service 2>/dev/null || {
        echo "==> systemctl failed, restarting gunicorn directly..."
        pkill -f "gunicorn" || true
        sleep 2
        .venv/bin/gunicorn --workers 1 --threads 2 --timeout 180 --bind 127.0.0.1:5000 app:app --daemon
    }
else
    pkill -f "gunicorn" || true
    sleep 2
    .venv/bin/gunicorn --workers 1 --threads 2 --timeout 180 --bind 127.0.0.1:5000 app:app --daemon
fi

echo "==> Verifying service status..."
sleep 2
sudo systemctl status orion-rag.service --no-pager 2>/dev/null || true

echo "==> Health Check:"
curl -I http://127.0.0.1:5000/api/health || curl -I http://127.0.0.1:5000/
echo "==> Deployment complete!"
