#!/usr/bin/env bash
set -e

echo "==> Pulling latest changes from main..."
git stash || true
git pull origin main

echo "==> Restarting Gunicorn server..."
pkill -f "gunicorn" || true
sleep 2
.venv/bin/gunicorn --workers 1 --threads 2 --timeout 180 --bind 127.0.0.1:5000 app:app --daemon

echo "==> Testing local health check..."
sleep 2
curl -s http://127.0.0.1:5000/api/health
echo ""
echo "==> Update complete!"
