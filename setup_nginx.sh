#!/usr/bin/env bash
set -e

echo "==> Configuring Nginx for api.orionsofttechnologies.com..."

# Copy config
cp config/nginx_api.conf /etc/nginx/sites-available/api.orionsofttechnologies.com

# Enable site
ln -sf /etc/nginx/sites-available/api.orionsofttechnologies.com /etc/nginx/sites-enabled/api.orionsofttechnologies.com

# Test syntax
nginx -t

# Reload Nginx
systemctl reload nginx
echo "==> Nginx reloaded successfully!"

# Run certbot if available
if command -v certbot &> /dev/null; then
    echo "==> Requesting/renewing SSL certificate with Certbot..."
    certbot --nginx -d api.orionsofttechnologies.com --non-interactive --agree-tos -m consultations@orionsofttechnologies.com || echo "[!] Certbot warning: check DNS or rate limits"
else
    echo "==> Certbot not found, installing..."
    apt-get update && apt-get install -y certbot python3-certbot-nginx
    certbot --nginx -d api.orionsofttechnologies.com --non-interactive --agree-tos -m consultations@orionsofttechnologies.com || true
fi

echo "==> Ensuring Gunicorn/orion-rag is running..."
systemctl restart orion-rag || true

echo "==> Testing local backend health..."
sleep 2
curl -i http://127.0.0.1:5000/api/health || echo "Check if orion-rag service is running."

echo "==> All done!"
