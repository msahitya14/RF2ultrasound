#!/bin/bash
set -e

IP=$(ipconfig getifaddr en0)
echo "Detected IP: $IP"

# Kill anything on port 3000
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Generate certs for the current IP (named key.pem / cert.pem for app.py)
mkcert -key-file key.pem -cert-file cert.pem "$IP" localhost 127.0.0.1

# Build the React frontend
echo "Building frontend..."
npm run build

# Start the FastAPI server (serves built SPA + all API endpoints)
echo ""
echo "Open on phone: https://$IP:3000"
echo ""
python3 app.py "$@"
