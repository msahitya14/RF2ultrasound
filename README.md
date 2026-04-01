# RF2ultrasound

## Getting Started

This project serves a web app and WebSocket server for ultrasound angle guidance. Your phone acts as the sensor, sending gyroscope/orientation data to the server over WebSocket.

### Prerequisites

- Node.js (v16 or newer recommended)
- npm
- [mkcert](https://github.com/FiloSottile/mkcert) — for generating local HTTPS certificates

### Setup & Run

1. **Install dependencies:**
```sh
   npm install
```

2. **Install mkcert (first time only):**
```sh
   brew install mkcert
   mkcert -install
```

3. **Start the app:**
```sh
   bash start.sh
```

   This script will:
   - Detect your local IP address
   - Remove any old certificates and generate new ones for your current IP
   - Kill anything running on ports 3000 and 5173
   - Start the HTTPS + WebSocket backend on port 3000
   - Start the Vite dev server on port 5173

4. **Open on your phone:**

   The script will print a URL like:
```
   https://192.168.x.x:5173
```
   Open that on your phone (must be on the same Wi-Fi network). Accept the SSL warning if prompted.

### Notes

- HTTPS is required — Safari and Chrome will not expose gyroscope/orientation sensors over plain HTTP.
- The WebSocket endpoint is at `/ws` (e.g., `wss://<your-ip>:5173/ws`).
- The angles endpoint is at `/angles` (e.g., `https://<your-ip>:3000/angles`).
- Both your computer and phone must be on the same local network.