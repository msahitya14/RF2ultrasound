# RF2ultrasound

## Getting Started

This project serves a web app and WebSocket server for ultrasound angle guidance.

### Prerequisites

- Node.js (v16 or newer recommended)
- npm

### Setup & Run

1. **Install dependencies:**

   ```sh
   npm install
   ```

2. **Start the backend server (HTTPS + WebSocket):**

   ```sh
   node server.cjs
   ```

   - This serves the production build (from `dist/`) and the WebSocket endpoint on port 3000.
   - Make sure your SSL certificates (`localhost-key.pem` and `localhost-cert.pem`) are present in the project root. See `HTTPS_SETUP.md` for details.

3. **(Optional) Start the frontend in development mode:**

   ```sh
   npm run dev
   ```

   - This runs the Vite dev server (usually on port 5173).
   - For production, use the backend server only.

4. **Access the app from your device:**
   - Find your computer's LAN IP address (e.g., `172.31.70.230`).
   - On your phone or another device on the same Wi-Fi/LAN, open a browser and go to:
     - `https://<your-ip>:3000/` (for production)
     - or `https://<your-ip>:5173/` (for development)
   - Accept any SSL warnings if using self-signed certificates.

### Notes

- The WebSocket endpoint is at `/ws` (e.g., `wss://<your-ip>:3000/ws`).
- Both your computer and phone must be on the same local network.
- For HTTPS setup, see `HTTPS_SETUP.md`.

---

Feel free to update this README as your project evolves!
