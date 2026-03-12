// Self-signed certificate generation instructions for Vite HTTPS dev server
//
// 1. Open a terminal in your project directory.
// 2. Run the following command to generate a key and certificate:
//    openssl req -x509 -newkey rsa:2048 -nodes -keyout localhost-key.pem -out localhost-cert.pem -days 365 -subj "/CN=localhost"
// 3. Move the generated files to your project root (if not already there).
// 4. Update vite.config.ts to use these files for HTTPS.
// 5. Access your app via https://your-local-ip:5173 (accept the certificate warning on your iOS device).
