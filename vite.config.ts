import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import fs from "fs"
import path from "path"

export default defineConfig({
    plugins: [react()],
    server: {
        host: true,
        port: 5173,
        strictPort: true,
        allowedHosts: true,
        https: {
            key: fs.readFileSync('key.pem'),
            cert: fs.readFileSync('cert.pem'),
        },
        proxy: {
            "/ws": { target: "https://localhost:3000", ws: true, secure: false },
            "/angles": { target: "https://localhost:3000", secure: false },
        },
    },
})