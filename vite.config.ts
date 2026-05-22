import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import fs from "fs"

const hasCerts = fs.existsSync("key.pem") && fs.existsSync("cert.pem")

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: true,
    ...(hasCerts && {
      https: {
        key: fs.readFileSync("key.pem"),
        cert: fs.readFileSync("cert.pem"),
      },
    }),
    proxy: {
      "/ws":     { target: "https://localhost:3000", ws: true, secure: false },
      "/angles": { target: "https://localhost:3000", secure: false },
    },
  },
})
