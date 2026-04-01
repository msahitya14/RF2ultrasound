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
			key: fs.readFileSync(path.resolve(__dirname, "10.8.146.142+2-key.pem")),
			cert: fs.readFileSync(path.resolve(__dirname, "10.8.146.142+2.pem")),
		},
		proxy: {
			"/ws": { target: "https://10.8.146.142:3000", ws: true, secure: false },
			"/angles": { target: "https://10.8.146.142:3000", secure: false },
		},
	},
})
