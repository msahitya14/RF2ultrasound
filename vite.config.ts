// vite.config.ts

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
			key: fs.readFileSync(path.resolve(__dirname, "172.31.83.113+2-key.pem")),
			cert: fs.readFileSync(path.resolve(__dirname, "172.31.83.113+2.pem")),
		},
		proxy: {
			"/ws": {
				target: "https://172.31.83.113:3000",
				ws: true,
				secure: false,
			},
			"/angles": {
				target: "https://172.31.83.113:3000",
				secure: false,
			},
		},
	},
})
