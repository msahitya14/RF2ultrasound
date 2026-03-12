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
			key: fs.readFileSync(path.resolve(__dirname, "localhost-key.pem")),
			cert: fs.readFileSync(path.resolve(__dirname, "localhost-cert.pem")),
		},
	},
})
