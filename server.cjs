const express = require("express")
const path = require("path")
const https = require("https")
const fs = require("fs")
const WebSocket = require("ws")

const app = express()

const key  = fs.readFileSync(path.join(__dirname, "10.8.213.214+3-key.pem"))
const cert = fs.readFileSync(path.join(__dirname, "10.8.213.214+3.pem"))


const server = https.createServer({ key, cert }, app)

const buildPath = path.join(__dirname, "dist")
const indexHtml = path.join(buildPath, "index.html")

// let latestAngles = { x: 0, y: 0, calibratedAt: null, updatedAt: null }

app.get("/angles", (req, res) => {
	res.json(latestAngles)
})

if (fs.existsSync(indexHtml)) {
	app.use(express.static(buildPath))
	app.use((req, res) => {
		res.sendFile(indexHtml)
	})
} else {
	app.use((req, res, next) => {
		if (req.path === "/angles") return next()
		res.status(404).send("dist/index.html not found. Run 'npm run build' or use 'npm run dev' with Vite.")
	})
}

let latestAngles = { x: 0, y: 0, calibratedAt: null, updatedAt: null }

const wss = new WebSocket.Server({ server, path: "/ws" })

wss.on("connection", (ws, req) => {
	console.log("Client connected from:", req.socket.remoteAddress)

	ws.on("message", msg => {
		try {
			const data = JSON.parse(msg)
			if (data.type === "angles") {
				latestAngles = {
					x: data.x,
					y: data.y,
					calibratedAt: latestAngles.calibratedAt,
					updatedAt: new Date().toISOString(),
				}
			} else if (data.type === "calibrate") {
				latestAngles.calibratedAt = new Date().toISOString()
			}
			console.log("Received:", data)
		} catch {
			console.log("Invalid message:", msg)
		}
	})

	ws.on("close", () => console.log("Client disconnected"))
})

const HOST = "0.0.0.0"
const PORT = process.env.PORT || 3000

server.listen(PORT, HOST, () => {
	console.log("Server running on https://${HOST}:${PORT}")
	console.log("Angles endpoint: https://${HOST}:${PORT}/angles")
})
