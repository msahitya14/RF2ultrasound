const express = require("express")
const path = require("path")
const https = require("https")
const fs = require("fs")
const WebSocket = require("ws")

const app = express()

const key  = fs.readFileSync(path.join(__dirname, "10.8.146.142+2-key.pem"))
const cert = fs.readFileSync(path.join(__dirname, "10.8.146.142+2.pem"))

const server = https.createServer({ key, cert }, app)

const buildPath = path.join(__dirname, "dist")
app.use(express.static(buildPath))
app.use((req, res) => {
	res.sendFile(path.join(buildPath, "index.html"))
})

const wss = new WebSocket.Server({ server, path: "/ws" })
let latestAngles = { x: 0, y: 0, calibratedAt: null, updatedAt: null }

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

app.get("/angles", (req, res) => {
	res.json(latestAngles)
})

const HOST = "10.8.146.142"
const PORT = process.env.PORT || 3000

server.listen(PORT, "0.0.0.0", () => {
	console.log("Server running on https://${HOST}:${PORT}")
	console.log("Angles endpoint: https://${HOST}:${PORT}/angles")
})
