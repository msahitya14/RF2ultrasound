// server.js
const express = require("express")
const path = require("path")
const http = require("http")
const WebSocket = require("ws")

const app = express()
const server = http.createServer(app)


const buildPath = path.join(__dirname, "dist") // Vite build folder
app.use(express.static(buildPath))
app.use((req, res) => {
  res.sendFile(path.join(buildPath, "index.html"))
})

const wss = new WebSocket.Server({ server, path: "/ws" })
let latestAngles = { x: 0, y: 0, calibratedAt: null, updatedAt: null }

wss.on("connection", (ws) => {
  console.log("Client connected")

  ws.on("message", (msg) => {
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


const PORT = process.env.PORT || 3000
server.listen(PORT, () => {
  console.log(`Server + WebSocket running on port ${PORT}`)
  console.log(`HTTP endpoint for latest angles: http://localhost:${PORT}/angles`)
})