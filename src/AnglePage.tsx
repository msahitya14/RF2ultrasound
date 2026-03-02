import { useEffect, useRef, useState } from "react"

export default function AnglePage() {
  const [beta, setBeta] = useState(0)
  const [gamma, setGamma] = useState(0)
  const [baseBeta, setBaseBeta] = useState(0)
  const [baseGamma, setBaseGamma] = useState(0)
  const [calibrated, setCalibrated] = useState(false)
  const [permissionGranted, setPermissionGranted] = useState(false)
  const [wsStatus, setWsStatus] = useState("Disconnected")
  const [lastPayload, setLastPayload] = useState<{ x: number; y: number } | null>(null)

  const socketRef = useRef<WebSocket | null>(null)
  const lastBeta = useRef(0)
  const lastGamma = useRef(0)

  // Request iOS permissions
  const requestPermission = async () => {
    try {
      const anyOrientation = DeviceOrientationEvent as any
      const anyMotion = DeviceMotionEvent as any

      if (typeof anyOrientation?.requestPermission === "function") {
        const r1 = await anyOrientation.requestPermission()
        if (r1 !== "granted") {
          alert(
            "iPhone blocked sensors.\nEnable: Settings → Safari → Motion & Orientation Access → ON.\nReload Safari."
          )
          return
        }
      }

      if (typeof anyMotion?.requestPermission === "function") {
        const r2 = await anyMotion.requestPermission()
        if (r2 !== "granted") {
          alert(
            "iPhone blocked motion sensors.\nEnable: Settings → Safari → Motion & Orientation Access → ON.\nReload Safari."
          )
          return
        }
      }

      setPermissionGranted(true)
    } catch (err) {
      console.error(err)
      alert(String(err))
    }
  }

  // Device orientation listener
  useEffect(() => {
    if (!permissionGranted) return

    const handleOrientation = (event: DeviceOrientationEvent) => {
      if (event.beta != null && event.gamma != null) {
        lastBeta.current = event.beta
        lastGamma.current = event.gamma
        setBeta(event.beta)
        setGamma(event.gamma)
      }
    }

    window.addEventListener("deviceorientation", handleOrientation, true)
    return () => window.removeEventListener("deviceorientation", handleOrientation, true)
  }, [permissionGranted])

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket
    let reconnectTimeout: ReturnType<typeof setTimeout>

    const connect = () => {
      setWsStatus("Connecting...")
      const wsUrl = `${window.location.origin.replace(/^http/, "ws")}/ws`
      ws = new WebSocket(wsUrl)

      ws.onopen = () => setWsStatus("Connected")
      ws.onerror = () => setWsStatus("Error")
      ws.onclose = () => {
        setWsStatus("Disconnected")
        reconnectTimeout = setTimeout(connect, 2000)
      }

      socketRef.current = ws
    }

    connect()
    return () => {
      ws?.close()
      clearTimeout(reconnectTimeout)
    }
  }, [])

  // Calibrate
  const calibrate = () => {
    setBaseBeta(lastBeta.current)
    setBaseGamma(lastGamma.current)
    setCalibrated(true)

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "calibrate" }))
    }
  }

  const relX = calibrated ? lastBeta.current - baseBeta : 0
  const relY = calibrated ? lastGamma.current - baseGamma : 0

  // Send angles
useEffect(() => {
  if (!calibrated) return

  const interval = setInterval(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      const x = lastBeta.current - baseBeta
      const y = lastGamma.current - baseGamma

      const payload = { type: "angles", x, y }

      socketRef.current.send(JSON.stringify(payload))
      setLastPayload({ x, y })

      console.log("Sending:", payload)
    }
  }, 50)

  return () => clearInterval(interval)
}, [calibrated, baseBeta, baseGamma])
  return (
    <div
      onClick={() => {
        if (permissionGranted) calibrate()
      }}
      style={{ padding: 20, fontFamily: "sans-serif", minHeight: "100vh" }}
    >
      <h1>Angle Tracker</h1>

      {!permissionGranted && (
        <button style={{ fontSize: 18, padding: 10 }} onClick={requestPermission}>
          Enable Sensors
        </button>
      )}

      {permissionGranted && (
        <>
          <p>
            <strong>Beta:</strong> {beta.toFixed(2)}
          </p>
          <p>
            <strong>Gamma:</strong> {gamma.toFixed(2)}
          </p>

          <p>
            <strong>Relative X:</strong> {relX.toFixed(2)} | <strong>Relative Y:</strong>{" "}
            {relY.toFixed(2)}
          </p>

          <p>
            <strong>WebSocket Status:</strong> {wsStatus}
          </p>

          <p>
            <strong>Tap anywhere to calibrate</strong> (sets x,y to 0,0)
          </p>

          {lastPayload && (
            <p>
              <strong>Last Sent:</strong> x={lastPayload.x.toFixed(2)}, y={lastPayload.y.toFixed(2)}
            </p>
          )}
        </>
      )}
    </div>
  )
}