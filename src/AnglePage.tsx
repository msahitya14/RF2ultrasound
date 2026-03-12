import { useEffect, useRef, useState } from "react"
import NoSleep from "nosleep.js"

function clamp(v: number, a: number, b: number) {
  return Math.max(a, Math.min(b, v))
}

export default function AnglePage() {
  const [beta, setBeta] = useState(0)
  const [gamma, setGamma] = useState(0)
  const [baseBeta, setBaseBeta] = useState(0)
  const [baseGamma, setBaseGamma] = useState(0)
  const [calibrated, setCalibrated] = useState(false)
  const [permissionGranted, setPermissionGranted] = useState(false)
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected")

  const socketRef = useRef<WebSocket | null>(null)
  const lastBeta = useRef(0)
  const lastGamma = useRef(0)
  const noSleepRef = useRef<NoSleep | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const relX = calibrated ? beta - baseBeta : beta
  const relY = calibrated ? gamma - baseGamma : gamma
  const onTarget = Math.abs(relX) < 1.5 && Math.abs(relY) < 1.5

  const bubblePos = () => {
    const r = (containerRef.current?.offsetWidth ?? 240) / 2
    const maxR = r * 0.38
    return {
      bx: clamp((relY / 45) * maxR * 2.5, -maxR, maxR),
      by: clamp((relX / 45) * maxR * 2.5, -maxR, maxR),
    }
  }
  const { bx, by } = bubblePos()

  const requestPermission = async () => {
    try {
      const anyOr = DeviceOrientationEvent as any
      const anyMo = DeviceMotionEvent as any
      if (typeof anyOr?.requestPermission === "function") {
        const r = await anyOr.requestPermission()
        if (r !== "granted") { alert("Orientation sensor access denied. Reload and tap Allow."); return }
      }
      if (typeof anyMo?.requestPermission === "function") {
        const r = await anyMo.requestPermission()
        if (r !== "granted") { alert("Motion sensor access denied. Reload and tap Allow."); return }
      }
      setPermissionGranted(true)
    } catch (err) {
      alert(String(err))
    }
  }

  useEffect(() => {
    if (!permissionGranted) return
    const handle = (e: DeviceOrientationEvent) => {
      if (e.beta == null || e.gamma == null) return
      lastBeta.current = e.beta
      lastGamma.current = e.gamma
      setBeta(e.beta)
      setGamma(e.gamma)
      if (calibrated && socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({
          type: "angles",
          x: e.beta - baseBeta,
          y: e.gamma - baseGamma,
        }))
      }
    }
    window.addEventListener("deviceorientation", handle, true)
    return () => window.removeEventListener("deviceorientation", handle, true)
  }, [permissionGranted, calibrated, baseBeta, baseGamma])

  useEffect(() => {
    let ws: WebSocket
    let timer: ReturnType<typeof setTimeout>
    const connect = () => {
      setWsStatus("connecting")
      const proto = location.protocol === "https:" ? "wss" : "ws"
      ws = new WebSocket(`${proto}://${location.hostname}:3000/ws`)
      ws.onopen = () => setWsStatus("connected")
      ws.onerror = () => setWsStatus("error")
      ws.onclose = () => { setWsStatus("disconnected"); timer = setTimeout(connect, 2000) }
      socketRef.current = ws
    }
    connect()
    return () => { ws?.close(); clearTimeout(timer) }
  }, [])

  const calibrate = () => {
    setBaseBeta(lastBeta.current)
    setBaseGamma(lastGamma.current)
    setCalibrated(true)
    if (!noSleepRef.current) noSleepRef.current = new NoSleep()
    noSleepRef.current.enable()
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "calibrate" }))
    }
  }

  const wsColor =
    wsStatus === "connected" ? "rgba(50,220,100,0.9)" :
    wsStatus === "error" ? "#ff453a" :
    "rgba(255,255,255,0.25)"

  const wsLabel =
    wsStatus === "connected" ? "Connected" :
    wsStatus === "connecting" ? "Connecting…" :
    wsStatus === "error" ? "Connection error" : "Disconnected"

  return (
    <>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body, #root { height: 100%; background: #000; }
        body { overflow: hidden; }
      `}</style>

      <div style={{
        minHeight: "100dvh",
        background: "#000",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "space-evenly",
        padding: "env(safe-area-inset-top, 16px) 24px env(safe-area-inset-bottom, 16px)",
        fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
        color: "#fff",
        userSelect: "none",
      }}>

        {/* Label */}
        <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
          Level
        </div>

        {/* Bubble */}
        <div ref={containerRef} style={{
          width: "min(62vw, 260px)",
          height: "min(62vw, 260px)",
          position: "relative",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}>
          {[["100%","rgba(255,255,255,0.14)","solid"],["71%","rgba(255,255,255,0.2)","solid"],["40%","rgba(255,255,255,0.1)","dashed"]].map(([s, c, st], i) => (
            <div key={i} style={{ position: "absolute", width: s, height: s, borderRadius: "50%", border: `1.5px ${st} ${c}` }} />
          ))}
          <div style={{ position: "absolute", width: "88%", height: 1, background: "rgba(255,255,255,0.08)" }} />
          <div style={{ position: "absolute", width: 1, height: "88%", background: "rgba(255,255,255,0.08)" }} />
          <div style={{
            position: "absolute",
            width: "19%", height: "19%",
            borderRadius: "50%",
            background: onTarget ? "rgba(50,220,100,0.92)" : "rgba(255,214,0,0.92)",
            boxShadow: `0 0 0 4px ${onTarget ? "rgba(50,220,100,0.2)" : "rgba(255,214,0,0.18)"}`,
            transform: `translate(${bx}px, ${by}px)`,
            transition: "transform 0.1s ease, background 0.3s, box-shadow 0.3s",
          }} />
          <div style={{ position: "absolute", width: "3%", height: "3%", borderRadius: "50%", background: "rgba(255,255,255,0.35)" }} />
        </div>

        {/* On-target hint */}
        <div style={{ fontSize: 13, color: "rgba(50,220,100,0.9)", letterSpacing: "0.04em", height: 18, opacity: onTarget ? 1 : 0, transition: "opacity 0.25s" }}>
          Flat — hold still
        </div>

        {/* Angle readouts */}
        <div style={{ display: "flex", gap: "clamp(16px, 5vw, 32px)", alignItems: "flex-end" }}>
          {[{ val: relX, label: "Tilt" }, { val: relY, label: "Rotate" }].map(({ val, label }, i) => (
            <>
              {i === 1 && <div key="sep" style={{ width: 1, height: 44, background: "rgba(255,255,255,0.12)", alignSelf: "center", marginBottom: 26 }} />}
              <div key={label} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                <span style={{ fontSize: "clamp(34px, 10vw, 50px)", fontWeight: 200, letterSpacing: "-0.02em", fontVariantNumeric: "tabular-nums", color: onTarget ? "rgba(50,220,100,0.95)" : "#fff", transition: "color 0.3s" }}>
                  {Math.abs(val).toFixed(1)}
                </span>
                <span style={{ fontSize: 13, color: "rgba(255,255,255,0.45)", marginTop: -4 }}>°</span>
                <span style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", letterSpacing: "0.08em", textTransform: "uppercase", marginTop: 4 }}>{label}</span>
              </div>
            </>
          ))}
        </div>

        {/* Action button */}
        {!permissionGranted ? (
          <button onClick={requestPermission} style={btnStyle(false)}>Enable Sensors</button>
        ) : (
          <button onClick={calibrate} style={btnStyle(calibrated)}>
            {calibrated ? "Recalibrate" : "Calibrate"}
          </button>
        )}

        {/* WS status */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: wsColor, transition: "background 0.3s" }} />
          <span style={{ fontSize: 12, color: "rgba(255,255,255,0.3)" }}>{wsLabel}</span>
        </div>

      </div>
    </>
  )
}

function btnStyle(active: boolean): React.CSSProperties {
  return {
    width: "min(220px, 70vw)",
    height: 52,
    borderRadius: 26,
    background: "rgba(255,255,255,0.07)",
    border: `1.5px solid ${active ? "rgba(50,220,100,0.55)" : "rgba(255,255,255,0.2)"}`,
    color: active ? "rgba(50,220,100,0.95)" : "#fff",
    fontSize: 17,
    fontWeight: 400,
    cursor: "pointer",
    fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
    transition: "border-color 0.3s, color 0.3s",
  }
}