import { useEffect, useRef, useState } from "react"
import NoSleep from "nosleep.js"

export default function AnglePage() {
	const [beta, setBeta] = useState(0)
	const [gamma, setGamma] = useState(0)
	const [baseBeta, setBaseBeta] = useState(0)
	const [baseGamma, setBaseGamma] = useState(0)
	const [calibrated, setCalibrated] = useState(false)
	const [permissionGranted, setPermissionGranted] = useState(false)
	const [wsStatus, setWsStatus] = useState("Disconnected")
	const [lastPayload, setLastPayload] = useState<{
		x: number
		y: number
	} | null>(null)

	const socketRef = useRef<WebSocket | null>(null)
	const lastBeta = useRef(0)
	const lastGamma = useRef(0)
	const noSleepRef = useRef<NoSleep | null>(null)

	// Request iOS permissions
	const requestPermission = async () => {
		try {
			const anyOrientation = DeviceOrientationEvent as any
			const anyMotion = DeviceMotionEvent as any

			if (typeof anyOrientation?.requestPermission === "function") {
				const r1 = await anyOrientation.requestPermission()
				if (r1 !== "granted") {
					alert(
						[
							"Sensor access denied!",
							"\nTo use this feature, please:",
							"1. Reload the page.",
							"2. When prompted, tap 'Allow' to enable device orientation sensors.",
							"3. If you do not see a prompt, try closing all Safari tabs, clearing website data (Settings → Safari → Clear History and Website Data), and reopening the page.",
							"4. Make sure you are not in Private Browsing mode.",
							"\nIf you denied by mistake, reload the page and try again.",
						].join("\n"),
					)
					return
				}
			}

			if (typeof anyMotion?.requestPermission === "function") {
				const r2 = await anyMotion.requestPermission()
				if (r2 !== "granted") {
					alert(
						"Motion sensor access denied. Please tap 'Allow' when prompted to enable motion sensors. If you denied by mistake, reload the page and try again.",
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

				// Only send angles if calibrated and WS open
				if (calibrated && socketRef.current?.readyState === WebSocket.OPEN) {
					const x = event.beta - baseBeta
					const y = event.gamma - baseGamma
					socketRef.current.send(JSON.stringify({ type: "angles", x, y }))
					setLastPayload({ x, y })
				}
			}
		}

		window.addEventListener("deviceorientation", handleOrientation, true)
		return () =>
			window.removeEventListener("deviceorientation", handleOrientation, true)
	}, [permissionGranted, calibrated, baseBeta, baseGamma])

	// WebSocket connection
	useEffect(() => {
		let ws: WebSocket
		let reconnectTimeout: ReturnType<typeof setTimeout>

		const connect = () => {
			setWsStatus("Connecting...")
			// Use the backend server's port (3000) for WebSocket
			const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws"
			const wsHost = window.location.hostname
			const wsPort = 3000
			const wsUrl = `${wsProtocol}://${wsHost}:${wsPort}/ws`
			ws = new WebSocket(wsUrl)

			ws.onopen = () => {
				console.log("WebSocket connected")
				setWsStatus("Connected")
			}

			ws.onerror = err => {
				console.error("WebSocket error", err)
				setWsStatus("Error")
			}

			ws.onclose = () => {
				console.log("WebSocket disconnected")
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

	// Calibrate button
	const calibrate = () => {
		setBaseBeta(lastBeta.current)
		setBaseGamma(lastGamma.current)
		setCalibrated(true)

		console.log("Calibrated at:", lastBeta.current, lastGamma.current)

		// Keep the screen awake
		if (!noSleepRef.current) {
			noSleepRef.current = new NoSleep()
		}
		noSleepRef.current.enable()

		if (socketRef.current?.readyState === WebSocket.OPEN) {
			socketRef.current.send(JSON.stringify({ type: "calibrate" }))
		}
	}

	// Compute relative angles
	const relX = calibrated ? beta - baseBeta : 0
	const relY = calibrated ? gamma - baseGamma : 0

	return (
		<div style={{ padding: 20, fontFamily: "sans-serif", minHeight: "100vh" }}>
			<h1>Angle Tracker</h1>

			{!permissionGranted && (
				<button
					style={{ fontSize: 18, padding: 10 }}
					onClick={requestPermission}
				>
					Enable Sensors
				</button>
			)}

			{permissionGranted && (
				<>
					<button
						onClick={calibrate}
						style={{ fontSize: 18, padding: 10, marginBottom: 20 }}
					>
						Calibrate
					</button>

					<p>
						<strong>Beta:</strong> {beta.toFixed(2)}
					</p>

					<p>
						<strong>Gamma:</strong> {gamma.toFixed(2)}
					</p>

					<p>
						<strong>Relative X:</strong> {relX.toFixed(2)} |{" "}
						<strong>Relative Y:</strong> {relY.toFixed(2)}
					</p>

					<p>
						<strong>WebSocket Status:</strong> {wsStatus}
					</p>

					{lastPayload && (
						<p>
							<strong>Last Sent:</strong> x={lastPayload.x.toFixed(2)}, y=
							{lastPayload.y.toFixed(2)}
						</p>
					)}
				</>
			)}
		</div>
	)
}
