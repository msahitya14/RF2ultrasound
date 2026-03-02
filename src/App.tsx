import { useEffect, useMemo, useRef, useState } from "react";
import "./app.css";

type Guidance = {
  quality: number; // 0..100
  tiltDeg: number; // signed
  rotDeg: number;  // signed
  confidence?: number; // 0..1 optional
};

function clamp(x: number, a: number, b: number) {
  return Math.max(a, Math.min(b, x));
}

function fmtArrowDeg(x: number, pos: string, neg: string) {
  const deadband = 1.0;
  const v = Math.abs(x) < deadband ? 0 : x;
  const deg = Math.round(Math.abs(v));
  if (deg === 0) return "0°";
  return `${v > 0 ? pos : neg} ${deg}°`;
}

function useSimulatedGuidance(enabled: boolean) {
  const [g, setG] = useState<Guidance>({ quality: 0, tiltDeg: 0, rotDeg: 0, confidence: 0 });

  // smoothing (EMA)
  const s = useRef({ q: 0, t: 0, r: 0, c: 0 });
  const alpha = 0.25;

  useEffect(() => {
    if (!enabled) return;
    let raf = 0;

    const tick = () => {
      const tt = performance.now() / 1000;

      // simulated signals (replace with WS later)
      const quality = clamp(55 + 35 * Math.sin(tt * 0.45), 0, 100);
      const tilt = 10 * Math.sin(tt * 0.7);
      const rot = 6 * Math.cos(tt * 0.55);
      const conf = clamp(0.25 + 0.7 * (quality / 100), 0, 1);

      s.current.q = s.current.q * (1 - alpha) + quality * alpha;
      s.current.t = s.current.t * (1 - alpha) + tilt * alpha;
      s.current.r = s.current.r * (1 - alpha) + rot * alpha;
      s.current.c = s.current.c * (1 - alpha) + conf * alpha;

      setG({
        quality: s.current.q,
        tiltDeg: s.current.t,
        rotDeg: s.current.r,
        confidence: s.current.c,
      });

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [enabled]);

  return g;
}

// Optional: later replace simulated guidance with websocket guidance.
// Expected WS message: {"quality":82,"tiltDeg":-4,"rotDeg":3,"confidence":0.91}
function useWebSocketGuidance(url: string, enabled: boolean) {
  const [g, setG] = useState<Guidance | null>(null);

  useEffect(() => {
    if (!enabled) return;

    const ws = new WebSocket(url);
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (typeof msg.quality === "number" && typeof msg.tiltDeg === "number" && typeof msg.rotDeg === "number") {
          setG({
            quality: msg.quality,
            tiltDeg: msg.tiltDeg,
            rotDeg: msg.rotDeg,
            confidence: typeof msg.confidence === "number" ? msg.confidence : undefined,
          });
        }
      } catch {
        // ignore parse errors
      }
    };
    return () => ws.close();
  }, [url, enabled]);

  return g;
}

export default function App() {
  const [frozen, setFrozen] = useState(false);
  const [mode, setMode] = useState<"sim" | "ws">("sim");

  const sim = useSimulatedGuidance(mode === "sim" && !frozen);
  const ws = useWebSocketGuidance("ws://localhost:8765/guidance", mode === "ws" && !frozen);

  const g: Guidance = useMemo(() => {
    return mode === "ws" && ws ? ws : sim;
  }, [mode, ws, sim]);

  const onTarget = useMemo(() => {
    const confOk = (g.confidence ?? 1) > 0.75;
    return Math.abs(g.tiltDeg) < 2 && Math.abs(g.rotDeg) < 2 && g.quality > 75 && confOk;
  }, [g]);

  const instruction = onTarget ? "ON TARGET — hold still" : "Adjust tilt/rotate";

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">Cisterna Magna Guidance</div>
          <div className="sub">
            Mode: {mode === "sim" ? "Simulated data" : "WebSocket"}.
          </div>
        </div>

        <div className="headerBtns">
          <button className="btn" onClick={() => setMode((m) => (m === "sim" ? "ws" : "sim"))}>
            {mode === "sim" ? "Switch to WebSocket" : "Switch to Sim"}
          </button>
          <button className={`btn ${frozen ? "btnOn" : ""}`} onClick={() => setFrozen((v) => !v)}>
            {frozen ? "Unfreeze" : "Freeze"}
          </button>
        </div>
      </header>

      <main className="wrap">
        <section className="viewer" aria-label="Ultrasound view">
          <DemoUltrasound running={!frozen} />
          <div className="viewerBadge">Ultrasound view (placeholder)</div>
        </section>

        <aside className="panel">
          <div className="label">View quality</div>
          <div className="qRow">
            <div className="qBig">{Math.round(g.quality)}%</div>
            <div className={`state ${g.quality > 78 ? "good" : g.quality > 60 ? "warn" : ""}`}>
              {g.quality > 78 ? "Good" : g.quality > 60 ? "OK" : "Searching"}
            </div>
          </div>
          <div className="bar">
            <div className="barFill" style={{ width: `${clamp(g.quality, 0, 100)}%` }} />
          </div>

          <div className={`inst ${onTarget ? "instOk" : ""}`}>{instruction}</div>

          <div className="row">
            <div className="label">Tilt</div>
            <div className="val">{fmtArrowDeg(g.tiltDeg, "↑", "↓")}</div>
          </div>

          <div className="row">
            <div className="label">Rotate</div>
            <div className="val">{fmtArrowDeg(g.rotDeg, "↻", "↺")}</div>
          </div>

          <div className="row smallTop">
            <div className="label">Confidence</div>
            <div className="smallVal">{(g.confidence ?? 0).toFixed(2)}</div>
          </div>

          <div className="hint">
            
          </div>
        </aside>
      </main>
    </div>
  );
}

function DemoUltrasound({ running }: { running: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let t = 0;

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.round(r.width * devicePixelRatio);
      canvas.height = Math.round(r.height * devicePixelRatio);
    };

    const draw = () => {
      if (!running) {
        raf = requestAnimationFrame(draw);
        return;
      }
      t += 1;

      resize();
      const w = canvas.width, h = canvas.height;

      // noisy grayscale
      const img = ctx.createImageData(w, h);
      for (let i = 0; i < img.data.length; i += 4) {
        const n = (Math.random() * 70 + 10) | 0;
        img.data[i] = img.data[i + 1] = img.data[i + 2] = n;
        img.data[i + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);

      // a curved bright arc
      ctx.save();
      ctx.translate(w * 0.55, h * 0.55);
      ctx.rotate(Math.sin(t / 90) * 0.12);
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = Math.max(2, 2 * devicePixelRatio);
      ctx.beginPath();
      ctx.arc(0, 0, Math.min(w, h) * 0.35, Math.PI * 0.15, Math.PI * 0.85);
      ctx.stroke();
      ctx.restore();

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [running]);

  return <canvas className="ultra" ref={canvasRef} />;
}