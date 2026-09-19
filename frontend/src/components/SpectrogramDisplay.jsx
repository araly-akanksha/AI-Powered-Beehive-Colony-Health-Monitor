import { useEffect, useRef } from "react";

const GRADIENT_STOPS = [
  { stop: 0.0, color: [15, 23, 42] }, // slate-900 (background / silence)
  { stop: 0.35, color: [22, 78, 99] }, // teal-900
  { stop: 0.6, color: [8, 145, 178] }, // cyan-600
  { stop: 0.82, color: [245, 158, 11] }, // amber-500 (accent)
  { stop: 1.0, color: [253, 230, 138] } // amber-200 (hottest)
];

function colorFor(value) {
  const v = Math.max(0, Math.min(1, value));
  for (let i = 0; i < GRADIENT_STOPS.length - 1; i++) {
    const a = GRADIENT_STOPS[i];
    const b = GRADIENT_STOPS[i + 1];
    if (v >= a.stop && v <= b.stop) {
      const t = (v - a.stop) / (b.stop - a.stop || 1);
      const r = Math.round(a.color[0] + (b.color[0] - a.color[0]) * t);
      const g = Math.round(a.color[1] + (b.color[1] - a.color[1]) * t);
      const bl = Math.round(a.color[2] + (b.color[2] - a.color[2]) * t);
      return `rgb(${r}, ${g}, ${bl})`;
    }
  }
  return "rgb(15, 23, 42)";
}

function drawMatrix(canvas, bands) {
  const numBands = bands.length;
  const numFrames = bands[0]?.length || 0;
  if (!numBands || !numFrames) return;

  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 480;
  const cssHeight = canvas.clientHeight || 200;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const cellWidth = cssWidth / numFrames;
  const cellHeight = cssHeight / numBands;

  for (let b = 0; b < numBands; b++) {
    // Low frequency bands drawn at the bottom, like a conventional spectrogram.
    const y = cssHeight - (b + 1) * cellHeight;
    for (let t = 0; t < numFrames; t++) {
      ctx.fillStyle = colorFor(bands[b][t]);
      ctx.fillRect(t * cellWidth, y, cellWidth + 0.5, cellHeight + 0.5);
    }
  }
}

export default function SpectrogramDisplay({ spectrogram, isDemo }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (spectrogram?.bands && canvasRef.current) {
      drawMatrix(canvasRef.current, spectrogram.bands);
    }
  }, [spectrogram]);

  if (!spectrogram) {
    return <div className="spectrogram spectrogram--empty">Upload or select a clip to generate a spectrogram.</div>;
  }

  if (spectrogram.imageBase64) {
    return (
      <div className="spectrogram">
        <img
          className="spectrogram__image"
          src={`data:image/png;base64,${spectrogram.imageBase64}`}
          alt="Log-mel spectrogram of the analyzed clip"
        />
      </div>
    );
  }

  return (
    <div className="spectrogram">
      <canvas ref={canvasRef} className="spectrogram__canvas" role="img" aria-label="Log-mel spectrogram of the analyzed clip" />
      <div className="spectrogram__axis">
        <span>0s</span>
        <span>2.0s</span>
      </div>
      {isDemo && <p className="spectrogram__note">Simulated for demo mode — real backend returns the actual PyTorch log-mel output.</p>}
    </div>
  );
}
