import StatusPill from "./StatusPill";
import { DEMO_MODE } from "../config";

const VIEW_OPTIONS = [
  { value: "compare", label: "Compare All 3" },
  { value: "dann", label: "DANN only" },
  { value: "crnn", label: "CRNN only" },
  { value: "baseline_cnn", label: "Baseline CNN only" }
];

export default function Header({ status, viewMode, onViewModeChange }) {
  const statusTone = status === "analyzing" ? "amber" : status === "error" ? "danger" : "success";
  const statusLabel =
    status === "analyzing" ? "Analyzing…" : status === "error" ? "Backend Error" : "Model Ready";

  return (
    <header className="app-header">
      <div className="app-header__title">
        <span className="app-header__mark" aria-hidden="true">🐝</span>
        <div>
          <h1>Beehive Health Monitor</h1>
          <p>Live bioacoustic diagnosis from hive audio</p>
        </div>
      </div>

      <div className="app-header__controls">
        <a
          href="/live.html"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.35rem",
            padding: "0.3rem 0.75rem",
            borderRadius: "9999px",
            fontSize: "0.75rem",
            fontWeight: 700,
            background: "rgba(245, 158, 11, 0.15)",
            color: "#f59e0b",
            border: "1px solid rgba(245, 158, 11, 0.35)",
            textDecoration: "none",
            boxShadow: "0 0 10px rgba(245, 158, 11, 0.15)"
          }}
          title="Open Live 60FPS Waterfall & Web Audio Sentinel"
        >
          <span>✨</span>
          <span>Live Sentinel UI</span>
        </a>

        {DEMO_MODE && <StatusPill tone="neutral">Demo Mode</StatusPill>}
        <StatusPill tone={statusTone}>
          <span className="status-pill__dot" aria-hidden="true" />
          {statusLabel} · DANN / CRNN
        </StatusPill>

        <label className="model-select">
          <span>Model view</span>
          <select value={viewMode} onChange={(e) => onViewModeChange(e.target.value)}>
            {VIEW_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>
    </header>
  );
}
