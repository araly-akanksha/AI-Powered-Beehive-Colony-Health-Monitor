const PLACEHOLDER = {
  spectralCentroidHz: "—",
  rms: "—",
  zcr: "—",
  freqBand: "—",
  colonyStatus: "Awaiting clip"
};

export default function BioacousticMetrics({ metrics }) {
  const m = metrics || PLACEHOLDER;
  const statusTone = m.colonyStatus === "Normal" ? "success" : m.colonyStatus === "Distress" ? "danger" : "neutral";

  return (
    <dl className="metrics-list">
      <div className="metrics-list__row">
        <dt>Spectral Centroid</dt>
        <dd>{typeof m.spectralCentroidHz === "number" ? `${m.spectralCentroidHz.toLocaleString()} Hz` : m.spectralCentroidHz}</dd>
      </div>
      <div className="metrics-list__row">
        <dt>Energy Level (RMS)</dt>
        <dd>{typeof m.rms === "number" ? m.rms.toFixed(4) : m.rms}</dd>
      </div>
      <div className="metrics-list__row">
        <dt>Zero Crossing Rate</dt>
        <dd>{typeof m.zcr === "number" ? m.zcr.toFixed(3) : m.zcr}</dd>
      </div>
      <div className="metrics-list__row">
        <dt>Frequency Band</dt>
        <dd>{m.freqBand}</dd>
      </div>
      <div className="metrics-list__row">
        <dt>Colony Status</dt>
        <dd className={`metrics-list__status metrics-list__status--${statusTone}`}>{m.colonyStatus}</dd>
      </div>
    </dl>
  );
}
