// ============================================================
// components/SpectrogramDisplay.jsx
// Shows the mel-spectrogram PNG returned by the backend
// "What the model actually sees"
// ============================================================

export default function SpectrogramDisplay({ b64 }) {
  if (!b64) return null;
  return (
    <div className="glass-card spectrogram-panel">
      <div className="card-title">🌡️ Mel-Spectrogram — What the Model Sees</div>
      <img
        src={`data:image/png;base64,${b64}`}
        alt="Mel-spectrogram of the analysed audio clip"
        title="Frequency (y-axis) vs Time (x-axis). Brighter = louder at that frequency."
      />
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
        Y-axis: Mel frequency bins (0–8 kHz) · X-axis: Time frames · Colour: amplitude (dB)
      </p>
    </div>
  );
}
