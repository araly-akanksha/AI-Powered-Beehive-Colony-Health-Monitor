// ============================================================
// components/ResultCard.jsx
// Animated prediction result with confidence bars
// ============================================================

export default function ResultCard({ result, isUnseen }) {
  const { label, confidence, all_confidences, n_segments, processing_ms, model_name } = result;

  const isQueenPresent = label === 'queen_present';

  const labelDisplay = isQueenPresent ? 'Queen Present 👑' : 'Queen Absent ⚠️';
  const verdictClass = isQueenPresent ? 'queen-present' : 'queen-absent';
  const confPct      = (confidence * 100).toFixed(1);

  // Bar colour logic
  const barColor = (lbl) => {
    if (lbl === 'queen_present') return 'green';
    if (lbl === 'queen_absent')  return 'red';
    return 'amber';
  };

  return (
    <div className="glass-card result-card">
      <div className="card-title">📊 Prediction Result</div>

      {/* Generalization banner — shown when clip is from an unseen test hive */}
      {isUnseen && (
        <div className="gen-banner">
          <span className="gen-icon">🧪</span>
          <div className="gen-banner-text">
            <h3>Unseen Test Hive</h3>
            <p>
              This clip is from a hive the model <strong>has never seen during training</strong>.
              The accuracy drop from known hives to this hive is the cross-hive generalization
              gap — the core research finding of this project.
            </p>
          </div>
        </div>
      )}

      {/* Main verdict */}
      <div className={`result-verdict ${verdictClass}`}>
        <span className="verdict-icon">{isQueenPresent ? '👑' : '🚨'}</span>
        <div className="verdict-text">
          <h2>{labelDisplay}</h2>
          <div className="confidence-text">
            Model confidence: <strong>{confPct}%</strong>
          </div>
        </div>
      </div>

      {/* Per-class confidence bars */}
      <div className="confidence-bars">
        {Object.entries(all_confidences).map(([lbl, prob]) => (
          <div key={lbl} className="confidence-bar-row">
            <div className="confidence-bar-label">
              <span>{lbl.replace('_', ' ')}</span>
              <span className="pct">{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${barColor(lbl)}`}
                style={{ width: `${(prob * 100).toFixed(1)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Metadata chips */}
      <div className="result-meta">
        {n_segments && (
          <div className="meta-chip">📦 {n_segments} segments</div>
        )}
        {processing_ms && (
          <div className="meta-chip">⚡ {processing_ms}ms</div>
        )}
        {model_name && (
          <div className="meta-chip">🧠 {model_name}</div>
        )}
      </div>
    </div>
  );
}
