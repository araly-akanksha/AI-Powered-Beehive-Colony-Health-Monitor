// ============================================================
// components/SampleClips.jsx
// Pre-loaded demo clips for Demo Day
// Known hive clips + unseen test hive clips to demonstrate
// the cross-hive generalization gap live
// ============================================================

import { useState, useEffect } from 'react';
import { getSamples, predictSample } from '../api/beehive';

export default function SampleClips({ onResult, loading, setLoading, modelReady }) {
  const [clips, setClips]   = useState([]);
  const [active, setActive] = useState(null); // currently selected clip name

  useEffect(() => {
    getSamples()
      .then(setClips)
      .catch(() => setClips([]));
  }, []);

  const handleClick = async (clip) => {
    if (loading || modelReady === false) return;
    setActive(clip.name);
    setLoading(true);
    try {
      const result = await predictSample(clip.name);
      onResult(result, result.is_unseen || clip.is_unseen);
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message;
      alert(`Sample prediction failed: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card">
      <div className="card-title">
        📂 Demo Clips
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
          — click to analyse instantly
        </span>
      </div>

      {clips.length === 0 ? (
        <div className="no-samples-msg">
          No demo clips yet.<br />
          Add <code>.wav</code> files to <code>backend/samples/</code> to enable this panel.<br />
          <span style={{ opacity: 0.5, fontSize: '0.75rem' }}>
            Tip: name them like <code>known_hive_queen.wav</code> or <code>unseen_hive_noq.wav</code>
          </span>
        </div>
      ) : (
        <>
          {/* Legend */}
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
            <span className="sample-tag known">● Known hive</span>
            <span className="sample-tag unseen">● Unseen test hive</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Select an "unseen" clip to see the generalization gap in action
            </span>
          </div>

          <div className="samples-grid">
            {clips.map((clip) => (
              <button
                key={clip.name}
                className={`sample-btn ${clip.is_unseen ? 'unseen' : ''} ${active === clip.name && loading ? 'loading' : ''}`}
                onClick={() => handleClick(clip)}
                disabled={loading || modelReady === false}
                title={clip.is_unseen ? 'This hive was held out during training' : 'This hive was seen during training'}
              >
                <span className={`sample-tag ${clip.is_unseen ? 'unseen' : 'known'}`}>
                  {clip.is_unseen ? 'UNSEEN' : 'KNOWN'}
                </span>
                <br />
                {clip.name.replace('.wav', '').replace(/_/g, ' ')}
                <br />
                <span style={{ fontSize: '0.72rem', opacity: 0.5 }}>
                  {clip.size_kb} KB
                  {clip.label !== 'unknown' && ` · ${clip.label.replace('_', ' ')}`}
                </span>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
