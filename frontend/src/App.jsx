// ============================================================
// App.jsx — Main application shell
// Assembles all components, manages shared state
// ============================================================

import { useState, useEffect } from 'react';
import { checkHealth } from './api/beehive';
import AudioUploader from './components/AudioUploader';
import ResultCard from './components/ResultCard';
import SpectrogramDisplay from './components/SpectrogramDisplay';
import SampleClips from './components/SampleClips';

export default function App() {
  const [modelStatus, setModelStatus] = useState(null); // null=loading, true=ready, false=not ready
  const [result, setResult]           = useState(null); // prediction result from API
  const [loading, setLoading]         = useState(false);
  const [showUnseen, setShowUnseen]   = useState(false); // tracks if current clip is from unseen hive

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then(({ model_ready }) => setModelStatus(model_ready))
      .catch(() => setModelStatus(false));
  }, []);

  const handleResult = (data, isUnseen = false) => {
    setResult(data);
    setShowUnseen(isUnseen);
  };

  return (
    <div className="app-container">

      {/* ---- Header ---- */}
      <header className="app-header">
        <div className="bee-icon">🐝</div>
        <h1>Beehive Health Monitor</h1>
        <p className="subtitle">
          Deep learning audio classification for bee colony health — queen presence detection
        </p>

        {/* Model status badge */}
        <div className="model-badge">
          <span className={`dot ${modelStatus ? '' : 'offline'}`} />
          {modelStatus === null  && 'Connecting to model...'}
          {modelStatus === true  && 'Model ready · CRNN'}
          {modelStatus === false && 'Model not loaded — train first'}
        </div>
      </header>

      {/* ---- Upload & Analyse ---- */}
      <AudioUploader
        onResult={handleResult}
        loading={loading}
        setLoading={setLoading}
        modelReady={modelStatus}
      />

      {/* ---- Prediction Result ---- */}
      {result && (
        <>
          <ResultCard result={result} isUnseen={showUnseen} />
          {result.spectrogram_b64 && (
            <SpectrogramDisplay b64={result.spectrogram_b64} />
          )}
        </>
      )}

      {/* ---- Pre-loaded Demo Clips ---- */}
      <SampleClips
        onResult={handleResult}
        loading={loading}
        setLoading={setLoading}
        modelReady={modelStatus}
      />

    </div>
  );
}
