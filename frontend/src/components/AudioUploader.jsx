// ============================================================
// components/AudioUploader.jsx
// Drag-and-drop WAV upload with wavesurfer.js waveform
// ============================================================

import { useState, useRef, useEffect } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { predictFile } from '../api/beehive';

export default function AudioUploader({ onResult, loading, setLoading, modelReady }) {
  const [file, setFile]         = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [duration, setDuration] = useState(null);
  const [playing, setPlaying]   = useState(false);

  const waveRef    = useRef(null);
  const surferRef  = useRef(null);
  const inputRef   = useRef(null);

  // Initialise WaveSurfer when a file is loaded
  useEffect(() => {
    if (!file || !waveRef.current) return;

    // Destroy previous instance if any
    if (surferRef.current) { surferRef.current.destroy(); }

    surferRef.current = WaveSurfer.create({
      container:      waveRef.current,
      waveColor:      'rgba(251, 191, 36, 0.4)',
      progressColor:  '#f59e0b',
      cursorColor:    '#fbbf24',
      barWidth:       2,
      barGap:         1,
      barRadius:      2,
      height:         64,
      normalize:      true,
      backend:        'WebAudio',
    });

    const url = URL.createObjectURL(file);
    surferRef.current.load(url);
    surferRef.current.on('ready', () => {
      setDuration(surferRef.current.getDuration());
    });
    surferRef.current.on('finish', () => setPlaying(false));

    return () => {
      URL.revokeObjectURL(url);
      if (surferRef.current) surferRef.current.destroy();
    };
  }, [file]);

  const handleFileChange = (f) => {
    if (!f || !f.name.toLowerCase().endsWith('.wav')) {
      alert('Please select a .wav file');
      return;
    }
    setFile(f);
    setDuration(null);
    setPlaying(false);
  };

  const togglePlay = () => {
    if (!surferRef.current) return;
    surferRef.current.playPause();
    setPlaying(!playing);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const result = await predictFile(file);
      onResult(result, false);
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message;
      alert(`Prediction failed: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setDuration(null);
    if (inputRef.current) inputRef.current.value = '';
    if (surferRef.current) surferRef.current.destroy();
  };

  const formatDuration = (s) => {
    if (!s) return '...';
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1);
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  return (
    <div className="glass-card">
      <div className="card-title">🎙️ Audio Input</div>

      {!file ? (
        /* Drop zone */
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFileChange(e.dataTransfer.files[0]); }}
          onClick={() => inputRef.current?.click()}
        >
          <div className="upload-icon">🎵</div>
          <p>Drag &amp; drop a <span>.wav</span> file, or click to browse</p>
          <p style={{ fontSize: '0.78rem', marginTop: '6px', opacity: 0.5 }}>
            Supports any length — will be auto-segmented into {2}s windows
          </p>
          <input
            ref={inputRef}
            type="file"
            accept=".wav,audio/wav"
            style={{ display: 'none' }}
            onChange={(e) => handleFileChange(e.target.files[0])}
          />
        </div>
      ) : (
        /* File loaded state */
        <>
          <div className="file-loaded">
            <span style={{ fontSize: '2rem' }}>🔊</span>
            <div className="file-info">
              <div className="file-name">{file.name}</div>
              <div className="file-meta">
                {(file.size / 1024).toFixed(1)} KB
                {duration && ` · ${formatDuration(duration)}`}
              </div>
            </div>
            <button className="btn-secondary" onClick={togglePlay} title="Play/pause">
              {playing ? '⏸' : '▶️'}
            </button>
            <button className="btn-secondary" onClick={clearFile} title="Remove">✕</button>
          </div>

          {/* Waveform */}
          <div id="waveform" ref={waveRef} />

          {/* Analyse button */}
          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={loading || modelReady === false}
          >
            {loading ? (
              <><div className="spinner" /> Analysing audio...</>
            ) : modelReady === false ? (
              '⚠ Model not trained yet'
            ) : (
              '🔍 Analyse Hive Health'
            )}
          </button>
        </>
      )}
    </div>
  );
}
