import { useState } from "react";
import Header from "./components/Header";
import Panel from "./components/Panel";
import AudioUploader from "./components/AudioUploader";
import SampleClips from "./components/SampleClips";
import BioacousticMetrics from "./components/BioacousticMetrics";
import SpectrogramDisplay from "./components/SpectrogramDisplay";
import ResultCard from "./components/ResultCard";
import ModelComparison from "./components/ModelComparison";
import OODBadge from "./components/OODBadge";
import { useAudioAnalysis } from "./hooks/useAudioAnalysis";
import { DEMO_MODE } from "./config";

export default function App() {
  const [viewMode, setViewMode] = useState("compare");
  const { clip, stage, status, result, error, analyzeFile, analyzeSample } = useAudioAnalysis();

  const isAnalyzing = status === "analyzing";
  const activeSampleId = clip?.source === "sample" ? clip.sampleId : null;

  return (
    <div className="app-shell">
      <Header status={status} viewMode={viewMode} onViewModeChange={setViewMode} />

      {status === "error" && (
        <div className="banner banner--error">
          Couldn't reach the backend: {error}. {DEMO_MODE ? "" : "Double-check it's running, or flip VITE_DEMO_MODE=true to rehearse offline."}
        </div>
      )}

      {isAnalyzing && (
        <div className="banner banner--progress">
          <span className="banner__spinner" aria-hidden="true" />
          {stage}
        </div>
      )}

      <main className="dashboard-grid">
        <Panel icon="🎧" title="1. Audio Input" className="panel--audio">
          <AudioUploader clip={clip} onFileSelected={analyzeFile} disabled={isAnalyzing} />
          <SampleClips activeSampleId={activeSampleId} onSelect={analyzeSample} disabled={isAnalyzing} />
        </Panel>

        <Panel icon="📊" title="2. Bioacoustic Metrics" className="panel--metrics">
          <BioacousticMetrics metrics={result?.metrics} />
        </Panel>

        <div className="headline-row">
          <ResultCard models={result?.models} viewMode={viewMode} />
          <OODBadge hiveType={result?.hiveType} />
        </div>

        <Panel icon="🔬" title="4. Multi-Model Prediction" subtitle="Baseline CNN vs. CRNN vs. DANN" className="panel--models">
          <ModelComparison models={result?.models} viewMode={viewMode} />
        </Panel>

        <Panel icon="🌈" title="3. Log-Mel Spectrogram" className="panel--spectrogram">
          <SpectrogramDisplay spectrogram={result?.spectrogram} isDemo={DEMO_MODE} />
        </Panel>
      </main>

      <footer className="app-footer">
        Beehive Health Monitor · Cross-hive generalization demo · {DEMO_MODE ? "Demo mode — no backend connected" : "Live backend"}
      </footer>
    </div>
  );
}
