import { SAMPLE_CLIPS } from "../services/mockData";

export default function SampleClips({ activeSampleId, onSelect, disabled }) {
  return (
    <div className="sample-clips">
      <p className="sample-clips__label">Quick Demo Clips</p>
      <div className="sample-clips__grid">
        {SAMPLE_CLIPS.map((sample) => (
          <button
            key={sample.id}
            type="button"
            className={`sample-chip ${activeSampleId === sample.id ? "sample-chip--active" : ""}`}
            onClick={() => onSelect(sample.id)}
            disabled={disabled}
            title={sample.note}
          >
            {sample.label}
          </button>
        ))}
      </div>
    </div>
  );
}
