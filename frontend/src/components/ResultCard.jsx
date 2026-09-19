import { PRESENT_LABEL } from "../config";

export default function ResultCard({ models, viewMode }) {
  if (!models || models.length === 0) {
    return (
      <div className="result-card result-card--empty">
        <p>No clip analyzed yet. Drop a file or pick a demo clip to see a diagnosis here.</p>
      </div>
    );
  }

  const headlineModel =
    viewMode === "compare"
      ? models.find((m) => m.key === "dann") || models[models.length - 1]
      : models.find((m) => m.key === viewMode) || models[0];

  const isPresent = headlineModel.label === PRESENT_LABEL;
  const confidencePct = Math.round(headlineModel.confidence * 100);

  return (
    <div className={`result-card ${isPresent ? "result-card--healthy" : "result-card--distress"}`}>
      <div className="result-card__badge">
        <span aria-hidden="true">{isPresent ? "🟢" : "🔴"}</span>
        <div>
          <p className="result-card__label">{isPresent ? "QUEEN PRESENT" : "QUEEN ABSENT"}</p>
          <p className="result-card__sublabel">{isPresent ? "Healthy Colony" : "Colony in Distress"}</p>
        </div>
      </div>

      <div className="result-card__confidence">
        <div className="confidence-bar" role="progressbar" aria-valuenow={confidencePct} aria-valuemin={0} aria-valuemax={100}>
          <div className="confidence-bar__fill" style={{ width: `${confidencePct}%` }} />
        </div>
        <p className="result-card__confidence-label">
          {confidencePct}% confidence · per {headlineModel.name}
        </p>
      </div>
    </div>
  );
}
