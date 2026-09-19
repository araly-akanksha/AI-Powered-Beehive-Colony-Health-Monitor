import { PRESENT_LABEL } from "../config";

export default function ModelComparison({ models, viewMode }) {
  if (!models || models.length === 0) {
    return <p className="model-comparison__empty">Run a clip to compare all three models side by side.</p>;
  }

  const bestKey = models.reduce((best, m) => (m.confidence > best.confidence ? m : best), models[0]).key;

  return (
    <ul className="model-comparison">
      {models.map((m) => {
        const isPresent = m.label === PRESENT_LABEL;
        const isHighlighted = viewMode === "compare" ? m.key === bestKey : m.key === viewMode;
        const confidencePct = Math.round(m.confidence * 100);

        return (
          <li
            key={m.key}
            className={`model-row ${isHighlighted ? "model-row--highlighted" : ""} ${isPresent ? "model-row--present" : "model-row--absent"}`}
          >
            <div className="model-row__name">
              <span>{m.name}</span>
              {isHighlighted && <span aria-hidden="true">✅</span>}
            </div>
            <div className="model-row__bar">
              <div className="model-row__bar-fill" style={{ width: `${confidencePct}%` }} />
            </div>
            <div className="model-row__meta">
              <span className="model-row__pct">{confidencePct}%</span>
              <span className="model-row__tag">{m.tag}</span>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
