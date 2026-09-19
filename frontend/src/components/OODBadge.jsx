export default function OODBadge({ hiveType }) {
  if (!hiveType) return null;

  const isUnseen = hiveType === "unseen";

  return (
    <div className={`ood-badge ${isUnseen ? "ood-badge--warning" : "ood-badge--ok"}`}>
      <span className="ood-badge__icon" aria-hidden="true">{isUnseen ? "⚠️" : "✅"}</span>
      <span>
        {isUnseen ? "Unseen Hive Hardware — Out-of-Distribution Test" : "Standard Apiary — In-Distribution"}
      </span>
    </div>
  );
}
