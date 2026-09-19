export default function Panel({ icon, title, subtitle, children, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel__header">
        <span className="panel__icon" aria-hidden="true">{icon}</span>
        <div>
          <h2>{title}</h2>
          {subtitle && <p className="panel__subtitle">{subtitle}</p>}
        </div>
      </header>
      <div className="panel__body">{children}</div>
    </section>
  );
}
