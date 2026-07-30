export default function SourceCard({ source, citationIds }) {
  const cited = citationIds.has(source.chunk_id)
  return (
    <details className={`source-card ${cited ? 'cited' : ''}`}>
      <summary>
        <span className="source-rank">#{source.rank}</span>
        <span className="source-title">
          <strong>{source.relative_path}</strong>
          <small>{source.section_path?.join(' › ') || 'Document root'}</small>
        </span>
        <span className="source-score">{source.score.toFixed(4)}</span>
      </summary>
      <div className="source-body">
        {cited && <span className="cited-label">Used in answer</span>}
        <pre>{source.text}</pre>
        <code>{source.chunk_id}</code>
      </div>
    </details>
  )
}
