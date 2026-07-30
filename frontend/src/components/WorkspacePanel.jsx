function formatDate(value) {
  if (!value) return ''
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
}

export default function WorkspacePanel({ workspaces, selectedId, onSelect, onDelete, deleting }) {
  return (
    <section className="panel workspace-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Indexed collections</span>
          <h2>Workspaces</h2>
        </div>
        <span className="step-badge">2</span>
      </div>
      {workspaces.length === 0 ? (
        <div className="empty-state compact">
          <strong>No workspace yet</strong>
          <span>Upload a ZIP file to create your first searchable collection.</span>
        </div>
      ) : (
        <div className="workspace-list">
          {workspaces.map((workspace) => (
            <article
              className={`workspace-card ${selectedId === workspace.workspace_id ? 'selected' : ''}`}
              key={workspace.workspace_id}
            >
              <button className="workspace-select" onClick={() => onSelect(workspace.workspace_id)}>
                <span className={`status-dot ${workspace.status}`} />
                <span className="workspace-copy">
                  <strong>{workspace.title}</strong>
                  <small>{workspace.text_file_count ?? 0} files · {workspace.chunk_count ?? 0} chunks</small>
                  <small>{formatDate(workspace.indexed_at || workspace.created_at)}</small>
                </span>
              </button>
              <button
                className="icon-button danger"
                type="button"
                title="Delete workspace"
                disabled={deleting === workspace.workspace_id}
                onClick={() => onDelete(workspace)}
              >
                {deleting === workspace.workspace_id ? '…' : '×'}
              </button>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
