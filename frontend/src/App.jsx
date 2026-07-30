import { useEffect, useMemo, useState } from 'react'
import { askWorkspace, checkHealth, deleteWorkspace, listWorkspaces, uploadWorkspace } from './api'
import ChatPanel from './components/ChatPanel'
import UploadPanel from './components/UploadPanel'
import WorkspacePanel from './components/WorkspacePanel'

export default function App() {
  const [health, setHealth] = useState('checking')
  const [workspaces, setWorkspaces] = useState([])
  const [selectedId, setSelectedId] = useState(() => localStorage.getItem('rag-workspace-id') || '')
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [deleting, setDeleting] = useState('')
  const [notice, setNotice] = useState(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('rag-theme') || 'light')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('rag-theme', theme)
  }, [theme])

  const selectedWorkspace = useMemo(
    () => workspaces.find((workspace) => workspace.workspace_id === selectedId) || null,
    [workspaces, selectedId],
  )

  useEffect(() => {
    async function initialise() {
      try {
        await checkHealth()
        setHealth('online')
        const data = await listWorkspaces()
        setWorkspaces(data)
        if (!data.some((workspace) => workspace.workspace_id === selectedId) && data[0]) {
          selectWorkspace(data[0].workspace_id)
        }
      } catch (error) {
        setHealth('offline')
        setNotice({ type: 'error', text: error.message })
      }
    }
    initialise()
  }, [])

  function selectWorkspace(workspaceId) {
    setSelectedId(workspaceId)
    localStorage.setItem('rag-workspace-id', workspaceId)
  }

  async function handleUpload(file, title) {
    setUploading(true)
    setNotice(null)
    try {
      const workspace = await uploadWorkspace(file, title)
      setWorkspaces((current) => [workspace, ...current.filter((item) => item.workspace_id !== workspace.workspace_id)])
      selectWorkspace(workspace.workspace_id)
      setNotice({ type: 'success', text: `Indexed ${workspace.text_file_count} files into ${workspace.chunk_count} chunks.` })
    } catch (error) {
      setNotice({ type: 'error', text: error.message })
      throw error
    } finally {
      setUploading(false)
    }
  }

  async function handleAsk(payload) {
    if (!selectedWorkspace) throw new Error('Select a workspace first.')
    setAsking(true)
    setNotice(null)
    try {
      return await askWorkspace(selectedWorkspace.workspace_id, payload)
    } finally {
      setAsking(false)
    }
  }

  async function handleDelete(workspace) {
    if (!window.confirm(`Delete “${workspace.title}” and its vector index?`)) return
    setDeleting(workspace.workspace_id)
    setNotice(null)
    try {
      await deleteWorkspace(workspace.workspace_id)
      const next = workspaces.filter((item) => item.workspace_id !== workspace.workspace_id)
      setWorkspaces(next)
      if (selectedId === workspace.workspace_id) {
        const nextId = next[0]?.workspace_id || ''
        setSelectedId(nextId)
        if (nextId) localStorage.setItem('rag-workspace-id', nextId)
        else localStorage.removeItem('rag-workspace-id')
      }
      setNotice({ type: 'success', text: 'Workspace deleted.' })
    } catch (error) {
      setNotice({ type: 'error', text: error.message })
    } finally {
      setDeleting('')
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">R</span>
          <span><strong>Text RAG</strong><small>Workspace</small></span>
        </a>
        <div className="topbar-right">
          <div className={`server-status ${health}`}><span /> API {health}</div>
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </header>
      {notice && <div className={`notice ${notice.type}`}><span>{notice.text}</span><button onClick={() => setNotice(null)}>×</button></div>}
      <main className="main-layout">
        <aside className="sidebar">
          <UploadPanel busy={uploading} onUpload={handleUpload} />
          <WorkspacePanel
            workspaces={workspaces}
            selectedId={selectedId}
            onSelect={selectWorkspace}
            onDelete={handleDelete}
            deleting={deleting}
          />
        </aside>
        <ChatPanel workspace={selectedWorkspace} asking={asking} onAsk={handleAsk} />
      </main>
    </div>
  )
}
