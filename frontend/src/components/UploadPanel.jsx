import { useRef, useState } from 'react'

export default function UploadPanel({ busy, onUpload }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [dragging, setDragging] = useState(false)

  function chooseFile(nextFile) {
    if (!nextFile) return
    if (!nextFile.name.toLowerCase().endsWith('.zip')) {
      window.alert('Choose a ZIP file containing TXT files.')
      return
    }
    setFile(nextFile)
    if (!title) setTitle(nextFile.name.replace(/\.zip$/i, ''))
  }

  async function submit(event) {
    event.preventDefault()
    if (!file || busy) return
    try {
      await onUpload(file, title)
      setFile(null)
      setTitle('')
      if (inputRef.current) inputRef.current.value = ''
    } catch {
      // The parent displays the API error and keeps the selected file for retry.
    }
  }

  return (
    <section className="panel upload-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Knowledge base</span>
          <h2>Upload documents</h2>
        </div>
        <span className="step-badge">1</span>
      </div>
      <form onSubmit={submit}>
        <button
          className={`drop-zone ${dragging ? 'dragging' : ''}`}
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault()
            setDragging(false)
            chooseFile(event.dataTransfer.files?.[0])
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".zip,application/zip"
            hidden
            onChange={(event) => chooseFile(event.target.files?.[0])}
          />
          <span className="upload-icon" aria-hidden="true">↑</span>
          <strong>{file ? file.name : 'Drop a ZIP file here'}</strong>
          <small>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB selected` : 'The ZIP may contain folders, but documents must be .txt files.'}</small>
        </button>
        <label className="field-label" htmlFor="workspace-title">Workspace name</label>
        <input
          id="workspace-title"
          className="text-input"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Example: Python documentation"
          maxLength={120}
        />
        <button className="primary-button" type="submit" disabled={!file || busy}>
          {busy ? <><span className="spinner" /> Extracting and indexing…</> : 'Create RAG workspace'}
        </button>
      </form>
      <p className="panel-note">Each upload receives its own index. Existing backend data and configuration are not changed.</p>
    </section>
  )
}
