import { useEffect, useRef, useState } from 'react'
import SourceCard from './SourceCard'

export default function ChatPanel({ workspace, asking, onAsk }) {
  const [question, setQuestion] = useState('')
  const [retriever, setRetriever] = useState('bm25')
  const [generator, setGenerator] = useState('extractive')
  const [topK, setTopK] = useState(5)
  const [messages, setMessages] = useState([])
  const bottomRef = useRef(null)

  useEffect(() => {
    setMessages([])
    setQuestion('')
  }, [workspace?.workspace_id])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, asking])

  async function submit(event) {
    event.preventDefault()
    const cleanQuestion = question.trim()
    if (!cleanQuestion || !workspace || asking) return
    setQuestion('')
    setMessages((current) => [...current, { role: 'user', question: cleanQuestion }])
    try {
      const response = await onAsk({ question: cleanQuestion, retriever, generator, top_k: Number(topK) })
      setMessages((current) => [...current, { role: 'assistant', response }])
    } catch (error) {
      setMessages((current) => [...current, { role: 'error', message: error.message }])
    }
  }

  const examples = [
    'Summarise the main concepts in these documents.',
    'What does the documentation say about this topic?',
    'Compare the two approaches described in the files.',
  ]

  return (
    <section className="chat-shell">
      <div className="chat-header">
        <div>
          <span className="eyebrow">Grounded question answering</span>
          <h1>{workspace ? workspace.title : 'Select a workspace'}</h1>
          <p>{workspace ? `${workspace.text_file_count} text files indexed into ${workspace.chunk_count} chunks.` : 'Upload or select a document workspace before asking questions.'}</p>
        </div>
        <span className="step-badge large">3</span>
      </div>

      <div className="chat-content">
        {!workspace ? (
          <div className="empty-state hero-empty">
            <span className="empty-symbol">⌁</span>
            <strong>Your document assistant is ready</strong>
            <span>Create a workspace on the left, then ask questions grounded in its text files.</span>
          </div>
        ) : messages.length === 0 ? (
          <div className="starter-area">
            <div className="empty-state">
              <span className="empty-symbol">⌕</span>
              <strong>Ask about {workspace.title}</strong>
              <span>The answer will include retrieved chunks and source locations.</span>
            </div>
            <div className="example-grid">
              {examples.map((example) => <button key={example} onClick={() => setQuestion(example)}>{example}</button>)}
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((message, index) => {
              if (message.role === 'user') return <div className="message user-message" key={index}><span>You</span><p>{message.question}</p></div>
              if (message.role === 'error') return <div className="message error-message" key={index}><span>Error</span><p>{message.message}</p></div>
              const { response } = message
              const citationIds = new Set(response.citations.map((citation) => citation.chunk_id))
              return (
                <div className="message assistant-message" key={index}>
                  <div className="assistant-label"><span>RAG</span><small>{response.metadata?.retrieval_seconds ? `${(response.metadata.retrieval_seconds * 1000).toFixed(1)} ms retrieval` : ''}</small></div>
                  <p className="answer-text">{response.answer}</p>
                  <div className="answer-status">{response.answerable ? 'Answer supported by retrieved context' : 'Insufficient context for a reliable answer'}</div>
                  <div className="sources-heading"><strong>Retrieved sources</strong><span>{response.retrieved.length} chunks</span></div>
                  <div className="source-list">
                    {response.retrieved.map((source) => <SourceCard key={source.chunk_id} source={source} citationIds={citationIds} />)}
                  </div>
                </div>
              )
            })}
            {asking && <div className="message assistant-message loading-message"><div className="assistant-label"><span>RAG</span></div><p><span className="spinner dark" /> Retrieving evidence and preparing the answer…</p></div>}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        <div className="rag-controls">
          <label>Retriever
            <select value={retriever} onChange={(event) => setRetriever(event.target.value)} disabled={!workspace || asking}>
              <option value="bm25">BM25</option>
              <option value="dense">Dense</option>
            </select>
          </label>
          <label>Generator
            <select value={generator} onChange={(event) => setGenerator(event.target.value)} disabled={!workspace || asking}>
              <option value="extractive">Extractive</option>
              <option value="openai-compatible">Local LLM</option>
            </select>
          </label>
          <label>Top K
            <select value={topK} onChange={(event) => setTopK(event.target.value)} disabled={!workspace || asking}>
              {[3, 5, 8, 10, 15, 20].map((value) => <option value={value} key={value}>{value}</option>)}
            </select>
          </label>
        </div>
        <div className="question-row">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }}
            placeholder={workspace ? 'Ask a question about the uploaded files…' : 'Select a workspace first…'}
            disabled={!workspace || asking}
            rows={2}
          />
          <button className="send-button" disabled={!workspace || asking || !question.trim()}>{asking ? '…' : 'Ask'}</button>
        </div>
        <small className="composer-note">Enter to send · Shift + Enter for a new line</small>
      </form>
    </section>
  )
}
