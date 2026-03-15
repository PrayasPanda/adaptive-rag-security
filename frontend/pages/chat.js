import { useState, useEffect } from 'react'
import Link from 'next/link'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function ChatPage() {
    const [question, setQuestion] = useState('')
    const [docName, setDocName] = useState('')
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [availableDocs, setAvailableDocs] = useState([])

    useEffect(() => {
        // Fetch available documents on mount
        axios.get(`${API}/documents`)
            .then(res => {
                setAvailableDocs(res.data.documents)
                if (res.data.documents.length > 0) {
                    setDocName(res.data.documents[0].name)
                }
            })
            .catch(err => console.error("Failed to load documents:", err))
    }, [])

    const handleQuery = async () => {
        if (!question.trim() || !docName.trim()) return
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const res = await axios.post(`${API}/query`, {
                question,
                doc_name: docName.toLowerCase().replace(/ /g, '_')
            })
            setResult(res.data)
        } catch (e) {
            setError(e.response?.data?.detail || 'Query failed')
        } finally {
            setLoading(false)
        }
    }

    const strategyClass = (s) => {
        if (!s) return ''
        if (s === 'SIMPLE') return 'strategy-simple'
        if (s === 'MULTI') return 'strategy-multi'
        return 'strategy-decompose'
    }

    return (
        <div className="container">
            <div className="header">
                <h1>🔒 Adaptive RAG Security Assistant</h1>
                <p>GDPR-Compliant · 100% Local · LangGraph Pipeline</p>
                <div className="gdpr-badge">● Zero data leaves your machine</div>
            </div>

            <nav className="nav">
                <Link href="/">Upload</Link>
                <Link href="/chat" className="active">Ask Questions</Link>
                <Link href="/health">System Health</Link>
            </nav>

            <div className="card">
                <div className="input-row">
                    {availableDocs.length > 0 ? (
                        <select
                            value={docName}
                            onChange={e => setDocName(e.target.value)}
                            style={{
                                maxWidth: '240px',
                                padding: '11px 14px',
                                background: '#111827',
                                border: '1px solid rgba(255,255,255,0.08)',
                                borderRadius: '8px',
                                color: '#e2e8f0',
                                fontFamily: 'inherit',
                                fontSize: '13px',
                                outline: 'none',
                                appearance: 'none',
                                cursor: 'pointer'
                            }}
                        >
                            {availableDocs.map((doc, idx) => (
                                <option key={idx} value={doc.name}>{doc.name}</option>
                            ))}
                        </select>
                    ) : (
                        <input
                            type="text"
                            placeholder="No documents uploaded yet..."
                            disabled
                            style={{ maxWidth: '240px', padding: '11px 14px', background: '#111827', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', color: '#e2e8f0' }}
                        />
                    )}
                    <input
                        type="text"
                        placeholder="Ask a security question..."
                        value={question}
                        onChange={e => setQuestion(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleQuery()}
                    />
                    <button
                        className="btn btn-primary"
                        onClick={handleQuery}
                        disabled={loading || !question || !docName}
                    >
                        {loading ? '...' : 'Ask'}
                    </button>
                </div>

                <p style={{ fontSize: '11px', color: '#1f2937' }}>
                    Try: "What are the critical vulnerabilities?" ·
                    "What fixes are recommended?" ·
                    "Compare injection and XSS risks"
                </p>

                {loading && (
                    <div className="loading">
                        🧠 Running adaptive RAG pipeline...
                    </div>
                )}

                {error && <p className="error">❌ {error}</p>}

                {result && (
                    <>
                        <div className="section-label">Answer</div>
                        <div className="answer-box">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {result.answer}
                            </ReactMarkdown>
                        </div>

                        {result.sources?.length > 0 && (
                            <>
                                <div className="section-label">
                                    Sources ({result.sources.length})
                                </div>
                                {result.sources.map((src, i) => (
                                    <div key={i} className="source-card">
                                        <div className="source-page">📄 Page {src.page}</div>
                                        <div className="source-preview">{src.content_preview}</div>
                                    </div>
                                ))}
                            </>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}
