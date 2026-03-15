import { useState, useEffect } from 'react'
import Link from 'next/link'
import axios from 'axios'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function HealthPage() {
    const [health, setHealth] = useState(null)
    const [docs, setDocs] = useState([])

    useEffect(() => {
        axios.get(`${API}/health`).then(r => setHealth(r.data)).catch(() => { })
        axios.get(`${API}/documents`).then(r => setDocs(r.data.documents)).catch(() => { })
    }, [])

    return (
        <div className="container">
            <div className="header">
                <h1>🔒 Adaptive RAG Security Assistant</h1>
                <p>GDPR-Compliant · 100% Local · LangGraph Pipeline</p>
                <div className="gdpr-badge">● Zero data leaves your machine</div>
            </div>

            <nav className="nav">
                <Link href="/">Upload</Link>
                <Link href="/chat">Ask Questions</Link>
                <Link href="/health" className="active">System Health</Link>
            </nav>

            <div className="card">
                <div className="section-label">System Status</div>

                {health ? (
                    <>
                        <div className="health-row">
                            <span>Backend Status</span>
                            <span className="status-ok">● {health.status}</span>
                        </div>
                        <div className="health-row">
                            <span>Ollama LLM</span>
                            <span className={health.ollama_status === 'connected'
                                ? 'status-ok' : 'status-err'}>
                                ● {health.ollama_status}
                            </span>
                        </div>
                        <div className="health-row">
                            <span>GDPR Compliant</span>
                            <span className={health.local_inference ? "status-ok" : "status-err"}>
                                {health.local_inference ? "● Yes — 100% local" : "● No — Cloud inference in use"}
                            </span>
                        </div>
                        <div className="health-row">
                            <span>Inference Mode</span>
                            <span className="status-ok">
                                {health.local_inference ? "● Local (Ollama)" : "● Cloud (Ollama)"}
                            </span>
                        </div>
                        <div className="health-row">
                            <span>Vector Store</span>
                            <span className="status-ok">● FAISS (local)</span>
                        </div>
                        <div className="health-row">
                            <span>Pipeline</span>
                            <span className="status-ok">● LangGraph Adaptive RAG</span>
                        </div>
                    </>
                ) : (
                    <p className="loading">Checking system status...</p>
                )}
            </div>

            {docs.length > 0 && (
                <div className="card">
                    <div className="section-label">Indexed Documents ({docs.length})</div>
                    {docs.map((doc, i) => (
                        <div key={i} className="health-row">
                            <span>📄 {doc.name}</span>
                            <span className="status-ok">● Ready</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
