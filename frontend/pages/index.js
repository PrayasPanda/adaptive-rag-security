import { useState, useRef } from 'react'
import Link from 'next/link'
import axios from 'axios'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function UploadPage() {
    const [file, setFile] = useState(null)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const fileRef = useRef()

    const handleFile = (e) => {
        setFile(e.target.files[0])
        setResult(null)
        setError(null)
    }

    const handleUpload = async () => {
        if (!file) return
        setLoading(true)
        setError(null)
        const form = new FormData()
        form.append('file', file)
        try {
            const res = await axios.post(`${API}/upload`, form)
            setResult(res.data)
        } catch (e) {
            setError(e.response?.data?.detail || 'Upload failed')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="container">
            <div className="header">
                <h1>🔒 Adaptive RAG Security Assistant</h1>
                <p>GDPR-Compliant · 100% Local · LangGraph Pipeline</p>
                <div className="gdpr-badge">● Zero data leaves your machine</div>
            </div>

            <nav className="nav">
                <Link href="/" className="active">Upload</Link>
                <Link href="/chat">Ask Questions</Link>
                <Link href="/health">System Health</Link>
            </nav>

            <div className="card">
                <div className="upload-zone" onClick={() => fileRef.current.click()}>
                    <input ref={fileRef} type="file" accept=".pdf" onChange={handleFile} />
                    {file
                        ? <p style={{ color: '#00d4ff' }}>📄 {file.name}</p>
                        : <>
                            <p style={{ fontSize: '28px', marginBottom: '10px' }}>📑</p>
                            <p style={{ color: '#475569' }}>Click to upload security PDF</p>
                            <p style={{ color: '#1f2937', fontSize: '11px', marginTop: '6px' }}>
                                CVE reports · OWASP docs · Pentest reports · Security advisories
                            </p>
                        </>
                    }
                </div>

                {file && (
                    <div style={{ textAlign: 'center', marginTop: '16px' }}>
                        <button
                            className="btn btn-primary"
                            onClick={handleUpload}
                            disabled={loading}
                        >
                            {loading ? 'Indexing...' : 'Upload & Index Document'}
                        </button>
                    </div>
                )}

                {loading && (
                    <div className="loading">
                        <p>⚙️ Creating semantic chunks...</p>
                        <p>⚙️ Generating OpenAI embeddings...</p>
                        <p>⚙️ Building FAISS index...</p>
                    </div>
                )}

                {result && (
                    <div className="success">
                        <p>✅ <strong>{result.doc_name}</strong> indexed successfully</p>
                        <p style={{ marginTop: '6px' }}>
                            📦 {result.chunks_indexed} chunks indexed in FAISS
                        </p>
                        <p style={{ marginTop: '10px' }}>
                            <Link href="/chat" style={{ color: '#00d4ff' }}>
                                → Start asking questions
                            </Link>
                        </p>
                    </div>
                )}

                {error && <p className="error">❌ {error}</p>}
            </div>
        </div>
    )
}
