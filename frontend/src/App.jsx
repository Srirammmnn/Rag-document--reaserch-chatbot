import { useState, useRef, useEffect } from 'react';
import { Bot, Send, User, Upload, Settings, RefreshCw, FileText, ChevronRight, Layers, Database, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_URL = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [health, setHealth] = useState({ status: 'checking', vectorstore_loaded: false, total_vectors: -1 });
  const [sourcesInfo, setSourcesInfo] = useState({ sources: [], total_chunks: 0 });
  const [sessionId] = useState(() => (window.crypto && window.crypto.randomUUID) ? window.crypto.randomUUID() : Date.now().toString());
  
  const endOfMessagesRef = useRef(null);

  useEffect(() => {
    checkHealth();
    fetchSources();
    
    // Poll health status every 5 seconds
    const interval = setInterval(() => {
      checkHealth();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) {
        setHealth(await res.json());
      } else {
        setHealth({ status: 'error' });
      }
    } catch {
      setHealth({ status: 'error' });
    }
  };

  const fetchSources = async () => {
    try {
      const res = await fetch(`${API_URL}/sources`);
      if (res.ok) {
        setSourcesInfo(await res.json());
      }
    } catch {
      console.error("Could not fetch sources");
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    // Set uploading state (we can use health status as a proxy for UI)
    const toastId = addToast(`Uploading ${file.name}...`, 'info');
    
    try {
      const res = await fetch(`${API_URL}/ingest`, {
        method: 'POST',
        body: formData,
      });
      
      if (res.ok) {
        const data = await res.json();
        addToast(`Successfully added ${data.chunks_added} chunks!`, 'success');
        fetchSources();
        checkHealth();
      } else {
        addToast(`Upload failed: ${res.statusText}`, 'error');
      }
    } catch (err) {
      addToast(`Error: ${err.message}`, 'error');
    }
  };

  const [toasts, setToasts] = useState([]);
  const addToast = (msg, type) => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, msg, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 3000);
    return id;
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isTyping) return;

    const userMessage = { role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      // Stream processing via SSE
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.content, session_id: sessionId })
      });

      if (!res.ok) throw new Error("Failed to connect to agent");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantMsg = { role: 'assistant', content: '', tool_calls: [], sources: [] };
      
      setMessages(prev => [...prev, assistantMsg]);
      
      let done = false;
      let buffer = "";

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || "";
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'tool_call') {
                assistantMsg.tool_calls.push(data.tool);
              } else if (data.type === 'answer') {
                assistantMsg.content = data.content;
              } else if (data.type === 'tool_result') {
                // Not strictly displayed, but good to know
              } else if (data.type === 'done') {
                setIsTyping(false);
              } else if (data.type === 'error') {
                assistantMsg.content = `❌ Error: ${data.message}`;
                setIsTyping(false);
              }

              // Try to extract sources from answer text as a fallback
              if (assistantMsg.content.includes('(source:')) {
                const srcMatches = assistantMsg.content.match(/\(source:\s*([^\)]+)\)/g);
                if (srcMatches) {
                  const srcs = srcMatches.map(s => s.replace('(source: ', '').replace(')', '').trim());
                  assistantMsg.sources = [...new Set([...assistantMsg.sources, ...srcs])];
                }
              }

              // Update the last message
              setMessages(prev => {
                const newArr = [...prev];
                newArr[newArr.length - 1] = { ...assistantMsg };
                return newArr;
              });
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ Connection error: ${error.message}` }]);
    } finally {
      setIsTyping(false);
    }
  };

  const clearChat = async () => {
    setMessages([]);
    try {
      await fetch(`${API_URL}/session/${sessionId}`, { method: 'DELETE' });
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="app-container">
      {/* Toast Notifications */}
      <div className="toast-container">
        <AnimatePresence>
          {toasts.map(t => (
            <motion.div 
              key={t.id}
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className={`toast toast-${t.type}`}
            >
              {t.msg}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Sidebar */}
      <div className="sidebar glass-panel">
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="logo-icon"><Layers size={24} color="#f8fafc" /></div>
            <h2>NeuRAG</h2>
          </div>
          <p className="subtitle">Intelligence Platform</p>
        </div>

        <div className="sidebar-section">
          <h3>System Status</h3>
          <div className="status-card">
            <div className={`status-indicator ${health.status === 'ok' ? 'active' : 'error'}`}></div>
            <span>{health.status === 'ok' ? 'Backend Online' : 'Backend Offline'}</span>
          </div>
          <div className="status-card">
            <Database size={16} />
            <span>Pinecone: {health.vectorstore_loaded ? 'Connected' : 'Empty'}</span>
          </div>
        </div>

        <div className="sidebar-section">
          <h3>Knowledge Base</h3>
          <label className="upload-btn">
            <Upload size={16} />
            <span>Ingest Document</span>
            <input type="file" hidden accept=".pdf,.txt" onChange={handleFileUpload} />
          </label>
          
          <div className="sources-list">
            <div className="sources-header">
              <span style={{fontSize: '0.8rem', color: 'var(--text-muted)'}}>INDEXED FILES ({sourcesInfo.sources.length})</span>
            </div>
            {(!sourcesInfo || !sourcesInfo.sources || sourcesInfo.sources.length === 0) ? (
              <div className="empty-sources">No documents indexed</div>
            ) : (
              sourcesInfo.sources.map((s, i) => (
                <div key={i} className="source-item">
                  <FileText size={14} />
                  <span className="truncate">{typeof s === 'string' ? s : JSON.stringify(s)}</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <button onClick={clearChat} className="action-btn">
            <RefreshCw size={16} />
            <span>Clear Session</span>
          </button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-area glass-panel">
        <div className="chat-header" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', position: 'relative' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Bot size={20} color="var(--primary)" />
              <h3 className="text-gradient">Rag -Agent</h3>
            </div>
            <span className="session-id">Session: {sessionId.split('-')[0]}</span>
          </div>
          <div style={{ position: 'absolute', right: '2rem' }}>
            <button className="icon-btn" title="Settings"><Settings size={20} /></button>
          </div>
        </div>

        <div className="messages-container">
          <div className="chat-watermark">
            <Bot size={240} />
          </div>
          
          {messages.length === 0 ? (
            <div className="empty-chat" style={{ zIndex: 1 }}>
              <motion.div 
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="empty-chat-icon"
              >
                <Bot size={48} />
              </motion.div>
              <h2>Rag -Agent is ready.</h2>
              <p>Upload a document to the knowledge base and ask me anything about it.</p>
            </div>
          ) : (
            messages.map((msg, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`message-wrapper ${msg.role === 'user' ? 'user' : 'assistant'}`}
              >
                <div className="avatar">
                  {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                </div>
                <div className="message-content" style={{ zIndex: 1 }}>
                  <div className="message-header">
                    {msg.role === 'user' ? 'You' : 'Rag -Agent'}
                  </div>
                  
                  {msg.tool_calls && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0 && (
                    <div className="tool-calls">
                      {msg.tool_calls.map((tc, idx) => (
                         <span key={idx} className="tool-badge">
                           <ChevronRight size={12} /> Routed via {typeof tc === 'string' ? tc : JSON.stringify(tc)}
                         </span>
                      ))}
                    </div>
                  )}

                  <div className="bubble">
                    {msg.role === 'user' ? (
                      typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
                    ) : (
                      <div className="markdown-body">
                        <ReactMarkdown>
                          {String(msg.content) || (isTyping && i === messages.length - 1 ? '...' : '')}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                  
                  {msg.sources && Array.isArray(msg.sources) && msg.sources.length > 0 && (
                    <div className="message-sources">
                      {msg.sources.map((s, idx) => (
                        <span key={idx} className="source-pill"><FileText size={10} /> {typeof s === 'string' ? s : JSON.stringify(s)}</span>
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            ))
          )}
          
          {isTyping && (
             <div className="typing-indicator-wrapper">
               <div className="avatar"><Bot size={18} /></div>
               <div className="typing-dots">
                 <span></span><span></span><span></span>
               </div>
             </div>
          )}
          
          <div ref={endOfMessagesRef} />
        </div>

        <div className="input-area">
          <form onSubmit={sendMessage} className="input-form">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={health.status === 'error' ? 'Backend offline...' : 'Ask a question about your documents...'}
              disabled={health.status === 'error' || isTyping}
            />
            <button 
              type="submit" 
              disabled={!input.trim() || health.status === 'error' || isTyping}
              className={input.trim() ? 'active' : ''}
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
