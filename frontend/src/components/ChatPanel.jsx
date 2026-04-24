import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import ConfidenceBadge from './ConfidenceBadge';
import TracePanel from './TracePanel';
import { sendMessage, clearSession } from '../api';

const TypingIndicator = () => (
  <div className="flex items-center gap-1 px-4 py-3">
    <Bot size={16} className="text-indigo-400 mr-2" />
    <div className="flex gap-1">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className="w-2 h-2 bg-gray-500 rounded-full typing-dot"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
    <span className="text-xs text-gray-500 ml-2">
      Generating answer... this may take 1-2 minutes on CPU
    </span>
  </div>
);

const Message = ({ msg }) => {
  const [showTrace, setShowTrace] = useState(false);
  const isUser = msg.role === 'user';

  return (
    <div className={`message-appear flex gap-3 px-4 py-3 ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="shrink-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center">
          <Bot size={16} />
        </div>
      )}

      <div className={`max-w-[75%] ${isUser ? 'order-first' : ''}`}>
        <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? 'bg-indigo-600 text-white rounded-tr-sm ml-auto'
            : 'bg-gray-800 text-gray-100 rounded-tl-sm'
        }`}>
          {isUser ? (
            <p>{msg.content}</p>
          ) : (
            <ReactMarkdown>
              {msg.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Confidence badge for assistant messages */}
        {!isUser && msg.confidence !== undefined && (
          <ConfidenceBadge
            confidence={msg.confidence}
            retries={msg.retries}
          />
        )}

        {/* Citations */}
        {!isUser && msg.citations?.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {msg.citations.map((c, i) => (
              <span
                key={i}
                className="text-xs px-2 py-0.5 bg-gray-700 text-gray-400 rounded-full"
              >
                📄 {c}
              </span>
            ))}
          </div>
        )}

        {/* Pipeline trace toggle */}
        {!isUser && msg.trace && (
          <button
            onClick={() => setShowTrace(!showTrace)}
            className="mt-1 text-xs text-gray-500 hover:text-gray-400 underline"
          >
            {showTrace ? 'Hide' : 'Show'} pipeline trace
          </button>
        )}

        {showTrace && <TracePanel trace={msg.trace} />}
      </div>

      {isUser && (
        <div className="shrink-0 w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center">
          <User size={16} />
        </div>
      )}
    </div>
  );
};

const ChatPanel = ({ sessionId, onSessionClear }) => {
  const [messages,  setMessages]  = useState([]);
  const [input,     setInput]     = useState('');
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput('');
    setError(null);
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setLoading(true);

    try {
      const data = await sendMessage(q, sessionId);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        confidence: data.confidence,
        retries: data.retries,
        citations: data.citations,
        trace: data.trace,
      }]);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    await clearSession(sessionId);
    setMessages([]);
    if (onSessionClear) onSessionClear();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <div>
          <h1 className="font-semibold text-gray-100">SI-RAG</h1>
          <p className="text-xs text-gray-500">Self-Verifying Adaptive RAG</p>
        </div>
        <button
          onClick={handleClear}
          className="flex items-center gap-1 text-xs text-gray-500 hover:text-red-400 transition-colors"
        >
          <Trash2 size={14} />
          Clear chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center px-8">
            <div className="w-16 h-16 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center mb-4">
              <Bot size={32} className="text-indigo-400" />
            </div>
            <h2 className="text-lg font-semibold text-gray-200 mb-2">
              Ask anything from your documents
            </h2>
            <p className="text-sm text-gray-500 max-w-sm">
              Upload PDFs or text files on the left, then ask questions.
              Every answer is verified with a confidence score.
            </p>
            <div className="mt-6 grid grid-cols-1 gap-2 w-full max-w-sm">
              {[
                "What is retrieval augmented generation?",
                "How does BM25 work?",
                "What is hallucination in AI?",
              ].map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => setInput(suggestion)}
                  className="text-left text-xs text-gray-400 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-xl px-3 py-2 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}

        {loading && <TypingIndicator />}

        {error && (
          <div className="mx-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-xs text-red-400">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t border-gray-700">
        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            className="flex-1 bg-gray-800 border border-gray-600 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500 resize-none"
            style={{ maxHeight: '120px' }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="w-10 h-10 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:cursor-not-allowed flex items-center justify-center transition-colors shrink-0"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-xs text-gray-600 mt-2 text-center">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
};

export default ChatPanel;