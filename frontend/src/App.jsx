import React, { useState, useEffect } from 'react';
import ChatPanel from './components/ChatPanel';
import UploadPanel from './components/UploadPanel';
import { checkHealth } from './api';
import { Activity, AlertTriangle } from 'lucide-react';

const SESSION_ID = 'session_' + Math.random().toString(36).slice(2, 9);

function App() {
  const [apiOnline,   setApiOnline]   = useState(null);
  const [, setUploadCount] = useState(0);

  useEffect(() => {
    const check = async () => {
      const ok = await checkHealth();
      setApiOnline(ok);
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 overflow-hidden">

      {/* Left sidebar — upload panel */}
      <div className="w-72 shrink-0 bg-gray-850 border-r border-gray-700 flex flex-col"
           style={{ background: '#111' }}>

        {/* Logo */}
        <div className="px-4 py-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-sm font-bold">
              SI
            </div>
            <div>
              <p className="text-sm font-semibold">SI-RAG</p>
              <p className="text-xs text-gray-500">Adaptive RAG System</p>
            </div>
          </div>
        </div>

        {/* API status */}
        <div className="px-4 py-2 border-b border-gray-700">
          <div className="flex items-center gap-2 text-xs">
            {apiOnline === null ? (
              <span className="text-gray-500">Checking API...</span>
            ) : apiOnline ? (
              <>
                <Activity size={12} className="text-green-400" />
                <span className="text-green-400">API Online</span>
              </>
            ) : (
              <>
                <AlertTriangle size={12} className="text-red-400" />
                <span className="text-red-400">API Offline</span>
              </>
            )}
          </div>
        </div>

        {/* Upload panel */}
        <div className="flex-1 overflow-y-auto">
          <UploadPanel
            onUploadSuccess={() => setUploadCount(c => c + 1)}
          />
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-700">
          <p className="text-xs text-gray-600 text-center">
            Self-Verifying Adaptive RAG
          </p>
          <p className="text-xs text-gray-700 text-center">
            NAU Research · 2025
          </p>
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {apiOnline === false ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <AlertTriangle size={48} className="text-red-400 mx-auto mb-4" />
              <h2 className="text-lg font-semibold text-gray-200 mb-2">
                API Server Offline
              </h2>
              <p className="text-sm text-gray-500 mb-4">
                Start the FastAPI server to use SI-RAG
              </p>
              <code className="text-xs bg-gray-800 px-3 py-2 rounded-lg text-green-400">
                uvicorn api.main:app --reload --port 8000
              </code>
            </div>
          </div>
        ) : (
          <ChatPanel
            sessionId={SESSION_ID}
            onSessionClear={() => {}}
          />
        )}
      </div>
    </div>
  );
}

export default App;