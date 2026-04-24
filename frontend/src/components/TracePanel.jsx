import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Search, FileText, Brain, RefreshCw } from 'lucide-react';

const Section = ({ title, icon, children, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden mb-2">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-750 text-left text-sm font-medium text-gray-300"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {icon}
        {title}
      </button>
      {open && (
        <div className="p-3 bg-gray-900 text-xs text-gray-400 space-y-1">
          {children}
        </div>
      )}
    </div>
  );
};

const TracePanel = ({ trace }) => {
  if (!trace) return null;

  return (
    <div className="mt-3 space-y-1">
      {/* Query optimization */}
      <Section
        title="Query Optimization"
        icon={<Brain size={13} />}
        defaultOpen={true}
      >
        <div>
          <span className="text-gray-500">Original: </span>
          <span>{trace.original_query}</span>
        </div>
        {trace.optimized_query !== trace.original_query && (
          <div>
            <span className="text-green-500">Optimized: </span>
            <span>{trace.optimized_query}</span>
          </div>
        )}
        {trace.strategy_used && (
          <div>
            <span className="text-yellow-500">Strategy: </span>
            <span>{trace.strategy_used}</span>
          </div>
        )}
      </Section>

      {/* Retrieved docs */}
      <Section
        title={`Retrieved Documents (${trace.retrieved_docs?.length || 0})`}
        icon={<Search size={13} />}
      >
        {trace.retrieved_docs?.map((doc, i) => (
          <div key={i} className="border border-gray-700 rounded p-2 mb-1">
            <div className="flex justify-between mb-1">
              <span className="text-indigo-400 font-medium">#{i + 1} {doc.source}</span>
              <span className="text-green-400">score: {doc.score}</span>
            </div>
            <p className="text-gray-400 leading-relaxed">{doc.text}...</p>
          </div>
        ))}
      </Section>

      {/* Claim verification */}
      {trace.claims?.length > 0 && (
        <Section
          title={`Claim Verification (${trace.claims.length} claims)`}
          icon={<FileText size={13} />}
        >
          {trace.claims.map((claim, i) => (
            <div key={i} className="flex gap-2 mb-1 items-start">
              <span className={`shrink-0 font-bold text-xs px-1 rounded ${
                claim.label === 'ENTAILED'
                  ? 'bg-green-500/20 text-green-400'
                  : claim.label === 'CONTRADICTED'
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-gray-500/20 text-gray-400'
              }`}>
                {claim.label === 'ENTAILED' ? '✓' : claim.label === 'CONTRADICTED' ? '✗' : '?'}
              </span>
              <span className="text-gray-400">{claim.claim}</span>
              <span className="shrink-0 text-gray-600 ml-auto">{(claim.score * 100).toFixed(0)}%</span>
            </div>
          ))}
        </Section>
      )}

      {/* Retries */}
      {trace.retries > 0 && (
        <Section
          title={`Retries (${trace.retries})`}
          icon={<RefreshCw size={13} />}
        >
          <p className="text-yellow-400">
            System retried {trace.retries} time{trace.retries > 1 ? 's' : ''} to improve answer quality.
          </p>
        </Section>
      )}
    </div>
  );
};

export default TracePanel;