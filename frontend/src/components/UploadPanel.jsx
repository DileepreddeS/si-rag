import React, { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { uploadFile } from '../api';

const UploadPanel = ({ onUploadSuccess }) => {
  const [dragging,  setDragging]  = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress,  setProgress]  = useState(0);
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState(null);
  const [totalDocs, setTotalDocs] = useState(null);
  const fileRef = useRef();

  const handleFile = async (file) => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setResult(null);
    setProgress(0);

    try {
      const data = await uploadFile(file, setProgress);
      setResult(data);
      setTotalDocs(data.total_docs);
      if (onUploadSuccess) onUploadSuccess(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  return (
    <div className="p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        <FileText size={16} />
        Document Library
      </h2>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
          dragging
            ? 'border-indigo-500 bg-indigo-500/10'
            : 'border-gray-600 hover:border-gray-500 hover:bg-gray-800/50'
        }`}
      >
        <Upload size={24} className="mx-auto mb-2 text-gray-500" />
        <p className="text-xs text-gray-400">
          Drop PDF, TXT, or MD files here
        </p>
        <p className="text-xs text-gray-600 mt-1">or click to browse</p>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.md"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {/* Upload progress */}
      {uploading && (
        <div className="mt-3">
          <div className="flex items-center gap-2 text-xs text-gray-400 mb-1">
            <Loader size={12} className="animate-spin" />
            Indexing document... {progress}%
          </div>
          <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Success */}
      {result && !uploading && (
        <div className="mt-3 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="flex items-center gap-2 text-green-400 text-xs font-medium mb-1">
            <CheckCircle size={14} />
            Indexed successfully
          </div>
          <p className="text-xs text-gray-400">{result.filename}</p>
          <p className="text-xs text-gray-500">
            {result.chunks_added} chunks added · {result.total_docs} total
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
          <div className="flex items-center gap-2 text-red-400 text-xs font-medium">
            <AlertCircle size={14} />
            Upload failed
          </div>
          <p className="text-xs text-gray-500 mt-1">{error}</p>
        </div>
      )}

      {/* Total docs */}
      {totalDocs !== null && (
        <p className="text-xs text-gray-600 mt-3 text-center">
          {totalDocs} chunks in knowledge base
        </p>
      )}
    </div>
  );
};

export default UploadPanel;