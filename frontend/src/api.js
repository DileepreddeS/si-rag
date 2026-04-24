import axios from 'axios';

const BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 600000, // 5 minutes for slow CPU inference
});

export const sendMessage = async (question, sessionId) => {
  const response = await api.post('/chat', {
    question,
    session_id: sessionId,
    stream: false,
  });
  return response.data;
};

export const uploadFile = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress) {
        onProgress(Math.round((e.loaded * 100) / e.total));
      }
    },
  });
  return response.data;
};

export const getDocuments = async () => {
  const response = await api.get('/documents');
  return response.data;
};

export const clearSession = async (sessionId) => {
  await api.delete(`/session/${sessionId}`);
};

export const checkHealth = async () => {
  try {
    const response = await axios.get('http://localhost:8000/health');
    return response.data.status === 'healthy';
  } catch {
    return false;
  }
};