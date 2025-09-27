import React, { useState, ChangeEvent, FormEvent } from 'react';

interface QueryResponse {
  response?: string;
  sources?: Array<{
    text: string;
    score: number;
    metadata: any;
    file_name?: string;
    file_type?: string;
  }>;
  file_type?: string;
  query?: string;
  detail?: string;
}

interface UploadedFile {
  name: string;
  type: string;
  size: number;
  status: string;
  result?: any;
}

interface Message {
  id: number;
  text: string;
  type: 'success' | 'error' | 'warning' | 'info';
  timestamp: number;
}

const App: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [response, setResponse] = useState<string>('');
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([]);

  // Add message function
  const addMessage = (text: string, type: 'success' | 'error' | 'warning' | 'info') => {
    const newMessage: Message = {
      id: Date.now(),
      text,
      type,
      timestamp: Date.now()
    };
    setMessages(prev => [newMessage, ...prev.slice(0, 4)]); // Keep only last 5 messages
    
    // Auto-remove message after 4 seconds
    setTimeout(() => {
      setMessages(prev => prev.filter(msg => msg.id !== newMessage.id));
    }, 4000);
  };

  // Clear messages function
  const clearMessages = () => setMessages([]);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      const fileExtension = file.name.toLowerCase().split('.').pop();
      
      if (fileExtension && ['pdf', 'xlsx', 'xls'].includes(fileExtension)) {
        setSelectedFile(file);
      } else {
        addMessage('Please select a PDF or Excel file', 'warning');
        e.target.value = '';
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      addMessage('Please select a file first', 'warning');
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      
      // Refresh file list
      await fetchUploadedFiles();
      
      setSelectedFile(null);
      (document.getElementById('fileInput') as HTMLInputElement).value = '';
      
      addMessage(`File uploaded successfully! ${data.type} processed with ${data.result.chunks || data.result.sheets} chunks/sheets.`, 'success');
      
    } catch (error: unknown) {
      if (error instanceof Error) {
        addMessage(`Upload error: ${error.message}`, 'error');
      } else {
        addMessage('Upload failed', 'error');
      }
    } finally {
      setIsUploading(false);
    }
  };

  const fetchUploadedFiles = async () => {
    try {
      const res = await fetch('/api/files');
      if (res.ok) {
        const data = await res.json();
        setUploadedFiles(data.files || []);
      }
    } catch (error) {
      console.error('Error fetching files:', error);
    }
  };

  const handleQuery = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    
    if (!query.trim()) {
      addMessage('Please enter a query', 'warning');
      return;
    }

    if (uploadedFiles.length === 0) {
      addMessage('Please upload at least one file first', 'warning');
      return;
    }

    setIsQuerying(true);
    setResponse("Searching through your documents...");

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data: QueryResponse = await res.json();
      
      let responseText = data.response || 'No response';
      
      if (data.sources && data.sources.length > 0) {
        responseText += '\n\n--- Source Details ---\n';
        data.sources.forEach((source, index) => {
          responseText += `\nSource ${index + 1} (Score: ${source.score.toFixed(3)}):\n`;
          responseText += `File: ${source.file_name || 'Unknown'} (${source.file_type || 'Unknown'})\n`;
          responseText += `Content: ${source.text}\n`;
        });
      }
      
      setResponse(responseText);
      
    } catch (error: unknown) {
      if (error instanceof Error) {
        setResponse(`Error: ${error.message}`);
      } else {
        setResponse('Unknown error occurred.');
      }
    } finally {
      setIsQuerying(false);
    }
  };

  const deleteFile = async (filename: string) => {
    try {
      const res = await fetch(`/api/files/${filename}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        await fetchUploadedFiles();
        setResponse('');
        addMessage(`File '${filename}' deleted successfully`, 'success');
      } else {
        throw new Error('Delete failed');
      }
    } catch (error) {
      addMessage(`Error deleting file: ${error}`, 'error');
    }
  };

  // Load files on component mount
  React.useEffect(() => {
    fetchUploadedFiles();
  }, []);

  const handleQueryChange = (e: ChangeEvent<HTMLTextAreaElement>) => setQuery(e.target.value);

  return (
    <div style={{ 
      minHeight: '100vh',
      background: '#f8fafc',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      padding: '20px'
    }}>
      <style>
        {`
          @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          
          .card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e5e7eb;
          }
          
          .btn {
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
          }
          
          .btn:hover {
            background: #2563eb;
            transform: translateY(-1px);
          }
          
          .btn:disabled {
            background: #9ca3af;
            cursor: not-allowed;
            transform: none;
          }
          
          .input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.2s ease;
            box-sizing: border-box;
          }
          
          .input:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
          }
          
          .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-bottom: 8px;
            background: #f9fafb;
          }
          
          .message {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 8px;
            animation: slideIn 0.3s ease;
          }
          
          .message.success { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
          .message.error { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
          .message.warning { background: #fefce8; color: #a16207; border: 1px solid #fef3c7; }
          .message.info { background: #eff6ff; color: #1e40af; border: 1px solid #dbeafe; }
        `}
      </style>
      
      <div style={{ maxWidth: '800px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <h1 style={{ 
            fontSize: '2.5rem',
            fontWeight: '700',
            color: '#1f2937',
            marginBottom: '8px'
          }}>
            Document Chat
          </h1>
          <p style={{ color: '#6b7280', fontSize: '1.1rem' }}>
            Upload PDFs & Excel files, then ask questions
          </p>
        </div>

        {/* Messages */}
        {messages.length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.type}`}
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
              >
                <span>{message.text}</span>
                <button
                  onClick={() => setMessages(prev => prev.filter(msg => msg.id !== message.id))}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'inherit',
                    cursor: 'pointer',
                    fontSize: '18px',
                    padding: '0 4px'
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Upload Section */}
        <div className="card">
          <h2 style={{ margin: '0 0 20px 0', color: '#1f2937' }}>Upload Files</h2>
          <div style={{ marginBottom: '16px' }}>
            <input
              id="fileInput"
              type="file"
              accept=".pdf,.xlsx,.xls"
              onChange={handleFileSelect}
              className="input"
              style={{ marginBottom: '16px' }}
            />
            <button 
              className="btn"
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
            >
              {isUploading ? 'Uploading...' : 'Upload File'}
            </button>
          </div>
          <p style={{ color: '#6b7280', fontSize: '14px', margin: 0 }}>
            Supported: PDF, Excel (.xlsx, .xls)
          </p>
        </div>

        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <div className="card">
            <h2 style={{ margin: '0 0 20px 0', color: '#1f2937' }}>
              Files ({uploadedFiles.length})
            </h2>
            {uploadedFiles.map((file, index) => (
              <div key={index} className="file-item">
                <div>
                  <strong>{file.name}</strong>
                  <span style={{ 
                    marginLeft: '12px',
                    padding: '4px 8px',
                    background: file.type === 'PDF' ? '#ef4444' : '#10b981',
                    color: 'white',
                    borderRadius: '4px',
                    fontSize: '12px'
                  }}>
                    {file.type}
                  </span>
                  <span style={{ marginLeft: '12px', color: '#6b7280', fontSize: '14px' }}>
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                </div>
                <button
                  onClick={() => deleteFile(file.name)}
                  style={{
                    background: '#ef4444',
                    color: 'white',
                    border: 'none',
                    padding: '6px 12px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Query Section */}
        <div className="card">
          <h2 style={{ margin: '0 0 20px 0', color: '#1f2937' }}>Ask Questions</h2>
          <form onSubmit={handleQuery}>
            <textarea
              rows={4}
              placeholder="Ask questions about your documents..."
              value={query}
              onChange={handleQueryChange}
              className="input"
              style={{ 
                marginBottom: '16px',
                resize: 'vertical',
                minHeight: '100px'
              }}
              disabled={uploadedFiles.length === 0}
            />
            <button 
              type="submit"
              className="btn"
              disabled={uploadedFiles.length === 0 || isQuerying || !query.trim()}
            >
              {isQuerying ? 'Searching...' : 'Search'}
            </button>
            
            {uploadedFiles.length === 0 && (
              <p style={{ 
                color: '#f59e0b',
                marginTop: '12px',
                padding: '12px',
                background: '#fef3c7',
                borderRadius: '6px',
                fontSize: '14px'
              }}>
                Please upload documents first
              </p>
            )}
          </form>
        </div>

        {/* Response Section */}
        {response && (
          <div className="card">
            <h2 style={{ margin: '0 0 20px 0', color: '#1f2937' }}>Response</h2>
            <div style={{ 
              background: '#f8fafc',
              padding: '20px',
              borderRadius: '8px',
              border: '1px solid #e5e7eb'
            }}>
              <pre style={{ 
                whiteSpace: 'pre-wrap',
                fontSize: '14px',
                lineHeight: '1.6',
                margin: 0,
                fontFamily: 'inherit'
              }}>
                {response}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;