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
    
    // Auto-remove message after 5 seconds for success/info, 8 seconds for warnings/errors
    setTimeout(() => {
      setMessages(prev => prev.filter(msg => msg.id !== newMessage.id));
    }, type === 'success' || type === 'info' ? 5000 : 8000);
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
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      <style>
        {`
          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateY(-10px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
          
          @keyframes fadeOut {
            from {
              opacity: 1;
            }
            to {
              opacity: 0;
            }
          }

          @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
          }

          .glass-morphism {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
          }

          .hover-lift {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
          }

          .hover-lift:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
          }

          .modern-button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            border-radius: 12px;
            color: white;
            cursor: pointer;
            font-weight: 600;
            padding: 12px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.3);
          }

          .modern-button:hover {
            background: linear-gradient(45deg, #764ba2, #667eea);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.4);
          }

          .modern-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
          }

          .modern-input {
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid transparent;
            border-radius: 12px;
            padding: 12px 16px;
            font-size: 14px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          }

          .modern-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
            background: rgba(255, 255, 255, 1);
          }

          .floating-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
          }
        `}
      </style>
      
      <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
        <div className="floating-card hover-lift" style={{ textAlign: 'center', marginBottom: '40px' }}>
          <h1 style={{ 
            background: 'linear-gradient(45deg, #667eea, #764ba2)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontSize: '3.5rem',
            fontWeight: '800',
            margin: '0',
            letterSpacing: '-2px'
          }}>
            � AI Document Assistant
          </h1>
          <p style={{ 
            fontSize: '1.2rem', 
            color: '#6c757d', 
            margin: '10px 0 0 0',
            fontWeight: '400'
          }}>
            Upload PDFs & Excel files, then chat with your documents using AI
          </p>
        </div>
      
        {/* Messages Section */}
        {messages.length > 0 && (
          <div style={{ marginBottom: '30px' }}>
            {messages.map((message) => (
              <div
                key={message.id}
                className="glass-morphism hover-lift"
                style={{
                  padding: '16px 20px',
                  marginBottom: '12px',
                  borderRadius: '16px',
                  border: '1px solid',
                  borderColor: 
                    message.type === 'success' ? 'rgba(40, 167, 69, 0.3)' :
                    message.type === 'error' ? 'rgba(220, 53, 69, 0.3)' :
                    message.type === 'warning' ? 'rgba(255, 193, 7, 0.3)' : 'rgba(23, 162, 184, 0.3)',
                  background:
                    message.type === 'success' ? 'rgba(40, 167, 69, 0.1)' :
                    message.type === 'error' ? 'rgba(220, 53, 69, 0.1)' :
                    message.type === 'warning' ? 'rgba(255, 193, 7, 0.1)' : 'rgba(23, 162, 184, 0.1)',
                  color:
                    message.type === 'success' ? '#155724' :
                    message.type === 'error' ? '#721c24' :
                    message.type === 'warning' ? '#856404' : '#0c5460',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  animation: 'slideIn 0.4s ease-out',
                  backdropFilter: 'blur(10px)'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ marginRight: '12px', fontSize: '20px' }}>
                    {message.type === 'success' ? '✅' :
                     message.type === 'error' ? '❌' :
                     message.type === 'warning' ? '⚠️' : 'ℹ️'}
                  </span>
                  <span style={{ fontWeight: '500', fontSize: '15px' }}>{message.text}</span>
                </div>
                <button
                  onClick={() => setMessages(prev => prev.filter(msg => msg.id !== message.id))}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'inherit',
                    cursor: 'pointer',
                    fontSize: '20px',
                    padding: '4px 8px',
                    borderRadius: '8px',
                    transition: 'background-color 0.2s ease'
                  }}
                  onMouseEnter={(e) => (e.target as HTMLElement).style.backgroundColor = 'rgba(0,0,0,0.1)'}
                  onMouseLeave={(e) => (e.target as HTMLElement).style.backgroundColor = 'transparent'}
                >
                  ×
                </button>
              </div>
            ))}
            {messages.length > 1 && (
              <button
                className="modern-button"
                onClick={clearMessages}
                style={{
                  fontSize: '12px',
                  padding: '8px 16px',
                  marginTop: '12px'
                }}
              >
                Clear All Messages
              </button>
            )}
          </div>
        )}      {/* File Upload Section */}
      <div className="floating-card hover-lift">
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
          <div style={{ 
            background: 'linear-gradient(45deg, #667eea, #764ba2)',
            borderRadius: '12px',
            padding: '12px',
            marginRight: '16px'
          }}>
            <span style={{ fontSize: '24px' }}>📤</span>
          </div>
          <h2 style={{ 
            color: '#2c3e50', 
            margin: 0,
            fontSize: '1.8rem',
            fontWeight: '700'
          }}>
            Upload Documents
          </h2>
        </div>
        
        <div style={{ 
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
          borderRadius: '16px',
          padding: '24px',
          border: '2px dashed #667eea',
          textAlign: 'center' as const,
          marginBottom: '20px',
          transition: 'all 0.3s ease'
        }}>
          <input
            id="fileInput"
            type="file"
            accept=".pdf,.xlsx,.xls"
            onChange={handleFileSelect}
            className="modern-input"
            style={{ 
              marginBottom: '16px',
              width: '100%',
              maxWidth: '400px'
            }}
          />
          <div style={{ marginBottom: '16px' }}>
            <button 
              className="modern-button"
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
              style={{
                fontSize: '16px',
                padding: '14px 28px',
                animation: isUploading ? 'pulse 2s infinite' : 'none'
              }}
            >
              {isUploading ? '⏳ Uploading...' : '🚀 Upload File'}
            </button>
          </div>
          
          <p style={{ 
            color: '#6c757d', 
            fontSize: '14px', 
            margin: 0,
            fontWeight: '500'
          }}>
            📁 Supported formats: PDF, Excel (.xlsx, .xls)
          </p>
        </div>
      </div>

      {/* Uploaded Files Section */}
      {uploadedFiles.length > 0 && (
        <div className="floating-card hover-lift">
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
            <div style={{ 
              background: 'linear-gradient(45deg, #28a745, #20c997)',
              borderRadius: '12px',
              padding: '12px',
              marginRight: '16px'
            }}>
              <span style={{ fontSize: '24px' }}>📋</span>
            </div>
            <h2 style={{ 
              color: '#2c3e50', 
              margin: 0,
              fontSize: '1.8rem',
              fontWeight: '700'
            }}>
              Uploaded Files ({uploadedFiles.length})
            </h2>
          </div>
          
          <div style={{ display: 'grid', gap: '16px' }}>
            {uploadedFiles.map((file, index) => (
              <div key={index} className="glass-morphism hover-lift" style={{ 
                padding: '20px',
                borderRadius: '16px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                background: 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.3)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <div style={{
                    padding: '8px 12px',
                    background: file.type === 'PDF' 
                      ? 'linear-gradient(45deg, #dc3545, #e74c3c)' 
                      : 'linear-gradient(45deg, #28a745, #2ecc71)',
                    color: 'white',
                    borderRadius: '12px',
                    fontSize: '12px',
                    fontWeight: '600',
                    marginRight: '16px'
                  }}>
                    {file.type === 'PDF' ? '📄 PDF' : '📊 Excel'}
                  </div>
                  <div>
                    <strong style={{ fontSize: '16px', color: '#2c3e50' }}>{file.name}</strong>
                    <div style={{ 
                      color: '#6c757d', 
                      fontSize: '13px',
                      marginTop: '4px'
                    }}>
                      {(file.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>
                <button
                  className="modern-button"
                  onClick={() => deleteFile(file.name)}
                  style={{
                    background: 'linear-gradient(45deg, #dc3545, #e74c3c)',
                    padding: '8px 16px',
                    fontSize: '13px'
                  }}
                >
                  🗑️ Delete
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Query Section */}
      <div className="floating-card hover-lift">
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
          <div style={{ 
            background: 'linear-gradient(45deg, #ffc107, #ff9800)',
            borderRadius: '12px',
            padding: '12px',
            marginRight: '16px'
          }}>
            <span style={{ fontSize: '24px' }}>🔍</span>
          </div>
          <h2 style={{ 
            color: '#2c3e50', 
            margin: 0,
            fontSize: '1.8rem',
            fontWeight: '700'
          }}>
            Query Your Documents
          </h2>
        </div>
        
        <form onSubmit={handleQuery}>
          <div style={{ marginBottom: '20px' }}>
            <textarea
              rows={4}
              placeholder="💬 Ask questions about your uploaded documents..."
              value={query}
              onChange={handleQueryChange}
              className="modern-input"
              style={{ 
                width: '100%', 
                minHeight: '120px',
                resize: 'vertical',
                fontSize: '16px',
                lineHeight: '1.5'
              }}
              disabled={uploadedFiles.length === 0}
            />
          </div>
          
          <div style={{ textAlign: 'center' as const }}>
            <button 
              type="submit"
              className="modern-button"
              disabled={uploadedFiles.length === 0 || isQuerying || !query.trim()}
              style={{
                fontSize: '18px',
                padding: '16px 32px',
                animation: isQuerying ? 'pulse 2s infinite' : 'none',
                background: (uploadedFiles.length === 0 || isQuerying || !query.trim()) 
                  ? '#6c757d' : 'linear-gradient(45deg, #667eea, #764ba2)'
              }}
            >
              {isQuerying ? '🔍 Searching...' : '🚀 Search Documents'}
            </button>
          </div>
        </form>
        
        {uploadedFiles.length === 0 && (
          <div style={{ 
            textAlign: 'center' as const,
            marginTop: '20px',
            padding: '20px',
            background: 'rgba(255, 193, 7, 0.1)',
            borderRadius: '12px',
            border: '1px solid rgba(255, 193, 7, 0.3)'
          }}>
            <p style={{ 
              color: '#856404', 
              fontStyle: 'italic',
              margin: 0,
              fontSize: '16px',
              fontWeight: '500'
            }}>
              📁 Please upload documents first before querying.
            </p>
          </div>
        )}
      </div>

      {/* Response Section */}
      {response && (
        <div className="floating-card hover-lift">
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
            <div style={{ 
              background: 'linear-gradient(45deg, #17a2b8, #138496)',
              borderRadius: '12px',
              padding: '12px',
              marginRight: '16px'
            }}>
              <span style={{ fontSize: '24px' }}>📝</span>
            </div>
            <h2 style={{ 
              color: '#2c3e50', 
              margin: 0,
              fontSize: '1.8rem',
              fontWeight: '700'
            }}>
              AI Response
            </h2>
          </div>
          
          <div style={{
            background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
            borderRadius: '16px',
            padding: '24px',
            border: '1px solid rgba(0,0,0,0.1)',
            boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <pre style={{ 
              background: 'transparent',
              border: 'none',
              whiteSpace: 'pre-wrap',
              fontSize: '15px',
              lineHeight: '1.6',
              margin: 0,
              color: '#2c3e50',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
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