import React, { useState, ChangeEvent, FormEvent } from 'react';
import './App.scss';

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
    <div className="app-container">
      <div className="app-content">
        {/* Header */}
        <div className="app-header">
          <h1 className="app-title">
            Document Chat
          </h1>
          <p className="app-subtitle">
            Upload PDFs & Excel files, then ask questions
          </p>
        </div>

        {/* Messages */}
        {messages.length > 0 && (
          <div className="messages-container">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.type}`}
              >
                <span>{message.text}</span>
                <button
                  onClick={() => setMessages(prev => prev.filter(msg => msg.id !== message.id))}
                  className="message-close-btn"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Upload Section */}
        <div className="card upload-section">
          <h2 className="card-title">Upload Files</h2>
          <div>
            <input
              id="fileInput"
              type="file"
              accept=".pdf,.xlsx,.xls"
              onChange={handleFileSelect}
              className="input input-file"
            />
            <button 
              className="btn"
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
            >
              {isUploading ? 'Uploading...' : 'Upload File'}
            </button>
          </div>
          <p className="upload-hint">
            Supported: PDF, Excel (.xlsx, .xls)
          </p>
        </div>

        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <div className="card files-section">
            <h2 className="files-title">
              Files ({uploadedFiles.length})
            </h2>
            {uploadedFiles.map((file, index) => (
              <div key={index} className="file-item">
                <div className="file-info">
                  <span className="file-name">{file.name}</span>
                  <span className={`file-type-badge ${file.type === 'PDF' ? 'pdf-badge' : 'excel-badge'}`}>
                    {file.type}
                  </span>
                  <span className="file-size">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                </div>
                <button
                  onClick={() => deleteFile(file.name)}
                  className="btn-danger"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Query Section */}
        <div className="card query-section">
          <h2 className="query-title">Ask Questions</h2>
          <form onSubmit={handleQuery}>
            <textarea
              rows={4}
              placeholder="Ask questions about your documents..."
              value={query}
              onChange={handleQueryChange}
              className="input input-textarea"
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
              <p className="no-files-warning">
                Please upload documents first
              </p>
            )}
          </form>
        </div>

        {/* Response Section */}
        {response && (
          <div className="card response-section">
            <h2 className="response-title">Response</h2>
            <div className="response-container">
              <pre className="response-text">
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