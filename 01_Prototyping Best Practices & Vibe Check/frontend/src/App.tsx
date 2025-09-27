import React, { useState, ChangeEvent, FormEvent } from 'react';

interface ChatResponse {
  response?: string;
  detail?: string;
}

const App: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('');
  const [response, setResponse] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [apiKeyFocused, setApiKeyFocused] = useState<boolean>(false);
  const [promptFocused, setPromptFocused] = useState<boolean>(false);
  const [buttonHovered, setButtonHovered] = useState<boolean>(false);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    setResponse("Loading...");

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ prompt, api_key: apiKey }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data: ChatResponse = await res.json();
      setResponse(data.response ?? data.detail ?? 'No response');
    } catch (error: unknown) {
      if (error instanceof Error) {
        setResponse(`Error: ${error.message}`);
      } else {
        setResponse('Unknown error occurred.');
      }
    }
  };

  const handlePromptChange = (e: ChangeEvent<HTMLTextAreaElement>) => setPrompt(e.target.value);
  const handleApiKeyChange = (e: ChangeEvent<HTMLInputElement>) => setApiKey(e.target.value);

  return (
    <div style={{ 
      minHeight: '100vh',
      backgroundColor: '#f5f7fa',
      padding: '2rem 1rem',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1)',
        padding: '2.5rem',
        overflow: 'hidden'
      }}>
        <h1 style={{
          color: '#1f2937',
          fontSize: '2rem',
          fontWeight: '700',
          marginBottom: '2rem',
          textAlign: 'center',
          borderBottom: '2px solid #e5e7eb',
          paddingBottom: '1rem'
        }}>Chat with OpenAI</h1>
        
        <form onSubmit={handleSubmit} style={{ marginBottom: '2rem' }}>
          <input
            type="text"
            placeholder="Enter your OpenAI API key"
            value={apiKey}
            onChange={handleApiKeyChange}
            style={{ 
              width: '100%',
              padding: '0.75rem',
              marginBottom: '1rem',
              border: `2px solid ${apiKeyFocused ? '#3b82f6' : '#e5e7eb'}`,
              borderRadius: '8px',
              fontSize: '1rem',
              outline: 'none',
              transition: 'border-color 0.2s ease',
              boxSizing: 'border-box'
            }}
            onFocus={() => setApiKeyFocused(true)}
            onBlur={() => setApiKeyFocused(false)}
          />
          <textarea
            rows={4}
            placeholder="Type your message..."
            value={prompt}
            onChange={handlePromptChange}
            style={{ 
              width: '100%',
              padding: '0.75rem',
              marginBottom: '1.5rem',
              border: `2px solid ${promptFocused ? '#3b82f6' : '#e5e7eb'}`,
              borderRadius: '8px',
              fontSize: '1rem',
              resize: 'vertical',
              outline: 'none',
              transition: 'border-color 0.2s ease',
              fontFamily: 'inherit',
              boxSizing: 'border-box'
            }}
            onFocus={() => setPromptFocused(true)}
            onBlur={() => setPromptFocused(false)}
          />
          <button 
            type="submit"
            style={{
              backgroundColor: buttonHovered ? '#2563eb' : '#3b82f6',
              color: 'white',
              padding: '0.75rem 2rem',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'background-color 0.2s ease',
              outline: 'none'
            }}
            onMouseEnter={() => setButtonHovered(true)}
            onMouseLeave={() => setButtonHovered(false)}
          >
            Send Message
          </button>
        </form>
        
        <div style={{
          backgroundColor: '#f9fafb',
          border: '1px solid #e5e7eb',
          borderRadius: '8px',
          padding: '1.5rem',
          minHeight: '120px'
        }}>
          <h3 style={{
            color: '#374151',
            fontSize: '1.25rem',
            fontWeight: '600',
            marginBottom: '1rem',
            marginTop: '0'
          }}>Response:</h3>
          <pre style={{
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            color: '#1f2937',
            fontSize: '0.95rem',
            lineHeight: '1.5',
            margin: '0',
            fontFamily: 'inherit',
            maxHeight: '400px',
            overflowY: 'auto'
          }}>{response}</pre>
        </div>
      </div>
    </div>
  );
};

export default App;