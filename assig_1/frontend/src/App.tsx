import React, { useState, ChangeEvent, FormEvent } from 'react';

interface ChatResponse {
  response?: string;
  detail?: string;
}

const App: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('');
  const [response, setResponse] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');

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
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>Chat with OpenAI</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter your OpenAI API key"
          value={apiKey}
          onChange={handleApiKeyChange}
          style={{ width: '100%', marginBottom: '1rem' }}
        />
        <textarea
          rows={4}
          placeholder="Type your message..."
          value={prompt}
          onChange={handlePromptChange}
          style={{ width: '100%', marginBottom: '1rem' }}
        />
        <button type="submit">Send</button>
      </form>
      <h3>Response:</h3>
      <pre>{response}</pre>
    </div>
  );
};

export default App;