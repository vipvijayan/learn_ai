import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { ReactComponent as SchoolIcon } from './assets/SchoolIcon.svg';
import EventPopup from './components/EventPopup';

const API_BASE_URL = 'http://localhost:8000';

// Function to parse markdown bold syntax
const parseMarkdown = (text) => {
  const parts = [];
  let lastIndex = 0;
  const boldRegex = /\*\*(.+?)\*\*/g;
  let match;
  
  while ((match = boldRegex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: text.substring(lastIndex, match.index) });
    }
    // Add the bold text
    parts.push({ type: 'bold', content: match[1] });
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push({ type: 'text', content: text.substring(lastIndex) });
  }
  
  return parts.length > 0 ? parts : [{ type: 'text', content: text }];
};

// Function to render text with markdown formatting
const renderMarkdownText = (text, key) => {
  const parts = parseMarkdown(text);
  return (
    <span key={key}>
      {parts.map((part, i) => 
        part.type === 'bold' ? (
          <strong key={i}>{part.content}</strong>
        ) : (
          <span key={i}>{part.content}</span>
        )
      )}
    </span>
  );
};

// Function to format time in seconds to readable format
const formatResponseTime = (seconds) => {
  if (seconds < 60) {
    return `${seconds}s`;
  }
  
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours} hour${hours > 1 ? 's' : ''}: ${minutes} min: ${secs} sec${secs !== 1 ? 's' : ''}`;
  } else {
    return `${minutes} min: ${secs} sec${secs !== 1 ? 's' : ''}`;
  }
};

// Function to format LLM response text into HTML
const formatResponseText = (text) => {
  if (!text) return '';
  
  // Split by single newlines first to handle headers properly
  const lines = text.split('\n').map(l => l.trim()).filter(l => l);
  const elements = [];
  let currentParagraph = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Check for markdown headers
    if (line.match(/^###\s+/)) {
      // Flush current paragraph
      if (currentParagraph.length > 0) {
        elements.push({ type: 'paragraph', content: currentParagraph.join('\n') });
        currentParagraph = [];
      }
      elements.push({ type: 'h3', content: line.replace(/^###\s+/, '') });
    } else if (line.match(/^##\s+/)) {
      if (currentParagraph.length > 0) {
        elements.push({ type: 'paragraph', content: currentParagraph.join('\n') });
        currentParagraph = [];
      }
      elements.push({ type: 'h2', content: line.replace(/^##\s+/, '') });
    } else if (line.match(/^#\s+/)) {
      if (currentParagraph.length > 0) {
        elements.push({ type: 'paragraph', content: currentParagraph.join('\n') });
        currentParagraph = [];
      }
      elements.push({ type: 'h1', content: line.replace(/^#\s+/, '') });
    } else if (line.match(/^\d+\.\s+/)) {
      // Numbered list (e.g., "1. ", "2. ")
      if (currentParagraph.length > 0) {
        elements.push({ type: 'paragraph', content: currentParagraph.join('\n') });
        currentParagraph = [];
      }
      // Collect consecutive numbered items
      const numberedItems = [line.replace(/^\d+\.\s*/, '')];
      while (i + 1 < lines.length && lines[i + 1].match(/^\d+\.\s+/)) {
        i++;
        numberedItems.push(lines[i].replace(/^\d+\.\s*/, ''));
      }
      elements.push({ type: 'numbered-list', items: numberedItems });
    } else if (line.startsWith('•') || line.startsWith('-')) {
      // Bullet point
      if (currentParagraph.length > 0) {
        elements.push({ type: 'paragraph', content: currentParagraph.join('\n') });
        currentParagraph = [];
      }
      // Collect consecutive bullet points
      const bulletItems = [line.replace(/^[•-]\s*/, '')];
      while (i + 1 < lines.length && (lines[i + 1].startsWith('•') || lines[i + 1].startsWith('-'))) {
        i++;
        bulletItems.push(lines[i].replace(/^[•-]\s*/, ''));
      }
      elements.push({ type: 'list', items: bulletItems });
    } else {
      // Regular line - add to current paragraph
      currentParagraph.push(line);
    }
  }
  
  // Flush remaining paragraph
  if (currentParagraph.length > 0) {
    elements.push({ type: 'paragraph', content: currentParagraph.join('\n') });
  }
  
  // Render elements
  return elements.map((element, idx) => {
    switch (element.type) {
      case 'h1':
        return <h1 key={idx} className="response-header" style={{ fontSize: '1.6em', fontWeight: 'bold', marginTop: '1em', marginBottom: '0.5em' }}>
          {renderMarkdownText(element.content, idx)}
        </h1>;
      case 'h2':
        return <h2 key={idx} className="response-header" style={{ fontSize: '1.4em', fontWeight: 'bold', marginTop: '1em', marginBottom: '0.5em' }}>
          {renderMarkdownText(element.content, idx)}
        </h2>;
      case 'h3':
        return <h3 key={idx} className="response-header" style={{ fontSize: '1.2em', fontWeight: 'bold', marginTop: '1em', marginBottom: '0.5em' }}>
          {renderMarkdownText(element.content, idx)}
        </h3>;
      case 'numbered-list':
        return <ol key={idx} className="formatted-list" style={{ marginLeft: '1.5em', marginTop: '0.5em', marginBottom: '0.5em' }}>
          {element.items.map((item, i) => (
            <li key={i}>{renderMarkdownText(item, i)}</li>
          ))}
        </ol>;
      case 'list':
        return <ul key={idx} className="formatted-list" style={{ marginLeft: '1.5em', marginTop: '0.5em', marginBottom: '0.5em' }}>
          {element.items.map((item, i) => (
            <li key={i}>{renderMarkdownText(item, i)}</li>
          ))}
        </ul>;
      case 'paragraph':
        // Check if it's a header-like line (ends with :)
        if (element.content.endsWith(':') && element.content.length < 50 && !element.content.includes('\n')) {
          return <h4 key={idx} className="response-header" style={{ fontWeight: 'bold', marginTop: '0.8em', marginBottom: '0.3em' }}>
            {renderMarkdownText(element.content, idx)}
          </h4>;
        }
        return <p key={idx} className="response-text" style={{ marginBottom: '0.5em' }}>
          {renderMarkdownText(element.content, idx)}
        </p>;
      default:
        return null;
    }
  });
};

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('chat'); // 'events', 'chat', 'evaluation', 'comparison', or 'settings'
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [comparisonResults, setComparisonResults] = useState({
    original: null,
    naive: null
  });
  // eslint-disable-next-line no-unused-vars
  const [isComparing, setIsComparing] = useState(false);
  const [isRunningComparison, setIsRunningComparison] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [currentMethod, setCurrentMethod] = useState('naive');
  const messagesEndRef = useRef(null);
  const [events, setEvents] = useState([]);
  const [isLoadingEvents, setIsLoadingEvents] = useState(true);
  // eslint-disable-next-line no-unused-vars
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [selectedEventDetails, setSelectedEventDetails] = useState(null);
  const [showSplash, setShowSplash] = useState(true);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'online', 'offline'
  // eslint-disable-next-line no-unused-vars
  const [backendError, setBackendError] = useState('');
  const [useWebSocket, setUseWebSocket] = useState(true); // Toggle for WebSocket vs HTTP
  const wsRef = useRef(null); // WebSocket connection reference
  const [streamingMessage, setStreamingMessage] = useState(null); // Current streaming message
  
  // Check backend health on mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`, {
          timeout: 5000
        });
        
        if (response.status === 200) {
          setBackendStatus('online');
          
          // Hide splash screen after 4 seconds if backend is online
          setTimeout(() => {
            setShowSplash(false);
          }, 4000);
        }
      } catch (err) {
        console.error('Backend health check failed:', err);
        setBackendStatus('offline');
        
        if (err.code === 'ECONNABORTED') {
          setBackendError('Connection timeout - Backend server is not responding');
        } else if (err.code === 'ERR_NETWORK' || err.message.includes('Network Error')) {
          setBackendError('Backend server is not running on http://localhost:8000');
        } else {
          setBackendError(`Backend error: ${err.message}`);
        }
      }
    };
    
    checkBackendHealth();
  }, []);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch events on component mount
  useEffect(() => {
    const fetchEvents = async () => {
      try {
        setIsLoadingEvents(true);
        const response = await axios.get(`${API_BASE_URL}/events`);
        setEvents(response.data.events);
      } catch (err) {
        console.error('Error fetching events:', err);
      } finally {
        setIsLoadingEvents(false);
      }
    };

    fetchEvents();
  }, []);

  const handleShowEventDetails = (event, e) => {
    e.stopPropagation(); // Prevent event card click
    setSelectedEventDetails(event);
  };

  const handleCloseEventPopup = () => {
    setSelectedEventDetails(null);
  };

  const handleEventClick = (event) => {
    // Switch to chat tab
    setActiveTab('chat');
    
    // Create a query about the event
    let query = `Tell me more about ${event.name}`;
    if (event.organization) {
      query += ` by ${event.organization}`;
    }
    
    // Set the input value
    setInputValue(query);
  };

  const sendMessageWithWebSocket = (userMessage) => {
    // Create WebSocket connection if not exists
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      const ws = new WebSocket('ws://localhost:8000/ws/multi-agent-stream');
      
      ws.onopen = () => {
        console.log('🔌 WebSocket connected');
        // Send the question
        ws.send(JSON.stringify({ question: userMessage }));
        
        // Initialize streaming message with initial status
        setStreamingMessage({
          type: 'assistant',
          content: '🔍 Starting search...',
          source: 'System',
          isStreaming: true
        });
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('📨 WebSocket message:', data);
        
        // Helper function to remove [Source: ...] prefix from content
        const cleanContent = (content) => {
          if (!content) return content;
          // Remove [Source: ...] tag if present at the beginning
          return content.replace(/^\[Source:\s*[^\]]+\]\s*/i, '').trim();
        };
        
        if (data.type === 'status') {
          // Show status update
          setStreamingMessage(prev => ({
            ...prev,
            content: cleanContent(data.content),
            source: 'System',
            tool: data.tool
          }));
        } else if (data.type === 'update') {
          // Update streaming content - show tool information
          const toolInfo = data.tool ? ` (using ${data.tool})` : '';
          const displayContent = data.agent === 'system' 
            ? cleanContent(data.content) 
            : cleanContent(data.content);
          
          setStreamingMessage(prev => ({
            ...prev,
            content: displayContent,
            source: data.agent,
            tool: data.tool
          }));
        } else if (data.type === 'final') {
          // Final message received - clean the content
          const cleanedContent = cleanContent(data.content);
          setMessages(prev => [...prev, {
            type: 'assistant',
            content: cleanedContent,
            source: data.agent,
            tool: data.tool,
            responseTime: data.response_time
          }]);
          setStreamingMessage(null);
          setIsLoading(false);
        } else if (data.type === 'error') {
          setError(`❌ ${data.content}`);
          setStreamingMessage(null);
          setIsLoading(false);
        }
      };
      
      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setError('❌ WebSocket connection error');
        setStreamingMessage(null);
        setIsLoading(false);
      };
      
      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        wsRef.current = null;
      };
      
      wsRef.current = ws;
    } else {
      // WebSocket already open, send message
      wsRef.current.send(JSON.stringify({ question: userMessage }));
      
      // Initialize streaming message with initial status
      setStreamingMessage({
        type: 'assistant',
        content: '🔍 Starting search...',
        source: 'System',
        isStreaming: true
      });
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setError('');

    // Add user message to chat
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setIsLoading(true);

    // Use WebSocket for streaming if enabled
    if (useWebSocket) {
      sendMessageWithWebSocket(userMessage);
      return;
    }

    // Fall back to HTTP POST (original behavior)
    try {
      const response = await axios.post(`${API_BASE_URL}/multi-agent-query`, {
        question: userMessage
      });

      const { answer, context, agent_used, source, response_time } = response.data;
      
      // Optional: Log which agent was used for debugging
      if (agent_used) {
        console.log(`🤖 Query answered by: ${agent_used}`);
      }
      if (source) {
        console.log(`📚 Source: ${source}`);
      }
      if (response_time) {
        console.log(`⏱️ Response time: ${response_time}s`);
      }

      // Add assistant response to chat
      setMessages(prev => [...prev, { 
        type: 'assistant', 
        content: answer,
        context: context,
        source: source || 'Unknown',
        responseTime: response_time
      }]);

    } catch (err) {
      console.error('Error sending message:', err);
      
      let errorMessage = 'An error occurred while processing your request.';
      
      if (err.code === 'ERR_NETWORK' || err.message.includes('Network Error')) {
        errorMessage = '❌ Backend server is not running. Please start the Python backend first on http://localhost:8000';
      } else if (err.response) {
        // Server responded with an error
        errorMessage = `❌ Server Error (${err.response.status}): ${err.response.data.detail || err.response.statusText}`;
      } else if (err.request) {
        // Request was made but no response received
        errorMessage = '❌ No response from backend server. Please check if the server is running.';
      }
      
      setError(errorMessage);
      
      // Add error message to chat
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: errorMessage
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const runEvaluation = async () => {
    setIsEvaluating(true);
    setError('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/evaluate-ragas`);
      setEvaluationResults(response.data);
    } catch (err) {
      console.error('Error running evaluation:', err);
      
      let errorMessage = 'An error occurred while running evaluation.';
      
      if (err.code === 'ERR_NETWORK' || err.message.includes('Network Error')) {
        errorMessage = '❌ Backend server is not running. Please start the Python backend first.';
      } else if (err.response) {
        errorMessage = `❌ Server Error (${err.response.status}): ${err.response.data.detail || err.response.statusText}`;
      }
      
      setError(errorMessage);
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleEvaluationTabClick = () => {
    setActiveTab('evaluation');
    // Auto-run evaluation when tab is clicked, but only if not already evaluating
    if (!isEvaluating && !evaluationResults) {
      runEvaluation();
    }
  };

  const runComparison = async () => {
    setIsRunningComparison(true);
    setError('');
    
    try {
      // Step 1: Switch to original method and evaluate
      console.log('Switching to original method...');
      await axios.post(`${API_BASE_URL}/retrieval-method`, { method: 'original' });
      
      console.log('Evaluating original method...');
      const originalResponse = await axios.post(`${API_BASE_URL}/evaluate-ragas`);
      
      // Step 2: Switch to naive method and evaluate
      console.log('Switching to naive method...');
      await axios.post(`${API_BASE_URL}/retrieval-method`, { method: 'naive' });
      
      console.log('Evaluating naive method...');
      const naiveResponse = await axios.post(`${API_BASE_URL}/evaluate-ragas`);
      
      // Store both results
      setComparisonResults({
        original: originalResponse.data,
        naive: naiveResponse.data
      });
      
      console.log('Comparison complete!');
    } catch (err) {
      console.error('Error running comparison:', err);
      
      let errorMessage = 'An error occurred while running comparison.';
      
      if (err.code === 'ERR_NETWORK' || err.message.includes('Network Error')) {
        errorMessage = '❌ Backend server is not running. Please start the Python backend first.';
      } else if (err.response) {
        errorMessage = `❌ Server Error (${err.response.status}): ${err.response.data.detail || err.response.statusText}`;
      }
      
      setError(errorMessage);
    } finally {
      setIsRunningComparison(false);
    }
  };

  const handleComparisonTabClick = () => {
    setActiveTab('comparison');
    // Auto-run comparison when tab is clicked
    if (!isRunningComparison && (!comparisonResults.original || !comparisonResults.naive)) {
      runComparison();
    }
  };

  // Error screen if backend is offline
  if (backendStatus === 'offline') {
    return (
      <div className="splash-screen error-screen">
        <div className="splash-content">
          <div className="error-icon">⚠️</div>
          <h1 className="splash-title error-title">Server Unavailable</h1>
          <p className="error-message">Unable to connect to the server.</p>
          <button 
            className="retry-button"
            onClick={() => window.location.reload()}
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  // Splash screen
  if (showSplash) {
    return (
      <div className="splash-screen">
        <div className="splash-content">
          <SchoolIcon className="splash-icon" />
          <h1 className="splash-title">School Events Assistant</h1>
          <div className="splash-loader">
            <div className="loader-dot"></div>
            <div className="loader-dot"></div>
            <div className="loader-dot"></div>
          </div>
        </div>
        <div className="backend-status">
          {backendStatus === 'checking' && (
            <span className="status-checking">
              🔍 Checking backend...
            </span>
          )}
          {backendStatus === 'online' && (
            <span className="status-online">
              ✅ Backend connected
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="header">
        <SchoolIcon className="header-icon" />
        <h1>School Events Assistant</h1>
      </div>

      {/* Main content area with sidebar and content */}
      <div className="main-content">
        {/* Sidebar with tabs */}
        <div className="sidebar">
          <button 
            className={`tab ${activeTab === 'events' ? 'active' : ''}`}
            onClick={() => setActiveTab('events')}
          >
            🎪&nbsp;&nbsp;Browse Events
          </button>
          <button 
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💬&nbsp;&nbsp;Chat
          </button>
          <button 
            className={`tab ${activeTab === 'evaluation' ? 'active' : ''}`}
            onClick={handleEvaluationTabClick}
          >
            📊&nbsp;&nbsp;RAGAS Evaluation
          </button>
          <button 
            className={`tab ${activeTab === 'comparison' ? 'active' : ''}`}
            onClick={handleComparisonTabClick}
          >
            📈&nbsp;&nbsp;Method Comparison
          </button>
          <button 
            className={`tab ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            ⚙️&nbsp;&nbsp;Settings
          </button>
        </div>

        {/* Content area */}
        <div className="content-area">
          {activeTab === 'events' && (
            <div className="events-tab-container">
              {isLoadingEvents ? (
                <div className="events-loading">
                  <div className="spinner-small"></div>
                  <p>Loading events...</p>
                </div>
              ) : (
                <div className="events-grid-wrapper">
                  {events.map((event, index) => (
                    <div 
                      key={event.id} 
                      className="event-card" 
                      style={{animationDelay: `${index * 0.1}s`}}
                      onClick={() => handleEventClick(event)}
                    >
                      <div className="event-header">
                        <div className="event-icon">
                          {event.type.includes('Camp') ? '🏕️' : 
                           event.type.includes('Challenge') ? '🎯' : 
                           event.type.includes('Audition') ? '🎭' :
                           event.type.includes('Clinic') ? '⚽' :
                           event.type.includes('Art') ? '🎨' : '📚'}
                        </div>
                        <h3 className="event-name">{event.name}</h3>
                      </div>
                      {event.organization && (
                        <div className="event-organization">{event.organization}</div>
                      )}
                      <p className="event-description">
                        {event.description.length > 100 
                          ? event.description.substring(0, 100) + '...' 
                          : event.description}
                      </p>
                      <div className="event-details">
                        {event.target_audience && (
                          <div className="event-detail">
                            <span className="detail-icon">👥</span>
                            <span>{event.target_audience}</span>
                          </div>
                        )}
                        <div className="event-details-row">
                          <div className="event-details-column">
                            {event.date && (
                              <div className="event-detail">
                                <span className="detail-icon">📅</span>
                                <span>{event.date}</span>
                              </div>
                            )}
                            {event.cost && (
                              <div className="event-detail">
                                <span className="detail-icon">💰</span>
                                <span className="event-cost">{event.cost}</span>
                              </div>
                            )}
                          </div>
                          <button 
                            className="event-more-info-btn"
                            onClick={(e) => handleShowEventDetails(event, e)}
                            title="View full details"
                          >
                            More Info
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'chat' && (
            <div className="chat-container">
              <div className="messages">
                {error && (
                  <div className="error-message">
                    {error}
                  </div>
                )}

          {messages.length === 0 && (
            <div className="welcome-message">
              <strong>Welcome to School Events Assistant! 🏫</strong>
              <br /><br />
              Ask me about school events, programs, and activities. Here are some examples:
              <br /><br />
              • "What coding programs are available?"
              <br />
              • "Tell me about holiday day camps"
              <br />
              • "What activities are there for middle school students?"
            </div>
          )}

          {messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <div className="message-icon">
                {message.type === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-content">
                {message.type === 'assistant' 
                  ? formatResponseText(message.content)
                  : message.content
                }
                {message.type === 'assistant' && message.source && (
                  <div style={{ 
                    marginTop: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'flex-end',
                    gap: '12px',
                    flexWrap: 'wrap'
                  }}>
                    <div className="source-badge" style={{
                      fontSize: '0.85em',
                      fontWeight: '500',
                      color: message.source === 'Local Database' ? '#1976d2' : 
                            message.source === 'Gmail' ? '#f57c00' : 
                            message.source === 'Web Search' ? '#7b1fa2' : '#616161'
                    }}>
                      📚 Source: {message.source}
                    </div>
                    {message.tool && (
                      <div style={{
                        fontSize: '0.75em',
                        color: '#757575',
                        fontStyle: 'italic',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}>
                        <span>🔧</span>
                        <span>{message.tool}</span>
                      </div>
                    )}
                    {message.responseTime && (
                      <div className="response-time-badge" style={{
                        fontSize: '0.85em',
                        fontWeight: '500',
                        color: '#666'
                      }}>
                        ⏱️ {formatResponseTime(message.responseTime)}
                      </div>
                    )}
                  </div>
                )}
              </div>
              {/* {message.context && (
                <div className="context-info">
                  <strong>Sources:</strong>
                  {message.context.map((ctx, ctxIndex) => (
                    <div key={ctxIndex} style={{ marginTop: '5px' }}>
                      {ctx}
                    </div>
                  ))}
                </div>
              )} */}
            </div>
          ))}

          {/* Show streaming message */}
          {streamingMessage && (
            <div className="message assistant streaming">
              <div className="message-icon">
                🤖
              </div>
              <div className="message-content">
                {streamingMessage.content 
                  ? formatResponseText(streamingMessage.content)
                  : <div style={{ fontStyle: 'italic', color: '#999' }}>Waiting for response...</div>
                }
                <div style={{ 
                  marginTop: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '12px',
                  flexWrap: 'wrap'
                }}>
                  {streamingMessage.source !== 'system' && (
                    <>
                      <div className="source-badge" style={{
                        fontSize: '0.85em',
                        fontWeight: '500',
                        color: streamingMessage.source === 'Gmail' ? '#f57c00' : 
                              streamingMessage.source === 'Local Database' ? '#1976d2' : 
                              streamingMessage.source === 'Web Search' ? '#7b1fa2' : '#616161',
                        animation: 'pulse 1.5s ease-in-out infinite'
                      }}>
                        📡 {streamingMessage.source} (streaming...)
                      </div>
                      {streamingMessage.tool && (
                        <div style={{
                          fontSize: '0.75em',
                          color: '#757575',
                          fontStyle: 'italic',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}>
                          <span>🔧</span>
                          <span>{streamingMessage.tool}</span>
                        </div>
                      )}
                    </>
                  )}
                  {streamingMessage.source === 'system' && (
                    <div style={{
                      fontSize: '0.85em',
                      fontWeight: '500',
                      color: '#4caf50',
                      animation: 'pulse 1.5s ease-in-out infinite',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}>
                      <span>⚡</span>
                      <span>Live Status Update</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {isLoading && !streamingMessage && (
            <div className="message assistant loading-message">
              <div className="message-icon loading-icon">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
              <div className="message-content">
                <div className="loading-text">
                  Searching school events data...
                </div>
              </div>
            </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              <div className="input-container">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about school events and programs..."
                  className="message-input"
                  disabled={isLoading}
                />
                <button
                  onClick={sendMessage}
                  disabled={!inputValue.trim() || isLoading}
                  className="send-button"
                >
                  {isLoading ? 'Sending...' : 'Send'}
                </button>
              </div>
            </div>
          )}

          {activeTab === 'evaluation' && (
            <div className="evaluation-container">
              <div className="evaluation-header">
                <h2>📊 RAGAS Evaluation Results</h2>
                <p>Evaluate the RAG pipeline using RAGAS metrics: Faithfulness, Response Relevancy, Context Precision, and Context Recall</p>
                {!isEvaluating && !evaluationResults && (
                  <p style={{ color: '#666', fontSize: '14px', fontStyle: 'italic' }}>
                    Evaluation will start automatically...
                  </p>
                )}
                {evaluationResults && (
                  <button 
                    onClick={runEvaluation} 
                    disabled={isEvaluating}
                    className="run-evaluation-button"
                  >
                    {isEvaluating ? '⏳ Re-running Evaluation...' : '🔄 Re-run Evaluation'}
                  </button>
                )}
              </div>

              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}

              {isEvaluating && (
                <div className="evaluation-loading">
                  <div className="spinner"></div>
                  <p>Running RAGAS evaluation... This may take 1-2 minutes.</p>
                </div>
              )}

              {evaluationResults && !isEvaluating && (
                <div className="evaluation-results">
                  <div className="results-summary">
                    <h3>✅ Evaluation Complete!</h3>
                    <p className="success-message">{evaluationResults.message}</p>
                    <p className="test-info">Evaluated with {evaluationResults.test_questions_count} test questions</p>
                  </div>

                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-icon">🎯</div>
                      <div className="metric-name">Faithfulness</div>
                      <div className="metric-value">{(evaluationResults.metrics.faithfulness * 100).toFixed(1)}%</div>
                      <div className="metric-description">Factual consistency with context</div>
                      <div className="metric-bar">
                        <div 
                          className="metric-bar-fill" 
                          style={{width: `${evaluationResults.metrics.faithfulness * 100}%`}}
                        ></div>
                      </div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-icon">📝</div>
                      <div className="metric-name">Answer Relevancy</div>
                      <div className="metric-value">{(evaluationResults.metrics.answer_relevancy * 100).toFixed(1)}%</div>
                      <div className="metric-description">Relevance to the question</div>
                      <div className="metric-bar">
                        <div 
                          className="metric-bar-fill" 
                          style={{width: `${evaluationResults.metrics.answer_relevancy * 100}%`}}
                        ></div>
                      </div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-icon">🎲</div>
                      <div className="metric-name">Context Precision</div>
                      <div className="metric-value">{(evaluationResults.metrics.context_precision * 100).toFixed(1)}%</div>
                      <div className="metric-description">Precision of retrieved contexts</div>
                      <div className="metric-bar">
                        <div 
                          className="metric-bar-fill" 
                          style={{width: `${evaluationResults.metrics.context_precision * 100}%`}}
                        ></div>
                      </div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-icon">📚</div>
                      <div className="metric-name">Context Recall</div>
                      <div className="metric-value">{(evaluationResults.metrics.context_recall * 100).toFixed(1)}%</div>
                      <div className="metric-description">Coverage of required context</div>
                      <div className="metric-bar">
                        <div 
                          className="metric-bar-fill" 
                          style={{width: `${evaluationResults.metrics.context_recall * 100}%`}}
                        ></div>
                      </div>
                    </div>
                  </div>

                  <div className="files-generated">
                    <p style={{color: '#666', fontStyle: 'italic', marginTop: '20px'}}>
                      💾 Evaluation results have been saved to the server for further analysis.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'comparison' && (
            <div className="evaluation-container">
              <div className="evaluation-header">
                <h2>📈 Retrieval Methods Comparison</h2>
                <p>Compare Original RAG (k=4) vs Naive Retrieval (k=10) using RAGAS metrics</p>
                {comparisonResults.original && comparisonResults.naive && (
                  <button 
                    onClick={runComparison} 
                    disabled={isRunningComparison}
                    className="run-evaluation-button"
                  >
                    {isRunningComparison ? '⏳ Re-running Comparison...' : '🔄 Re-run Comparison'}
                  </button>
                )}
              </div>

              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}

              {isRunningComparison && (
                <div className="evaluation-loading">
                  <div className="spinner"></div>
                  <p>Running comparison evaluation... This may take 3-5 minutes</p>
                  <p style={{fontSize: '14px', color: '#666'}}>
                    Evaluating both Original (k=4) and Naive (k=10) methods
                  </p>
                </div>
              )}

              {!isRunningComparison && comparisonResults.original && comparisonResults.naive && (
                <div className="comparison-results">
                  <div className="comparison-summary">
                    <h3>✅ Comparison Complete</h3>
                    <p>Both retrieval methods have been evaluated with {comparisonResults.original.test_questions_count} test questions</p>
                  </div>

                  <div className="comparison-table-container">
                    <table className="comparison-table">
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Original RAG (k=4)</th>
                          <th>Naive Retrieval (k=10)</th>
                      <th>Improvement</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>🎯 Faithfulness</strong><br/><span className="metric-desc">Factual accuracy</span></td>
                      <td className="metric-cell">
                        <div className="metric-value-large">{(comparisonResults.original.metrics.faithfulness * 100).toFixed(1)}%</div>
                        <div className="mini-bar">
                          <div className="mini-bar-fill" style={{width: `${comparisonResults.original.metrics.faithfulness * 100}%`}}></div>
                        </div>
                      </td>
                      <td className="metric-cell">
                        <div className="metric-value-large">{(comparisonResults.naive.metrics.faithfulness * 100).toFixed(1)}%</div>
                        <div className="mini-bar">
                          <div className="mini-bar-fill" style={{width: `${comparisonResults.naive.metrics.faithfulness * 100}%`}}></div>
                        </div>
                      </td>
                      <td className={`improvement-cell ${comparisonResults.naive.metrics.faithfulness >= comparisonResults.original.metrics.faithfulness ? 'positive' : 'negative'}`}>
                        {comparisonResults.naive.metrics.faithfulness >= comparisonResults.original.metrics.faithfulness ? '✅' : '⚠️'}
                        {((comparisonResults.naive.metrics.faithfulness - comparisonResults.original.metrics.faithfulness) * 100).toFixed(1)}%
                      </td>
                    </tr>

                    <tr>
                      <td><strong>📝 Answer Relevancy</strong><br/><span className="metric-desc">Relevance to question</span></td>
                      <td className="metric-cell">
                        <div className="metric-value-large">{(comparisonResults.original.metrics.answer_relevancy * 100).toFixed(1)}%</div>
                        <div className="mini-bar">
                          <div className="mini-bar-fill" style={{width: `${comparisonResults.original.metrics.answer_relevancy * 100}%`}}></div>
                        </div>
                      </td>
                      <td className="metric-cell">
                        <div className="metric-value-large">{(comparisonResults.naive.metrics.answer_relevancy * 100).toFixed(1)}%</div>
                        <div className="mini-bar">
                          <div className="mini-bar-fill" style={{width: `${comparisonResults.naive.metrics.answer_relevancy * 100}%`}}></div>
                        </div>
                      </td>
                      <td className={`improvement-cell ${comparisonResults.naive.metrics.answer_relevancy >= comparisonResults.original.metrics.answer_relevancy ? 'positive' : 'negative'}`}>
                        {comparisonResults.naive.metrics.answer_relevancy >= comparisonResults.original.metrics.answer_relevancy ? '✅' : '⚠️'}
                        {((comparisonResults.naive.metrics.answer_relevancy - comparisonResults.original.metrics.answer_relevancy) * 100).toFixed(1)}%
                      </td>
                    </tr>

                    <tr>
                      <td><strong>🎲 Context Precision</strong><br/><span className="metric-desc">Precision of contexts</span></td>
                      <td className="metric-cell">
                        <div className="metric-value-large">{(comparisonResults.original.metrics.context_precision * 100).toFixed(1)}%</div>
                        <div className="mini-bar">
                          <div className="mini-bar-fill" style={{width: `${comparisonResults.original.metrics.context_precision * 100}%`}}></div>
                        </div>
                      </td>
                      <td className="metric-cell">
                        <div className="metric-value-large">{(comparisonResults.naive.metrics.context_precision * 100).toFixed(1)}%</div>
                        <div className="mini-bar">
                          <div className="mini-bar-fill" style={{width: `${comparisonResults.naive.metrics.context_precision * 100}%`}}></div>
                        </div>
                      </td>
                      <td className={`improvement-cell ${comparisonResults.naive.metrics.context_precision >= comparisonResults.original.metrics.context_precision ? 'positive' : 'negative'}`}>
                        {comparisonResults.naive.metrics.context_precision >= comparisonResults.original.metrics.context_precision ? '✅' : '⚠️'}
                        {((comparisonResults.naive.metrics.context_precision - comparisonResults.original.metrics.context_precision) * 100).toFixed(1)}%
                      </td>
                    </tr>

                    <tr>
                      <td><strong>🔍 Context Recall</strong><br/><span className="metric-desc">Completeness of context</span></td>
                      <td className="metric-cell">
                        <div className="metric-value-large">{(comparisonResults.original.metrics.context_recall * 100).toFixed(1)}%</div>
                        <div className="mini-bar">
                          <div className="mini-bar-fill" style={{width: `${comparisonResults.original.metrics.context_recall * 100}%`}}></div>
                        </div>
                      </td>
                      <td className="metric-cell">
                        <div className="metric-value-large">{(comparisonResults.naive.metrics.context_recall * 100).toFixed(1)}%</div>
                        <div className="mini-bar">
                          <div className="mini-bar-fill" style={{width: `${comparisonResults.naive.metrics.context_recall * 100}%`}}></div>
                        </div>
                      </td>
                      <td className={`improvement-cell ${comparisonResults.naive.metrics.context_recall >= comparisonResults.original.metrics.context_recall ? 'positive' : 'negative'}`}>
                        {comparisonResults.naive.metrics.context_recall >= comparisonResults.original.metrics.context_recall ? '✅' : '⚠️'}
                        {((comparisonResults.naive.metrics.context_recall - comparisonResults.original.metrics.context_recall) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="comparison-insights">
                <h4>💡 Key Insights</h4>
                <div className="insights-grid">
                  <div className="insight-card">
                    <h5>Original RAG (k=4)</h5>
                    <ul>
                      <li>Retrieves 4 most relevant documents</li>
                      <li>Faster query processing</li>
                      <li>More focused context</li>
                    </ul>
                  </div>
                  <div className="insight-card">
                    <h5>Naive Retrieval (k=10)</h5>
                    <ul>
                      <li>Retrieves 10 most relevant documents</li>
                      <li>LCEL chain pattern from Advanced Retrieval</li>
                      <li>More comprehensive context</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="settings-container">
              <div className="settings-section">
                <h3>🔌 Connection Settings</h3>
                <div className="setting-item">
                  <div className="setting-content">
                    <div className="setting-label-group">
                      <label htmlFor="websocket-toggle" className="setting-label">
                        Real-time Streaming (WebSocket)
                      </label>
                      <p className="setting-description">
                        Enable live streaming of agent responses as they are generated. 
                        When disabled, responses will be delivered in full after completion.
                      </p>
                    </div>
                    <div className="toggle-switch">
                      <input
                        id="websocket-toggle"
                        type="checkbox"
                        checked={useWebSocket}
                        onChange={(e) => setUseWebSocket(e.target.checked)}
                        className="toggle-input"
                      />
                      <label htmlFor="websocket-toggle" className="toggle-label">
                        <span className="toggle-button"></span>
                      </label>
                    </div>
                  </div>
                  {useWebSocket && (
                    <div className="setting-status">
                      <span className="status-indicator active"></span>
                      <span className="status-text">Live updates enabled</span>
                    </div>
                  )}
                  {!useWebSocket && (
                    <div className="setting-status">
                      <span className="status-indicator inactive"></span>
                      <span className="status-text">Using standard HTTP requests</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="settings-section">
                <h3>ℹ️ About</h3>
                <div className="about-info">
                  <p><strong>School Events RAG Application</strong></p>
                  <p>Multi-agent system for searching and discovering school events</p>
                  <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
                    <li>Gmail integration for email search</li>
                    <li>Local database with curated school events</li>
                    <li>Web search for the latest information</li>
                    <li>Real-time streaming via WebSocket</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Event Details Popup */}
      <EventPopup 
        event={selectedEventDetails}
        onClose={handleCloseEventPopup}
        onAskAI={handleEventClick}
      />
    </div>
  );
}

export default App;