import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { ReactComponent as SchoolIcon } from './assets/SchoolIcon.svg';

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

// Function to format LLM response text into HTML
const formatResponseText = (text) => {
  if (!text) return '';
  
  // Split into paragraphs
  const paragraphs = text.split('\n\n').filter(p => p.trim());
  
  return paragraphs.map((para, idx) => {
    const trimmed = para.trim();
    
    // Check if it's a bullet point list (lines starting with • or -)
    if (trimmed.includes('\n•') || trimmed.includes('\n-') || trimmed.startsWith('•') || trimmed.startsWith('-')) {
      const lines = trimmed.split('\n');
      const listItems = [];
      let currentList = [];
      
      lines.forEach(line => {
        const trimmedLine = line.trim();
        if (trimmedLine.startsWith('•') || trimmedLine.startsWith('-')) {
          currentList.push(trimmedLine.replace(/^[•-]\s*/, ''));
        } else if (trimmedLine) {
          if (currentList.length > 0) {
            listItems.push({ type: 'list', items: currentList });
            currentList = [];
          }
          listItems.push({ type: 'text', content: trimmedLine });
        }
      });
      
      if (currentList.length > 0) {
        listItems.push({ type: 'list', items: currentList });
      }
      
      return (
        <div key={idx} className="response-section">
          {listItems.map((item, i) => 
            item.type === 'list' ? (
              <ul key={i} className="formatted-list">
                {item.items.map((li, j) => (
                  <li key={j}>{renderMarkdownText(li, j)}</li>
                ))}
              </ul>
            ) : (
              <p key={i} className="response-text">{renderMarkdownText(item.content, i)}</p>
            )
          )}
        </div>
      );
    }
    
    // Check if it's a header (ends with : or is all caps and short)
    if (trimmed.endsWith(':') && trimmed.length < 50) {
      return <h4 key={idx} className="response-header">{renderMarkdownText(trimmed, idx)}</h4>;
    }
    
    // Regular paragraph
    return <p key={idx} className="response-text">{renderMarkdownText(trimmed, idx)}</p>;
  });
};

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('events'); // 'events', 'chat', 'evaluation', or 'comparison'
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [comparisonResults, setComparisonResults] = useState({
    original: null,
    naive: null
  });
  const [isRunningComparison, setIsRunningComparison] = useState(false);
  const [currentMethod, setCurrentMethod] = useState('naive');
  const [events, setEvents] = useState([]);
  const [isLoadingEvents, setIsLoadingEvents] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setError('');

    // Add user message to chat
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/query`, {
        question: userMessage
      });

      const { answer, context } = response.data;

      // Add assistant response to chat
      setMessages(prev => [...prev, { 
        type: 'assistant', 
        content: answer,
        context: context 
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
            🎪 Browse Events
          </button>
          <button 
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💬 Chat
          </button>
          <button 
            className={`tab ${activeTab === 'evaluation' ? 'active' : ''}`}
            onClick={handleEvaluationTabClick}
          >
            📊 RAGAS Evaluation
          </button>
          <button 
            className={`tab ${activeTab === 'comparison' ? 'active' : ''}`}
            onClick={handleComparisonTabClick}
          >
            📈 Method Comparison
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

          {isLoading && (
            <div className="message loading">
              Searching school events data...
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
        </div>
      </div>
    </div>
  );
}

export default App;