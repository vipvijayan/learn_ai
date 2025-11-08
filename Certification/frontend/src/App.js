import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { ReactComponent as SchoolIcon } from './assets/SchoolIcon.svg';
import EventPopup from './components/EventPopup';
import SchoolSelection from './components/SchoolSelection';
import Login from './components/Login';

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
    return `${hours}h: ${minutes}m: ${secs}s`;
  } else {
    return `${minutes}m: ${secs}s`;
  }
};

// Function to format LLM response text into HTML
const formatResponseText = (text) => {
  if (!text) return '';
  
  // Split by single newlines first to handle headers properly
  const lines = text.split('\n').map(l => l.trim()).filter(l => l);
  const elements = [];
  let currentParagraph = [];
  let inEventBlock = false;
  let eventBlockContent = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Skip intro lines that mention "found the following" or similar 
    if (line.match(/(?:found the following|here (?:are|is)|based on)/i) && line.length < 150) {
      continue;
    }
    
    // Check if this line introduces an event (e.g., "...database: EventName")
    const eventIntroMatch = line.match(/(?:database|search|inbox):\s+(.+)$/i);
    if (eventIntroMatch && !inEventBlock) {
      // Extract event title from the intro line
      const eventTitle = eventIntroMatch[1].trim();
      // Flush current paragraph
      if (currentParagraph.length > 0) {
        elements.push({ type: 'paragraph', content: currentParagraph.join('\n') });
        currentParagraph = [];
      }
      inEventBlock = true;
      eventBlockContent.push({ type: 'event-title', content: eventTitle, number: '' });
      continue;
    }
    
    // Check if this is an event title (numbered item that looks like a title)
    const eventTitleMatch = line.match(/^(\d+)\.\s+(.+)$/);
    if (eventTitleMatch && !line.includes(':') && line.length > 10) {
      // This is likely an event title - start a new event block
      if (inEventBlock && eventBlockContent.length > 0) {
        // Save previous event block
        elements.push({ type: 'event-block', content: eventBlockContent });
        eventBlockContent = [];
      }
      inEventBlock = true;
      eventBlockContent.push({ type: 'event-title', content: eventTitleMatch[2], number: eventTitleMatch[1] });
      continue;
    }
    
    // If we're in an event block, collect content
    if (inEventBlock) {
      // Check for field labels in bullet points or plain format
      const fieldMatch = line.match(/^[•-]?\s*(Organizer|Type|Category|Registration|Call to Action|Contact Information|Website|Email|Phone|From|Summary|Key Details|Date|Location|Time|Description):\s*(.+)$/i);
      if (fieldMatch) {
        const label = fieldMatch[1];
        const content = fieldMatch[2].trim();
        eventBlockContent.push({ type: 'field', label, content });
        continue;
      }
      
      // Check if we're starting a new numbered item (next event)
      if (line.match(/^\d+\.\s+/) && !line.includes(':') && line.length > 10) {
        // Save current event block and start new one
        if (eventBlockContent.length > 0) {
          elements.push({ type: 'event-block', content: eventBlockContent });
          eventBlockContent = [];
        }
        const match = line.match(/^(\d+)\.\s+(.+)$/);
        eventBlockContent.push({ type: 'event-title', content: match[2], number: match[1] });
        continue;
      }
      
      // Check if this line signals end of event block (no colon, not a bullet, looks like regular text)
      if (!line.startsWith('•') && !line.startsWith('-') && !line.match(/^\d+\./) && line.length > 20 && !line.match(/:/)) {
        // End event block and add this as regular text
        if (eventBlockContent.length > 0) {
          elements.push({ type: 'event-block', content: eventBlockContent });
          eventBlockContent = [];
          inEventBlock = false;
        }
        currentParagraph.push(line);
        continue;
      }
      
      // Add other content to current event block (strip bullet if present)
      const cleanLine = line.replace(/^[•-]\s*/, '');
      if (cleanLine) {
        eventBlockContent.push({ type: 'text', content: cleanLine });
      }
      continue;
    }
    
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
  
  // Flush remaining event block
  if (inEventBlock && eventBlockContent.length > 0) {
    elements.push({ type: 'event-block', content: eventBlockContent });
  }
  
  // Flush remaining paragraph
  if (currentParagraph.length > 0) {
    elements.push({ type: 'paragraph', content: currentParagraph.join('\n') });
  }
  
  // Render elements
  return elements.map((element, idx) => {
    switch (element.type) {
      case 'event-block':
        return <div key={idx} className="event-card" style={{
          border: '2px solid #e3f2fd',
          borderRadius: '12px',
          padding: '24px',
          marginBottom: '24px',
          backgroundColor: '#fafafa',
          transition: 'all 0.3s ease'
        }}>
          {element.content.map((item, i) => {
            if (item.type === 'event-title') {
              return <h3 key={i} style={{
                color: '#1976d2',
                marginBottom: '16px',
                fontSize: '1.3em',
                fontWeight: '600',
                borderBottom: '2px solid #e3f2fd',
                paddingBottom: '12px'
              }}>
                {item.number}. {renderMarkdownText(item.content, i)}
              </h3>;
            } else if (item.type === 'field') {
              return <div key={i} style={{
                marginBottom: '12px',
                paddingLeft: '8px',
                borderLeft: '3px solid #90caf9'
              }}>
                <strong style={{ color: '#1565c0', fontSize: '0.95em' }}>{item.label}:</strong>
                {' '}
                <span style={{ color: '#424242' }}>{renderMarkdownText(item.content, i)}</span>
              </div>;
            } else if (item.type === 'text') {
              return <p key={i} style={{ color: '#616161', lineHeight: '1.6', marginTop: '8px' }}>
                {renderMarkdownText(item.content, i)}
              </p>;
            }
            return null;
          })}
        </div>;
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
  // Authentication state
  const [user, setUser] = useState(null); // Current logged-in user
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [needsSchoolSelection, setNeedsSchoolSelection] = useState(false);
  
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('chat'); // 'events', 'chat', 'comparison', or 'settings'
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
  const [schools, setSchools] = useState([]); // List of schools
  const [selectedSchoolDistrict, setSelectedSchoolDistrict] = useState(null); // Selected school from localStorage
  const [showSchoolSelection, setShowSchoolSelection] = useState(false); // Show school selection screen (LEGACY - using auth system now)
  const [copiedMessageIndex, setCopiedMessageIndex] = useState(null); // Track which message was copied
  const [showCopyToast, setShowCopyToast] = useState(false); // Show copy success toast
  const [bookmarks, setBookmarks] = useState([]); // Store bookmarked messages
  const [showBookmarkToast, setShowBookmarkToast] = useState(false); // Show bookmark toast
  const [bookmarkToastMessage, setBookmarkToastMessage] = useState(''); // Toast message text

  // Check for existing session on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        const parsedUser = JSON.parse(savedUser);
        setUser(parsedUser);
        setIsAuthenticated(true);
        // Check if user needs to select schools (no schools selected yet)
        if (!parsedUser.schools || parsedUser.schools.length === 0) {
          setNeedsSchoolSelection(true);
        }
      } catch (err) {
        console.error('Error parsing saved user:', err);
        localStorage.removeItem('user');
      }
    }
  }, []);

  // Handle successful login
  const handleLoginSuccess = async (userData) => {
    // Fetch Gmail connection status
    try {
      const gmailStatus = await axios.get(`${API_BASE_URL}/api/auth/gmail/status`, {
        params: { email: userData.email }
      });
      
      if (gmailStatus.data.connected) {
        userData.gmail_email = gmailStatus.data.gmail_email;
        userData.gmail_connected_at = gmailStatus.data.connected_at;
      }
    } catch (error) {
      console.error('Error fetching Gmail status:', error);
    }
    
    setUser(userData);
    setIsAuthenticated(true);
    localStorage.setItem('user', JSON.stringify(userData));
    
    // Check if user needs to select schools (no schools selected yet)
    if (!userData.schools || userData.schools.length === 0) {
      setNeedsSchoolSelection(true);
    }
  };

  // Handle school selection
  const handleSchoolSelected = (userData) => {
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
    setNeedsSchoolSelection(false);
  };

  // Handle logout
  const handleLogout = () => {
    // Clear all state
    setUser(null);
    setIsAuthenticated(false);
    setNeedsSchoolSelection(false);
    setMessages([]);
    setBookmarks([]);
    
    // Clear all localStorage
    localStorage.clear();
    
    // Switch to chat tab
    setActiveTab('chat');
    
    console.log('User logged out, localStorage cleared');
  };
  
  // Connect Gmail account
  const handleConnectGmail = async () => {
    try {
      // Get OAuth authorization URL
      const response = await axios.get(`${API_BASE_URL}/api/auth/gmail/authorize`, {
        params: { email: user.email }
      });
      
      // Open OAuth popup
      const authUrl = response.data.authorization_url;
      const popup = window.open(authUrl, 'Gmail OAuth', 'width=600,height=700');
      
      // Poll for popup closure and refresh user data
      const checkPopup = setInterval(async () => {
        if (popup.closed) {
          clearInterval(checkPopup);
          
          // Check Gmail connection status
          const statusResponse = await axios.get(`${API_BASE_URL}/api/auth/gmail/status`, {
            params: { email: user.email }
          });
          
          if (statusResponse.data.connected) {
            // Update user object with Gmail info
            setUser({
              ...user,
              gmail_email: statusResponse.data.gmail_email,
              gmail_connected_at: statusResponse.data.connected_at
            });
            alert('Gmail connected successfully!');
          }
        }
      }, 500);
      
    } catch (error) {
      console.error('Error connecting Gmail:', error);
      alert('Failed to connect Gmail. Please try again.');
    }
  };
  
  // Disconnect Gmail account
  const handleDisconnectGmail = async () => {
    if (!window.confirm('Are you sure you want to disconnect your Gmail account?')) {
      return;
    }
    
    try {
      await axios.post(`${API_BASE_URL}/api/auth/gmail/disconnect`, {
        email: user.email
      });
      
      // Update user object to remove Gmail info
      setUser({
        ...user,
        gmail_email: null,
        gmail_connected_at: null
      });
      
      alert('Gmail disconnected successfully');
    } catch (error) {
      console.error('Error disconnecting Gmail:', error);
      alert('Failed to disconnect Gmail');
    }
  };
  
  // Copy message content to clipboard
  const copyToClipboard = (content, index) => {
    // Strip markdown formatting for plain text copy
    const plainText = content.replace(/\*\*/g, '');
    
    navigator.clipboard.writeText(plainText).then(() => {
      setCopiedMessageIndex(index);
      setShowCopyToast(true);
      
      // Reset copied state after animation
      setTimeout(() => setCopiedMessageIndex(null), 2000);
      
      // Hide toast after 3 seconds
      setTimeout(() => setShowCopyToast(false), 3000);
    }).catch(err => {
      console.error('Failed to copy:', err);
    });
  };
  
  // Bookmark message
  const bookmarkMessage = async (message, index) => {
    // Check if already bookmarked
    const isAlreadyBookmarked = bookmarks.some(
      bm => bm.message_content === message.content && bm.message_type === message.type
    );
    
    if (isAlreadyBookmarked) {
      setBookmarkToastMessage('Already bookmarked!');
      setShowBookmarkToast(true);
      setTimeout(() => setShowBookmarkToast(false), 3000);
      return;
    }

    try {
      const bookmarkId = `bookmark_${Date.now()}_${index}`;
      const response = await axios.post('http://localhost:8000/api/bookmarks/add', {
        email: user.email,
        bookmark_id: bookmarkId,
        message_type: message.type,
        message_content: message.content,
        message_context: JSON.stringify({ timestamp: new Date().toISOString() }),
        message_source: message.source || null,
        message_index: index
      });

      if (response.data.success) {
        // Reload bookmarks from server
        await loadBookmarks();
        setBookmarkToastMessage('Bookmarked!');
        setShowBookmarkToast(true);
        setTimeout(() => setShowBookmarkToast(false), 3000);
      }
    } catch (error) {
      console.error('Error adding bookmark:', error);
      setBookmarkToastMessage('Failed to bookmark');
      setShowBookmarkToast(true);
      setTimeout(() => setShowBookmarkToast(false), 3000);
    }
  };
  
  // Remove bookmark
  const removeBookmark = async (bookmarkId) => {
    try {
      const response = await axios.post('http://localhost:8000/api/bookmarks/remove', {
        email: user.email,
        bookmark_id: bookmarkId
      });

      if (response.data.success) {
        // Reload bookmarks from server
        await loadBookmarks();
        setBookmarkToastMessage('Bookmark removed');
        setShowBookmarkToast(true);
        setTimeout(() => setShowBookmarkToast(false), 3000);
      }
    } catch (error) {
      console.error('Error removing bookmark:', error);
      setBookmarkToastMessage('Failed to remove bookmark');
      setShowBookmarkToast(true);
      setTimeout(() => setShowBookmarkToast(false), 3000);
    }
  };
  
  // Check if message is bookmarked
  const isMessageBookmarked = (message) => {
    return bookmarks.some(
      bm => bm.message_content === message.content && bm.message_type === message.type
    );
  };

  // Load bookmarks from backend
  const loadBookmarks = async () => {
    if (!user || !user.email) return;
    
    try {
      const response = await axios.get(`http://localhost:8000/api/bookmarks/${user.email}`);
      if (response.data.success) {
        setBookmarks(response.data.bookmarks);
      }
    } catch (error) {
      console.error('Error loading bookmarks:', error);
    }
  };
  
  // Load bookmarks from backend on mount and when user changes
  useEffect(() => {
    if (user && user.email) {
      loadBookmarks();
    } else {
      setBookmarks([]);
    }
  }, [user]);
  

  
  // Check backend health on mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`, {
          timeout: 5000
        });
        
        if (response.status === 200) {
          setBackendStatus('online');
          
          // Hide splash screen after 1 second if backend is online
          setTimeout(() => {
            setShowSplash(false);
          }, 1000);
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

  // Fetch schools on component mount
  useEffect(() => {
    const fetchSchools = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/schools`);
        setSchools(response.data.schools);
        console.log('Schools loaded:', response.data.schools);
      } catch (err) {
        console.error('Error fetching schools:', err);
      }
    };

    fetchSchools();
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
        
        // Extract email suffixes and district names from all selected schools
        const email_suffixes = user?.schools?.map(s => s.email_suffix).filter(Boolean) || [];
        const school_districts = user?.schools?.map(s => s.name).filter(Boolean) || [];
        
        console.log('📧 WebSocket sending query with school data:', {
          schools_count: user?.schools?.length || 0,
          email_suffixes,
          school_districts
        });
        
        // Send the question with email suffixes, school districts, and user email
        console.log('📧 Sending WebSocket message with user email:', user?.email);
        console.log('📧 Full user object:', user);
        
        ws.send(JSON.stringify({ 
          question: userMessage,
          user_email: user?.email,  // For per-user Gmail authentication
          email_suffixes: email_suffixes.length > 0 ? email_suffixes : null,
          school_districts: school_districts.length > 0 ? school_districts : null,
          // Legacy fields for backwards compatibility
          email_suffix: selectedSchoolDistrict?.email_suffix || (email_suffixes.length > 0 ? email_suffixes[0] : null),
          school_district: selectedSchoolDistrict?.district || (school_districts.length > 0 ? school_districts[0] : null)
        }));
        
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
            responseTime: data.response_time,
            evaluation: null // Will be updated when evaluation arrives
          }]);
          setStreamingMessage(null);
          setIsLoading(false);
        } else if (data.type === 'evaluation') {
          // Evaluation message received - update the last assistant message
          console.log('📊 Evaluation received from WebSocket:', data.evaluation);
          setMessages(prev => {
            const newMessages = [...prev];
            // Find the last assistant message and add evaluation
            for (let i = newMessages.length - 1; i >= 0; i--) {
              if (newMessages[i].type === 'assistant') {
                newMessages[i] = {
                  ...newMessages[i],
                  evaluation: data.evaluation
                };
                break;
              }
            }
            return newMessages;
          });
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
      // Extract email suffixes and district names from all selected schools
      const email_suffixes = user?.schools?.map(s => s.email_suffix).filter(Boolean) || [];
      const school_districts = user?.schools?.map(s => s.name).filter(Boolean) || [];
      
      console.log('📧 Sending query with school data:', {
        schools_count: user?.schools?.length || 0,
        email_suffixes,
        school_districts
      });
      
      const response = await axios.post(`${API_BASE_URL}/multi-agent-query`, {
        question: userMessage,
        email_suffixes: email_suffixes.length > 0 ? email_suffixes : null,
        school_districts: school_districts.length > 0 ? school_districts : null,
        // Legacy fields for backwards compatibility
        email_suffix: selectedSchoolDistrict?.email_suffix || (email_suffixes.length > 0 ? email_suffixes[0] : null),
        school_district: selectedSchoolDistrict?.district || (school_districts.length > 0 ? school_districts[0] : null)
      });

      const { answer, context, agent_used, source, response_time, evaluation } = response.data;
      
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
      if (evaluation) {
        console.log(`📊 Evaluation - Faithfulness: ${evaluation.faithfulness}, Relevancy: ${evaluation.response_relevancy}, Status: ${evaluation.status}`);
      }

      // Add assistant response to chat
      setMessages(prev => [...prev, { 
        type: 'assistant', 
        content: answer,
        context: context,
        source: source || 'Unknown',
        responseTime: response_time,
        evaluation: evaluation // Add evaluation metrics
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
          <h1 className="splash-title">School Assistant</h1>
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

  // Show login if not authenticated
  if (!isAuthenticated) {
    return <Login onLoginSuccess={handleLoginSuccess} />;
  }

  // Show school selection if user hasn't selected a school
  if (needsSchoolSelection) {
    console.log('Showing SchoolSelection component, user:', user);
    return <SchoolSelection user={user} onSchoolSelected={handleSchoolSelected} />;
  }

  // Show old school selection if not selected yet (legacy support)
  if (showSchoolSelection && !selectedSchoolDistrict) {
    // Skip this - we'll use the new authentication flow
    setShowSchoolSelection(false);
  }

  return (
    <div className="app">
      {/* Copy Toast Notification */}
      {showCopyToast && (
        <div className="copy-toast">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          <span>Copied to clipboard!</span>
        </div>
      )}

      {/* Bookmark Toast Notification */}
      {showBookmarkToast && (
        <div className="copy-toast" style={{ backgroundColor: '#4caf50' }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
          </svg>
          <span>{bookmarkToastMessage}</span>
        </div>
      )}

      <div className="header">
        <div 
          onClick={() => {
            console.log('School icon clicked!');
            console.log('Current state - isAuthenticated:', isAuthenticated, 'needsSchoolSelection:', needsSchoolSelection, 'user:', user);
            setNeedsSchoolSelection(true);
            console.log('Set needsSchoolSelection to true');
          }}
          style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}
          title="Change school"
        >
          <SchoolIcon className="header-icon" />
        </div>
        <h1>School Assistant</h1>
        {selectedSchoolDistrict && (
          <div 
            className="selected-school-badge"
            onClick={() => {
              setNeedsSchoolSelection(true);
            }}
            title="Click to change school"
          >
            📍 {selectedSchoolDistrict.district}
          </div>
        )}
      </div>

      {/* Main content area with sidebar and content */}
      <div className="main-content">
        {/* Sidebar with tabs */}
        <div className="sidebar">
          <button 
            className={`tab ${activeTab === 'events' ? 'active' : ''}`}
            onClick={() => setActiveTab('events')}
          >
            🎪&nbsp;&nbsp;Events
          </button>
          <button 
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💬&nbsp;&nbsp;Chat
          </button>
          <button 
            className={`tab ${activeTab === 'bookmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('bookmarks')}
          >
            🔖&nbsp;&nbsp;Bookmarks
          </button>
          {/* RAGAS Evaluation now happens automatically for each response */}
          {/* <button 
            className={`tab ${activeTab === 'evaluation' ? 'active' : ''}`}
            onClick={handleEvaluationTabClick}
          >
            📊&nbsp;&nbsp;RAGAS Evaluation
          </button> */}
          {/* <button 
            className={`tab ${activeTab === 'comparison' ? 'active' : ''}`}
            onClick={handleComparisonTabClick}
          >
            📈&nbsp;&nbsp;Method Comparison
          </button> */}
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
              <strong>Welcome to School Assistant! 🏫</strong>
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
                <div className="message-text">
                  {message.type === 'assistant' 
                    ? formatResponseText(message.content)
                    : message.content
                  }
                </div>
                {message.type === 'assistant' && message.source && (
                  <div style={{ 
                    marginTop: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '12px',
                    flexWrap: 'wrap'
                  }}>
                    {/* Left side: Evaluation status */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {message.evaluation && message.evaluation.status === 'completed' && (
                        <div style={{ 
                          fontSize: '0.85em', 
                          color: '#4caf50',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px'
                        }}>
                          <span>✓</span>
                          <span>Evaluated</span>
                        </div>
                      )}
                    </div>
                    
                    {/* Right side: Source, Tool, Response Time, Copy, Bookmark */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
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
                      <button 
                        className="copy-button"
                        onClick={() => copyToClipboard(message.content, index)}
                        title="Copy message"
                        style={{
                          position: 'relative',
                          right: 'auto',
                          top: 'auto',
                          background: 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          padding: '4px',
                          opacity: 0.6,
                          transition: 'opacity 0.2s'
                        }}
                      >
                        {copiedMessageIndex === index ? (
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="20 6 9 17 4 12"></polyline>
                          </svg>
                        ) : (
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                          </svg>
                        )}
                      </button>
                      <button 
                        className="bookmark-button"
                        onClick={() => bookmarkMessage(message, index)}
                        title={isMessageBookmarked(message) ? "Already bookmarked" : "Bookmark message"}
                        style={{
                          position: 'relative',
                          right: 'auto',
                          top: 'auto',
                          background: 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          padding: '4px',
                          opacity: isMessageBookmarked(message) ? 1 : 0.6,
                          transition: 'opacity 0.2s'
                        }}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill={isMessageBookmarked(message) ? "#f57c00" : "none"} stroke="currentColor" strokeWidth="2">
                          <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
                        </svg>
                      </button>
                    </div>
                  </div>
                )}
                {/* User message bottom row with copy button */}
                {message.type === 'user' && (
                  <div style={{ 
                    marginTop: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'flex-end',
                    gap: '12px'
                  }}>
                    <button 
                      className="copy-button"
                      onClick={() => copyToClipboard(message.content, index)}
                      title="Copy message"
                      style={{
                        position: 'relative',
                        right: 'auto',
                        top: 'auto',
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '4px',
                        opacity: 0.6,
                        transition: 'opacity 0.2s'
                      }}
                    >
                      {copiedMessageIndex === index ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                      ) : (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                      )}
                    </button>
                  </div>
                )}
                {/* Hidden detailed metrics for future use */}
                {message.type === 'assistant' && message.evaluation && message.evaluation.status === 'completed' && (
                  <div style={{ display: 'none' }}>
                    <div style={{ fontWeight: '600', color: '#555' }}>
                      📊 RAGAS Evaluation:
                    </div>
                    <div>Faithfulness: {(message.evaluation.faithfulness * 100).toFixed(1)}%</div>
                    <div>Relevancy: {(message.evaluation.response_relevancy * 100).toFixed(1)}%</div>
                  </div>
                )}
                {message.type === 'assistant' && message.evaluation && message.evaluation.status === 'failed' && (
                  <div style={{ 
                    marginTop: '12px',
                    padding: '8px',
                    background: '#fff3e0',
                    borderRadius: '6px',
                    fontSize: '0.8em',
                    color: '#e65100',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    <span>⚠️</span>
                    <span>Evaluation failed</span>
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
                      <span>Waiting for the magic...</span>
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

          {activeTab === 'bookmarks' && (
            <div className="chat-container">
              <div className="bookmarks-container">
              {bookmarks.length === 0 ? (
                <div className="welcome-message" style={{ textAlign: 'center', padding: '40px' }}>
                  <strong>No bookmarks yet</strong>
                  <br /><br />
                  Bookmark assistant messages from the Chat tab to save them here for later reference.
                  <br /><br />
                  Look for the 🔖 icon on assistant messages!
                </div>
              ) : (
                <div className="bookmarks-list" style={{ paddingBottom: '60px' }}>
                  {bookmarks.map((bookmark, index) => (
                    <div key={bookmark.bookmark_id} className="bookmark-item" style={{
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px',
                      padding: '16px',
                      marginBottom: '16px',
                      backgroundColor: '#fafafa',
                      position: 'relative'
                    }}>
                      <button
                        onClick={() => removeBookmark(bookmark.bookmark_id)}
                        style={{
                          position: 'absolute',
                          top: '12px',
                          right: '12px',
                          background: '#ff5252',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          padding: '6px 12px',
                          cursor: 'pointer',
                          fontSize: '0.85em',
                          fontWeight: '500',
                          transition: 'background 0.2s'
                        }}
                        onMouseOver={(e) => e.target.style.background = '#ff1744'}
                        onMouseOut={(e) => e.target.style.background = '#ff5252'}
                        title="Remove bookmark"
                      >
                        Remove
                      </button>
                      <div style={{ 
                        fontSize: '0.85em', 
                        color: '#666', 
                        marginBottom: '8px',
                        paddingRight: '80px'
                      }}>
                        Saved on {new Date(bookmark.created_at).toLocaleString()}
                      </div>
                      <div className="message-content" style={{ marginTop: '12px' }}>
                        <div className="message-text">
                          {formatResponseText(bookmark.message_content)}
                        </div>
                        {bookmark.message_source && (
                          <div style={{ 
                            marginTop: '12px',
                            fontSize: '0.85em',
                            color: '#666'
                          }}>
                            📚 Source: {bookmark.message_source}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="settings-container">
              <div className="settings-section">
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  marginBottom: '16px'
                }}>
                  <h3 style={{ margin: 0 }}>👤 Account & Gmail</h3>
                  <button 
                    className="logout-button"
                    onClick={handleLogout}
                    style={{
                      padding: '8px 16px',
                      backgroundColor: '#f44336',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      fontSize: '0.9em',
                      fontWeight: '600',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.backgroundColor = '#d32f2f';
                      e.target.style.transform = 'translateY(-2px)';
                      e.target.style.boxShadow = '0 4px 8px rgba(244, 67, 54, 0.3)';
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.backgroundColor = '#f44336';
                      e.target.style.transform = 'translateY(0)';
                      e.target.style.boxShadow = 'none';
                    }}
                  >
                    <span>🚪</span>
                    <span>Logout</span>
                  </button>
                </div>
                
                {/* Gmail Account Status */}
                <div className="setting-item">
                  <div className="setting-content">
                    <div className="setting-label-group">
                      <label className="setting-label">
                        Gmail Account
                      </label>
                      {user?.gmail_email ? (
                        <div style={{
                          padding: '12px',
                          background: '#e8f5e9',
                          borderRadius: '8px',
                          border: '1px solid #4caf50',
                          marginTop: '8px'
                        }}>
                          <p className="setting-description" style={{ marginBottom: '4px', color: '#2e7d32' }}>
                            ✓ Signed in as: <strong>{user.gmail_email}</strong>
                          </p>
                          {user.gmail_connected_at && (
                            <p className="setting-description" style={{ fontSize: '0.85em', color: '#666', marginTop: '4px' }}>
                              Connected on {new Date(user.gmail_connected_at).toLocaleDateString()}
                            </p>
                          )}
                          <p className="setting-description" style={{ fontSize: '0.85em', color: '#666', marginTop: '8px' }}>
                            Your Gmail is connected and can be searched for school-related emails.
                          </p>
                          
                          {/* Disconnect Button */}
                          <button
                            onClick={handleDisconnectGmail}
                            style={{
                              padding: '8px 14px',
                              background: '#fff',
                              color: '#f44336',
                              border: '2px solid #f44336',
                              borderRadius: '6px',
                              cursor: 'pointer',
                              fontSize: '0.85em',
                              fontWeight: '500',
                              transition: 'all 0.2s',
                              marginTop: '12px'
                            }}
                            onMouseEnter={(e) => {
                              e.target.style.background = '#f44336';
                              e.target.style.color = 'white';
                            }}
                            onMouseLeave={(e) => {
                              e.target.style.background = '#fff';
                              e.target.style.color = '#f44336';
                            }}
                          >
                            Disconnect Gmail
                          </button>
                        </div>
                      ) : (
                        <p className="setting-description" style={{ color: '#f57c00', marginTop: '8px' }}>
                          ⚠️ Gmail not connected. Please sign out and sign in again to connect Gmail.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="settings-section">
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  marginBottom: '16px'
                }}>
                  <h3 style={{ margin: 0 }}>🏫 School Settings</h3>
                  <button 
                    className="change-school-button"
                    onClick={() => {
                      setNeedsSchoolSelection(true);
                    }}
                    style={{
                      padding: '8px 16px',
                      fontSize: '0.9em'
                    }}
                  >
                    Manage Schools
                  </button>
                </div>
                <div className="setting-item">
                  <div className="setting-content">
                    <div className="setting-label-group">
                      <label className="setting-label">
                        Selected Schools ({user?.schools?.length || 0})
                      </label>
                      {user?.schools && user.schools.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '8px' }}>
                          {user.schools.map((school) => (
                            <div 
                              key={school.id}
                              style={{
                                padding: '12px',
                                background: '#f5f5f5',
                                borderRadius: '8px',
                                border: '1px solid #e0e0e0'
                              }}
                            >
                              <p className="setting-description" style={{ marginBottom: '4px' }}>
                                <strong>{school.name}</strong>
                              </p>
                              {school.location && (
                                <p className="setting-description" style={{ 
                                  color: '#666',
                                  fontSize: '0.85em',
                                  marginTop: '2px'
                                }}>
                                  📍 {school.location}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="setting-description">
                          <strong>None</strong>
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>


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
                  <p><strong>Your School Assistant</strong></p>
                  <p>I help you find school events by searching multiple places at once:</p>
                  <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
                    <li>📧 Your Gmail inbox for school emails</li>
                    <li>📚 Our local database of curated events</li>
                    <li>🌐 The web for the newest updates</li>
                    <li>⚡ Live updates as you search</li>
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