import BookmarksContainer from './components/BookmarksContainer';
import ComparisonInsights from './components/ComparisonInsights';
import CopyToast from './components/CopyToast';
import BookmarkToast from './components/BookmarkToast';
import Header from './components/Header';
import WelcomeMessage from './components/WelcomeMessage';
import CopySuccessIcon from './components/CopySuccessIcon';
import CopyIcon from './components/CopyIcon';
import EventPopup from './components/EventPopup';
import SchoolSelection from './components/SchoolSelection';
import Login from './components/Login';
import SplashScreen from './components/SplashScreen';
import ChildrenInput from './components/ChildrenInput';
import ErrorScreen from './components/ErrorScreen';
import { User, Bot, Calendar, Clock, Database, Mail, Bookmark, Tent, Target, Drama, Activity, Palette, BookOpen, Users, DollarSign, Settings, MessageSquare, Trash2 } from 'lucide-react';
import SettingsContainer from './components/SettingsContainer';
import ComparisonResults from './components/ComparisonResults';
import ComparisonHeader from './components/ComparisonHeader';
import ChildTabContent from './components/ChildTabContent';
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import logo from './assets/logo.png';
import {
  formatResponseTime,
  formatResponseText
} from './utils';
// ...existing imports...

// Determine API URL based on LOCAL_MODE flag
const isLocalMode = process.env.REACT_APP_LOCAL_MODE === 'true';
const API_BASE_URL = isLocalMode 
  ? 'http://localhost:8000' 
  : (process.env.REACT_APP_API_URL || 'https://school-assistant-production.up.railway.app');

function App() {
  // Authentication state
  const [user, setUser] = useState(null); // Current logged-in user
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [needsSchoolSelection, setNeedsSchoolSelection] = useState(false);
  const [needsChildrenInput, setNeedsChildrenInput] = useState(false);
  
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('chat'); // 'events', 'chat', 'comparison', or 'settings'
  const [comparisonResults, setComparisonResults] = useState({
    original: null,
    naive: null
  });
  const [isRunningComparison, setIsRunningComparison] = useState(false);
  const messagesEndRef = useRef(null);
  const [events, setEvents] = useState([]);
  const [isLoadingEvents, setIsLoadingEvents] = useState(true);
  const [selectedEventDetails, setSelectedEventDetails] = useState(null);
  const [showSplash, setShowSplash] = useState(true);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'online', 'offline'
  const [useWebSocket, setUseWebSocket] = useState(true); // Toggle for WebSocket vs HTTP
  const wsRef = useRef(null); // WebSocket connection reference
  const [selectedSchoolDistrict, setSelectedSchoolDistrict] = useState(null); // Selected school from localStorage
  const [showSchoolSelection, setShowSchoolSelection] = useState(false); // Show school selection screen (LEGACY - using auth system now)
  const [copiedMessageIndex, setCopiedMessageIndex] = useState(null); // Track which message was copied
  const [showCopyToast, setShowCopyToast] = useState(false); // Show copy success toast
  const [bookmarks, setBookmarks] = useState([]); // Store bookmarked messages
  const [showBookmarkToast, setShowBookmarkToast] = useState(false); // Show bookmark toast
  const [bookmarkToastMessage, setBookmarkToastMessage] = useState(''); // Toast message text
  const [showGmailDisconnectConfirm, setShowGmailDisconnectConfirm] = useState(false); // Show Gmail disconnect confirmation
  const [showGmailDisconnectSuccess, setShowGmailDisconnectSuccess] = useState(false); // Show Gmail disconnect success message
  const [childToDelete, setChildToDelete] = useState(null); // Track which child is being deleted for inline confirmation
  const [currentAgentUpdate, setCurrentAgentUpdate] = useState(null); // Track current agent processing updates
  const [studentReports, setStudentReports] = useState({}); // Store student reports by child name: {childName: reports[]}
  const [loadingReports, setLoadingReports] = useState({}); // Track loading state per child: {childName: boolean}
  const [expandedReports, setExpandedReports] = useState({}); // Track which report email contents are expanded: {reportId: boolean}
  const [attendanceData, setAttendanceData] = useState({}); // Store attendance emails by child id
  const [loadingAttendance, setLoadingAttendance] = useState({});

  // Helper function to remove [Source: ...] prefix from content
  const cleanContent = (content) => {
    if (!content) return content;
    // Remove [Source: ...] tag if present at the beginning
    return content.replace(/^\[Source:\s*[^\]]+\]\s*/i, '').trim();
  };

  // Modular WebSocket message handler
  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'status':
        // Status messages are for logging/debugging only, don't display as chat bubbles
        console.log('📊 Status:', data.content);
        break;
      case 'final': {
        // Final combined message from backend - already formatted with all agent results
        const cleanedContent = cleanContent(data.content);
        console.log(`🎯 FINAL: Received pre-combined message from backend`);
        console.log(`   Agent: ${data.agent}`);
        console.log(`   Content length: ${cleanedContent.length} chars`);
        console.log(`   Result count:`, data.result_count);
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: cleanedContent,
          source: data.agent || 'Combined Results',
          tool: data.tool,
          responseTime: data.response_time || null,
          evaluation: null,
          resultCount: data.result_count || 0
        }]);
        setIsLoading(false);
        setCurrentAgentUpdate(null); // Clear agent update when final response arrives
        break;
      }
      // Uncomment and handle evaluation type if needed in future
      // case 'evaluation': {
      //   // Evaluation message received - update the last assistant message
      //   console.log('📊 Evaluation received from WebSocket:', data.evaluation);
      //   setMessages(prev => {
      //     const newMessages = [...prev];
      //     // Find the last assistant message and add evaluation
      //     for (let i = newMessages.length - 1; i >= 0; i--) {
      //       if (newMessages[i].type === 'assistant') {
      //         newMessages[i] = {
      //           ...newMessages[i],
      //           evaluation: data.evaluation
      //         };
      //         break;
      //       }
      //     }
      //     return newMessages;
      //   });
      //   break;
      // }
      case 'update':
        // Update/progress messages from backend - show in UI
        console.log('🔄 Update:', data.content, 'from', data.agent);
        // Map agent names to user-friendly messages
        const agentMessages = {
          'Gmail': 'Searching your email...',
          'Web Agent': 'Searching school websites...',
          'Vector Store': 'Searching event database...',
          'Combined Results': 'Combining results...'
        };
        setCurrentAgentUpdate({
          agent: agentMessages[data.agent] || 'Processing your request...'
        });
        break;
      case 'error':
        setError(`❌ ${data.content}`);
        setIsLoading(false);
        break;
      default:
        console.warn('Unknown WebSocket message type:', data.type);
    }
  };

  // Check for existing session on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        const parsedUser = JSON.parse(savedUser);
        
        // Refresh user data from backend to get latest children/schools
        const refreshUserData = async () => {
          try {
            const response = await axios.post(`${API_BASE_URL}/api/auth/login`, {
              email: parsedUser.email
            });
            if (response.data.success && response.data.user) {
              const freshUser = response.data.user;
              // Preserve Gmail info from localStorage if not in response
              if (parsedUser.gmail_email && !freshUser.gmail_email) {
                freshUser.gmail_email = parsedUser.gmail_email;
                freshUser.gmail_name = parsedUser.gmail_name;
                freshUser.gmail_connected_at = parsedUser.gmail_connected_at;
              }
              setUser(freshUser);
              localStorage.setItem('user', JSON.stringify(freshUser));
              
              // Check if user needs to add children
              if (!freshUser.children || freshUser.children.length === 0) {
                setNeedsChildrenInput(true);
              } else if (!freshUser.schools || freshUser.schools.length === 0) {
                setNeedsSchoolSelection(true);
              }
            }
          } catch (err) {
            console.error('Error refreshing user data:', err);
            // Fall back to localStorage data
            setUser(parsedUser);
          }
        };
        
        setUser(parsedUser);
        setIsAuthenticated(true);
        refreshUserData(); // Refresh in background
        
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
        userData.gmail_name = gmailStatus.data.gmail_name;
        userData.gmail_connected_at = gmailStatus.data.connected_at;
      }
    } catch (error) {
      console.error('Error fetching Gmail status:', error);
    }
    
    setUser(userData);
    setIsAuthenticated(true);
    localStorage.setItem('user', JSON.stringify(userData));
    
    // Check if user needs to add children (no children added yet)
    if (!userData.children || userData.children.length === 0) {
      setNeedsChildrenInput(true);
    }
    // Check if user needs to select schools (no schools selected yet)
    else if (!userData.schools || userData.schools.length === 0) {
      setNeedsSchoolSelection(true);
    }
  };

  // Handle children added
  const handleChildrenAdded = (children) => {
    const formattedChildren = children.map(child => ({
      child_id: child.child_id || child.id,
      child_name: child.child_name || child.name || child,
      child_grade: child.child_grade || child.grade || null,
      child_school: child.child_school || child.school_name || null
    }));

    const existingChildren = user?.children || [];
    const mergedChildren = [...existingChildren];

    formattedChildren.forEach(newChild => {
      if (!mergedChildren.some(child => (child.child_id || child.id) === newChild.child_id)) {
        mergedChildren.push(newChild);
      }
    });

    const updatedUser = { ...user, children: mergedChildren };
    setUser(updatedUser);
    localStorage.setItem('user', JSON.stringify(updatedUser));
    setNeedsChildrenInput(false);
    
    // Move to school selection if needed
    if (!updatedUser.schools || updatedUser.schools.length === 0) {
      setNeedsSchoolSelection(true);
    }
  };
  
  // Handle skip children input
  const handleSkipChildren = () => {
    setNeedsChildrenInput(false);
    
    // Move to school selection if needed
    if (!user.schools || user.schools.length === 0) {
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
    setNeedsChildrenInput(false);
    setNeedsSchoolSelection(false);
    setMessages([]);
    setBookmarks([]);
    
    // Clear all localStorage
    localStorage.clear();
    
    // Switch to chat tab
    setActiveTab('chat');
    
    console.log('User logged out, localStorage cleared');
  };
  
  // Disconnect Gmail account
  const handleDisconnectGmail = async () => {
    // Show confirmation dialog
    setShowGmailDisconnectConfirm(true);
  };
  
  // Confirm Gmail disconnect
  const confirmDisconnectGmail = async () => {
    setShowGmailDisconnectConfirm(false);
    
    try {
      await axios.post(`${API_BASE_URL}/api/auth/gmail/disconnect`, {
        email: user.email
      });
      
      // Update user object to remove Gmail info
      setUser({
        ...user,
        gmail_email: null,
        gmail_connected_at: null,
        gmail_name: null
      });
      
      // Show success message
      setShowGmailDisconnectSuccess(true);
      
      // Hide success message after 5 seconds
      setTimeout(() => {
        setShowGmailDisconnectSuccess(false);
      }, 5000);
    } catch (error) {
      console.error('Error disconnecting Gmail:', error);
      alert('Failed to disconnect Gmail. Please try again.');
    }
  };
  
  // Cancel Gmail disconnect
  const cancelDisconnectGmail = () => {
    setShowGmailDisconnectConfirm(false);
  };

  // Show delete child confirmation
  const handleDeleteChild = (child) => {
    setChildToDelete(child);
  };

  // Cancel delete child
  const cancelDeleteChild = () => {
    setChildToDelete(null);
  };

  // Confirm delete child
  const confirmDeleteChild = async () => {
    if (!childToDelete) return;

    const childId = childToDelete.child_id || childToDelete.id;

    try {
      await axios.delete(`${API_BASE_URL}/api/auth/children/${childId}`, {
        params: { email: user.email }
      });

      const updatedChildren = (user.children || []).filter(
        child => (child.child_id || child.id) !== childId
      );

      const updatedUser = { ...user, children: updatedChildren };
      setUser(updatedUser);
      localStorage.setItem('user', JSON.stringify(updatedUser));
      setChildToDelete(null);
    } catch (error) {
      console.error('Error deleting child:', error);
      const detail = error?.response?.data?.detail || 'Failed to delete child. Please try again.';
      alert(detail);
      setChildToDelete(null);
    }
  };

  const handleUpdateChild = async (childId, updates) => {
    try {
      const response = await axios.put(`${API_BASE_URL}/api/auth/children/${childId}`, {
        email: user.email,
        ...updates
      });

      if (response.data?.child) {
        const updatedChild = response.data.child;
        const updatedChildren = (user.children || []).map(child =>
          (child.child_id || child.id) === childId
            ? {
                child_id: updatedChild.child_id || updatedChild.id || childId,
                child_name: updatedChild.child_name,
                child_grade: updatedChild.child_grade,
                child_school: updatedChild.child_school,
                child_age: updatedChild.child_age ?? null
              }
            : child
        );

        const updatedUser = { ...user, children: updatedChildren };
        setUser(updatedUser);
        localStorage.setItem('user', JSON.stringify(updatedUser));
      }

      return { success: true };
    } catch (error) {
      console.error('Error updating child:', error);
      const detail = error?.response?.data?.detail || 'Failed to update child.';
      return { success: false, message: detail };
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
      const response = await axios.post(`${API_BASE_URL}/api/bookmarks/add`, {
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
      const response = await axios.post(`${API_BASE_URL}/api/bookmarks/remove`, {
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
      const response = await axios.get(`${API_BASE_URL}/api/bookmarks/${user.email}`);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);
  

  
  // Check backend health on mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`, {
          timeout: 15000 // Increased timeout for backend startup
        });
        
        if (response.status === 200) {
          setBackendStatus('online');
          console.log('✅ Backend is online');
          
          // Hide splash screen after 1 second if backend is online
          setTimeout(() => {
            setShowSplash(false);
          }, 1000);
        }
      } catch (err) {
        console.warn('⚠️ Backend health check failed, but continuing:', err.message);
        setBackendStatus('offline');
        // Still hide splash screen so app can load
        setTimeout(() => {
          setShowSplash(false);
        }, 2000);
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

  // Fetch student reports for a specific child
  const fetchStudentReports = async (userEmail, childName) => {
    try {
      console.log('Fetching student reports for:', childName);
      setLoadingReports(prev => ({ ...prev, [childName]: true }));
      
      const response = await axios.get(
        `${API_BASE_URL}/student/reports/${encodeURIComponent(userEmail)}/${encodeURIComponent(childName)}`,
        { timeout: 300000 } // 5 minutes timeout for LLM processing
      );
      
      if (response.data.message) {
        console.log(`ℹ️ ${response.data.message}`);
      }
      
      if (response.data.reports && response.data.reports.length > 0) {
        console.log(`✅ Loaded ${response.data.reports.length} student reports for ${childName}`);
        setStudentReports(prev => ({ 
          ...prev, 
          [childName]: {
            all: response.data.reports,
            byYear: response.data.reports_by_year || {},
            years: response.data.years || []
          }
        }));
        return response.data.reports;
      } else {
        console.log(`ℹ️ No student reports found for ${childName}`);
        setStudentReports(prev => ({ ...prev, [childName]: { all: [], byYear: {}, years: [] } }));
        return [];
      }
    } catch (err) {
      if (err.code === 'ECONNABORTED' || err.name === 'CanceledError') {
        console.warn(`⚠️ Student reports fetch timed out for ${childName}`);
      } else {
        console.warn(`⚠️ Could not fetch student reports for ${childName}:`, err.message);
      }
      setStudentReports(prev => ({ ...prev, [childName]: { all: [], byYear: {}, years: [] } }));
      return [];
    } finally {
      setLoadingReports(prev => ({ ...prev, [childName]: false }));
    }
  };

  // Fetch attendance emails for a specific child (by child id)
  const fetchChildAttendance = async (userEmail, childId, maxResults = 10) => {
    try {
      console.log('Fetching attendance emails for child id:', childId);
      setLoadingAttendance(prev => ({ ...prev, [childId]: true }));

      const response = await axios.get(
        `${API_BASE_URL}/api/auth/children/${encodeURIComponent(childId)}/attendance-search`,
        {
          params: { email: userEmail, max_results: maxResults },
          timeout: 120000
        }
      );

      if (response.data && response.data.attendance_emails) {
        // Store both the flat list and the grouped by year data
        setAttendanceData(prev => ({ 
          ...prev, 
          [childId]: {
            all: response.data.attendance_emails,
            byYear: response.data.attendance_by_year || {},
            years: response.data.years || []
          }
        }));
        return response.data.attendance_emails;
      } else {
        setAttendanceData(prev => ({ ...prev, [childId]: { all: [], byYear: {}, years: [] } }));
        return [];
      }
    } catch (err) {
      console.warn('⚠️ Could not fetch attendance emails for child:', childId, err.message || err);
      setAttendanceData(prev => ({ ...prev, [childId]: { all: [], byYear: {}, years: [] } }));
      return [];
    } finally {
      setLoadingAttendance(prev => ({ ...prev, [childId]: false }));
    }
  };

  // Fetch email events for the logged-in user
  const fetchEmailEvents = async (userEmail) => {
    try {
      console.log('Fetching email events for:', userEmail);
      
      const response = await axios.get(
        `${API_BASE_URL}/events/email/${encodeURIComponent(userEmail)}`,
        { 
          timeout: 15000 // 15 second timeout
        }
      );
      
      if (response.data.message) {
        console.log(`ℹ️ ${response.data.message}`);
      }
      
      if (response.data.events && response.data.events.length > 0) {
        console.log(`✅ Loaded ${response.data.events.length} events from email`);
        return response.data.events;
      } else {
        console.log('ℹ️ No email events found. Make sure you have event-related emails in your inbox.');
        return [];
      }
    } catch (err) {
      if (err.code === 'ECONNABORTED' || err.name === 'CanceledError') {
        console.warn('⚠️ Email event fetch timed out - continuing without events');
      } else if (err.code === 'ERR_CANCELED') {
        console.warn('⚠️ Email event fetch was cancelled');
      } else {
        console.warn('⚠️ Could not fetch email events:', err.message);
      }
      return [];
    }
  };

  // Fetch events on component mount (non-blocking)
  useEffect(() => {
    // Don't block app startup - fetch events in background
    if (!user || !user.email) {
      console.log('ℹ️ No user logged in, skipping email events fetch');
      setIsLoadingEvents(false);
      return;
    }

    const fetchEvents = async () => {
      try {
        setIsLoadingEvents(true);
        console.log('📧 Fetching email events in background...');
        
        // Fetch email events asynchronously without blocking
        const emailEvents = await fetchEmailEvents(user.email);
        
        if (emailEvents && emailEvents.length > 0) {
          console.log(`✅ Loaded ${emailEvents.length} events from email`);
          setEvents(emailEvents);
        } else {
          console.log('ℹ️ No events found in email');
          setEvents([]);
        }
      } catch (err) {
        console.error('Error fetching events:', err);
        setEvents([]); // Set empty array on error
      } finally {
        setIsLoadingEvents(false);
      }
    };

    // Use setTimeout to ensure app renders first, then fetch events
    const timeoutId = setTimeout(() => {
      fetchEvents();
    }, 100); // Small delay to let UI render first

    return () => clearTimeout(timeoutId);
  }, [user]); // Re-fetch when user changes



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
      const wsUrl = API_BASE_URL.replace('https://', 'wss://').replace('http://', 'ws://');
      const ws = new WebSocket(`${wsUrl}/ws/multi-agent-stream`);
      
      ws.onopen = () => {
        console.log('🔌 WebSocket connected');
        
        // Extract email suffixes and district names from all selected schools
        const email_suffixes = user?.schools?.map(s => s.email_suffix).filter(Boolean) || [];
        const school_districts = user?.schools?.map(s => s.name).filter(Boolean) || [];
        const school_websites = user?.schools?.map(s => s.website).filter(Boolean) || [];
        
        console.log('📧 WebSocket sending query with school data:', {
          schools_count: user?.schools?.length || 0,
          email_suffixes,
          school_districts,
          school_websites
        });
        
        // Send the question with email suffixes, school districts, websites, and user email
        console.log('📧 Sending WebSocket message with user email:', user?.email);
        console.log('📧 Full user object:', user);
        
        ws.send(JSON.stringify({ 
          question: userMessage,
          user_email: user?.email,  // For per-user Gmail authentication
          email_suffixes: email_suffixes.length > 0 ? email_suffixes : null,
          school_districts: school_districts.length > 0 ? school_districts : null,
          school_websites: school_websites.length > 0 ? school_websites : null,
          // Legacy fields for backwards compatibility
          email_suffix: selectedSchoolDistrict?.email_suffix || (email_suffixes.length > 0 ? email_suffixes[0] : null),
          school_district: selectedSchoolDistrict?.district || (school_districts.length > 0 ? school_districts[0] : null)
        }));
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('📨 WebSocket message:', data);
        handleWebSocketMessage(data);
      };
      
      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setError('❌ WebSocket connection error');
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
      const school_websites = user?.schools?.map(s => s.website).filter(Boolean) || [];
      
      console.log('📧 Sending query with school data:', {
        schools_count: user?.schools?.length || 0,
        email_suffixes,
        school_districts,
        school_websites
      });
      
      const response = await axios.post(`${API_BASE_URL}/multi-agent-query`, {
        question: userMessage,
        email_suffixes: email_suffixes.length > 0 ? email_suffixes : null,
        school_districts: school_districts.length > 0 ? school_districts : null,
        school_websites: school_websites.length > 0 ? school_websites : null,
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
        errorMessage = `❌ Backend server is not running. Please check the connection to ${API_BASE_URL}`;
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

  // Error screen if backend is offline
  if (backendStatus === 'offline') {
    return <ErrorScreen />;
  }

  // Splash screen
  if (showSplash) {
    return <SplashScreen backendStatus={backendStatus} />;
  }

  // Show login if not authenticated
  if (!isAuthenticated) {
    return <Login onLoginSuccess={handleLoginSuccess} />;
  }

  // Show children input if user hasn't added children yet
  if (needsChildrenInput) {
    return <ChildrenInput 
      email={user?.email} 
      onChildrenAdded={handleChildrenAdded} 
      onSkip={handleSkipChildren}
      existingChildren={user?.children || []}
    />;
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
      {showCopyToast && <CopyToast />}

      {/* Bookmark Toast Notification */}
      {showBookmarkToast && <BookmarkToast message={bookmarkToastMessage} />}

      <Header 
        logo={logo}
        selectedSchoolDistrict={selectedSchoolDistrict}
        setNeedsSchoolSelection={setNeedsSchoolSelection}
        isAuthenticated={isAuthenticated}
        needsSchoolSelection={needsSchoolSelection}
        user={user}
      />

      {/* Main content area with sidebar and content */}
      <div className="main-content">
        {/* Sidebar with tabs */}
        <div className="sidebar">
          <button 
            className={`tab ${activeTab === 'events' ? 'active' : ''}`}
            onClick={() => setActiveTab('events')}
          >
            <Calendar size={18} />&nbsp;&nbsp;Events
          </button>
          <button 
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <MessageSquare size={18} />&nbsp;&nbsp;Chat
          </button>
          <button 
            className={`tab ${activeTab === 'bookmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('bookmarks')}
          >
            <Bookmark size={18} />&nbsp;&nbsp;Bookmarks
          </button>
          {/* Render child tabs above settings tab */}
          {user && user.children && user.children.length > 0 && (() => {
            console.log('User object:', user);
            console.log('User.children:', user.children);
            return user.children.map((child, idx) => {
              const childName = child.child_name || child.name || (typeof child === 'string' ? child : `Child ${idx + 1}`);
              return (
                <button
                  key={`child-tab-${childName}-${idx}`}
                  className={`tab ${activeTab === `child-${childName}` ? 'active' : ''}`}
                  onClick={() => {
                    console.log('Clicked child tab:', childName);
                    setActiveTab(`child-${childName}`);
                  }}
                >
                  <Users size={18} />&nbsp;&nbsp;{childName}
                </button>
              );
            });
          })()}
          <button 
            className={`tab ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <Settings size={18} />&nbsp;&nbsp;Settings
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
                          {event.type.includes('Camp') ? <Tent size={24} /> : 
                           event.type.includes('Challenge') ? <Target size={24} /> : 
                           event.type.includes('Audition') ? <Drama size={24} /> :
                           event.type.includes('Clinic') ? <Activity size={24} /> :
                           event.type.includes('Art') ? <Palette size={24} /> : <BookOpen size={24} />}
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
                            <span className="detail-icon"><Users size={16} /></span>
                            <span>{event.target_audience}</span>
                          </div>
                        )}
                        <div className="event-details-row">
                          <div className="event-details-column">
                            {event.date && (
                              <div className="event-detail">
                                <span className="detail-icon">
                                  <Calendar size={16} />
                                </span>
                                <span>{event.date}</span>
                              </div>
                            )}
                            {event.cost && (
                              <div className="event-detail">
                                <span className="detail-icon"><DollarSign size={16} /></span>
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
            <WelcomeMessage />
          )}

          {messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <div className="message-icon">
                {message.type === 'user' ? (
                  <User size={24} className="message-icon-user" />
                ) : (
                  <Bot size={24} className="message-icon-assistant" />
                )}
              </div>
              <div className="message-content">
                <div className="message-text">
                  {message.type === 'assistant' 
                    ? formatResponseText(message.content)
                    : message.content
                  }
                </div>
                
                {/* All metadata in one line */}
                {message.type === 'assistant' && message.source && (
                  <div className="assistant-metadata-row">
                    {/* Left side: Total result count and Evaluation status */}
                    <div className="assistant-metadata-left">
                      {message.resultCount > 0 && (
                        <span className="assistant-metadata-results">
                          📊 Results: {message.resultCount}
                        </span>
                      )}
                      {message.evaluation && message.evaluation.status === 'completed' && (
                        <span className="assistant-metadata-evaluated">
                          ✓ Evaluated
                        </span>
                      )}
                    </div>
                    {/* Right side: Response Time, Copy, Bookmark */}
                    <div className="assistant-metadata-right">
                      {message.responseTime && (
                        <span className="assistant-metadata-time">
                          <Clock size={14} />
                          {formatResponseTime(message.responseTime)}
                        </span>
                      )}
                      <button 
                        className="copy-button"
                        onClick={() => copyToClipboard(message.content, index)}
                        title="Copy message"
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
                        className={`bookmark-button${isMessageBookmarked(message) ? ' bookmarked' : ''}`}
                        onClick={() => bookmarkMessage(message, index)}
                        title={isMessageBookmarked(message) ? "Already bookmarked" : "Bookmark message"}
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
                  <div className="user-message-row">
                    <button 
                      className="copy-button"
                      onClick={() => copyToClipboard(message.content, index)}
                      title="Copy message"
                    >
                      {copiedMessageIndex === index ? (
                        <CopySuccessIcon />
                      ) : (
                        <CopyIcon />
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
                {/* Evaluation failed messages are hidden */}
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
              } */}
            </div>
          ))}

          {isLoading && (
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
                  {currentAgentUpdate ? (
                    currentAgentUpdate.agent
                  ) : (
                    'Searching school events data...'
                  )}
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
                  placeholder="Ask about school events, programs, activities and more..."
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
            <div>
              <ComparisonHeader
                comparisonResults={comparisonResults}
                isRunningComparison={isRunningComparison}
                runComparison={runComparison}
                error={error}
              />
              {!isRunningComparison && comparisonResults.original && comparisonResults.naive && (
                <div>
                  <ComparisonResults comparisonResults={comparisonResults} />
                  <ComparisonInsights />
                </div>
              )}
            </div>
          )}

          {activeTab === 'bookmarks' && (
            <BookmarksContainer
              bookmarks={bookmarks}
              removeBookmark={removeBookmark}
              formatResponseText={formatResponseText}
              BookmarkIcon={Bookmark}
            />
          )}

          {/* Child tabs content */}
          <ChildTabContent
            user={user}
            activeTab={activeTab}
            studentReports={studentReports}
            loadingReports={loadingReports}
            expandedReports={expandedReports}
            setExpandedReports={setExpandedReports}
            fetchStudentReports={fetchStudentReports}
            attendanceData={attendanceData}
            loadingAttendance={loadingAttendance}
            fetchChildAttendance={fetchChildAttendance}
          />

          {activeTab === 'settings' && (
            <SettingsContainer
              user={user}
              handleLogout={handleLogout}
              showGmailDisconnectConfirm={showGmailDisconnectConfirm}
              showGmailDisconnectSuccess={showGmailDisconnectSuccess}
              confirmDisconnectGmail={confirmDisconnectGmail}
              cancelDisconnectGmail={cancelDisconnectGmail}
              handleDisconnectGmail={handleDisconnectGmail}
              setNeedsChildrenInput={setNeedsChildrenInput}
              setNeedsSchoolSelection={setNeedsSchoolSelection}
              handleDeleteChild={handleDeleteChild}
              handleUpdateChild={handleUpdateChild}
              childToDelete={childToDelete}
              cancelDeleteChild={cancelDeleteChild}
              confirmDeleteChild={confirmDeleteChild}
              useWebSocket={useWebSocket}
              setUseWebSocket={setUseWebSocket}
              GmailIcon={Mail}
              DatabaseIcon={Database}
            />
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