import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const Login = ({ onLoginSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGoogleLogin = async () => {
    setError('');
    setLoading(true);

    try {
      // Get OAuth authorization URL (use placeholder email in state)
      const tempEmail = `login_${Date.now()}@temp.local`;
      const response = await axios.get(`${API_BASE_URL}/api/auth/gmail/authorize`, {
        params: { email: tempEmail }
      });

      // Open OAuth popup
      const authUrl = response.data.authorization_url;
      const width = 600;
      const height = 700;
      const left = window.screen.width / 2 - width / 2;
      const top = window.screen.height / 2 - height / 2;
      
      const popup = window.open(
        authUrl, 
        'Google Sign In',
        `width=${width},height=${height},left=${left},top=${top}`
      );

      // Listen for message from OAuth callback
      const handleMessage = async (event) => {
        // Check if message is from our OAuth callback
        if (event.data.type === 'gmail_oauth_success') {
          const gmailEmail = event.data.email;
          
          try {
            // Login with the Gmail email
            const loginResponse = await axios.post(`${API_BASE_URL}/api/auth/login`, {
              email: gmailEmail
            });

            if (loginResponse.data.success) {
              // Get Gmail status to include in user data
              const statusResponse = await axios.get(`${API_BASE_URL}/api/auth/gmail/status`, {
                params: { email: gmailEmail }
              });
              
              // Add Gmail info to user data
              const userData = {
                ...loginResponse.data.user,
                gmail_email: statusResponse.data.connected ? gmailEmail : null,
                gmail_connected_at: statusResponse.data.connected_at
              };
              
              onLoginSuccess(userData);
            } else {
              setError('Failed to login. Please try again.');
              setLoading(false);
            }
          } catch (err) {
            console.error('Post-OAuth login error:', err);
            setError('Failed to complete login. Please try again.');
            setLoading(false);
          }
          
          // Clean up event listener
          window.removeEventListener('message', handleMessage);
        }
      };

      // Add event listener for postMessage from popup
      window.addEventListener('message', handleMessage);

      // Also check if popup was closed without completing auth
      const checkPopup = setInterval(() => {
        if (popup && popup.closed) {
          clearInterval(checkPopup);
          // If still loading, user closed popup manually
          setTimeout(() => {
            if (loading) {
              setError('Sign in was cancelled.');
              setLoading(false);
              window.removeEventListener('message', handleMessage);
            }
          }, 1000);
        }
      }, 500);
      
    } catch (err) {
      console.error('Login error:', err);
      setError(err.response?.data?.detail || 'Failed to initiate sign in. Please try again.');
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        padding: '48px',
        maxWidth: '440px',
        width: '100%'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{
            fontSize: '2em',
            fontWeight: '700',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '8px'
          }}>
            🏫 School Assistant
          </h1>
          <p style={{ color: '#666', fontSize: '0.95em' }}>
            Sign in with your Gmail account to get started
          </p>
        </div>

        {error && (
          <div style={{
            background: '#fee',
            color: '#c33',
            padding: '12px 16px',
            borderRadius: '8px',
            marginBottom: '20px',
            fontSize: '0.9em',
            border: '1px solid #fcc'
          }}>
            {error}
          </div>
        )}

        <button
          onClick={handleGoogleLogin}
          disabled={loading}
          style={{
            width: '100%',
            padding: '16px',
            fontSize: '1.05em',
            fontWeight: '600',
            color: loading ? '#666' : '#333',
            background: loading ? '#f5f5f5' : 'white',
            border: '2px solid #e0e0e0',
            borderRadius: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px'
          }}
          onMouseOver={(e) => {
            if (!loading) {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
              e.target.style.borderColor = '#667eea';
            }
          }}
          onMouseOut={(e) => {
            e.target.style.transform = 'translateY(0)';
            e.target.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
            e.target.style.borderColor = '#e0e0e0';
          }}
        >
          {loading ? (
            <>
              <span style={{ 
                display: 'inline-block',
                animation: 'spin 1s linear infinite'
              }}>
                ⏳
              </span>
              <span>Signing in with Google...</span>
            </>
          ) : (
            <>
              <svg width="20" height="20" viewBox="0 0 20 20">
                <path fill="#4285F4" d="M19.6 10.23c0-.82-.1-1.42-.25-2.05H10v3.72h5.5c-.15.96-.74 2.31-2.04 3.22v2.45h3.16c1.89-1.73 2.98-4.3 2.98-7.34z"/>
                <path fill="#34A853" d="M13.46 15.13c-.83.59-1.96 1-3.46 1-2.64 0-4.88-1.74-5.68-4.15H1.07v2.52C2.72 17.75 6.09 20 10 20c2.7 0 4.96-.89 6.62-2.42l-3.16-2.45z"/>
                <path fill="#FBBC05" d="M3.99 10c0-.69.12-1.35.32-1.97V5.51H1.07A9.973 9.973 0 000 10c0 1.61.39 3.14 1.07 4.49l3.24-2.52c-.2-.62-.32-1.28-.32-1.97z"/>
                <path fill="#EA4335" d="M10 3.88c1.88 0 3.13.81 3.85 1.48l2.84-2.76C14.96.99 12.7 0 10 0 6.09 0 2.72 2.25 1.07 5.51l3.24 2.52C5.12 5.62 7.36 3.88 10 3.88z"/>
              </svg>
              <span>Sign in with Google</span>
            </>
          )}
        </button>

        <div style={{
          marginTop: '24px',
          paddingTop: '24px',
          borderTop: '1px solid #eee',
          textAlign: 'center',
          color: '#999',
          fontSize: '0.85em'
        }}>
          <p style={{ margin: 0 }}>
            Your email is only used for authentication and personalization
          </p>
        </div>
      </div>

      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default Login;
