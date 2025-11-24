import React from 'react';

const ErrorScreen = () => {
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
};

export default ErrorScreen;
