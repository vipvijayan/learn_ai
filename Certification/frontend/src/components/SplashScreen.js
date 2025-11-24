import React from 'react';
import logo from '../assets/logo.png'; // Adjust path if needed

const SplashScreen = ({ backendStatus }) => (
  <div className="splash-screen">
    <div className="splash-content">
      <img src={logo} alt="School Assistant" className="splash-icon" />
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

export default SplashScreen;
