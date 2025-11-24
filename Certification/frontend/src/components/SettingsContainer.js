import React from 'react';
import { User, LogOut, CheckCircle, AlertTriangle, Database, Radio, Wifi, Bot, MapPin, Trash2 } from 'lucide-react';

const SettingsContainer = ({
  user,
  handleLogout,
  showGmailDisconnectConfirm,
  showGmailDisconnectSuccess,
  confirmDisconnectGmail,
  cancelDisconnectGmail,
  handleDisconnectGmail,
  setNeedsChildrenInput,
  setNeedsSchoolSelection,
  handleDeleteChild,
  childToDelete,
  cancelDeleteChild,
  confirmDeleteChild,
  useWebSocket,
  setUseWebSocket,
  GmailIcon,
  DatabaseIcon
}) => (
  <div className="settings-container">
    <div className="settings-section">
      <div className="settings-section-header">
        <h3><User size={20} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }} /> Account</h3>
        <button 
          className="logout-button"
          onClick={handleLogout}
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
          <LogOut size={16} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
          Logout
        </button>
      </div>
      {/* Gmail Account Status */}
      <div className="setting-item">
        <div className="setting-content">
          <div className="setting-label-group">
            <label className="setting-label">Gmail Account</label>
            {user?.gmail_email ? (
              <div className="gmail-connected-status">
                {user.gmail_name && (
                  <p className="gmail-connected-name">
                    <CheckCircle size={16} style={{ display: 'inline', marginRight: '4px', color: '#4caf50', verticalAlign: 'middle' }} /> <strong>{user.gmail_name}</strong>
                  </p>
                )}
                <p className="gmail-connected-email">
                  {user.gmail_name ? 'Email: ' : <><CheckCircle size={16} style={{ display: 'inline', marginRight: '4px', color: '#4caf50', verticalAlign: 'middle' }} /> Signed in as: </>}<strong>{user.gmail_email}</strong>
                </p>
                {user.gmail_connected_at && (
                  <p className="gmail-connected-date">
                    Connected on {new Date(user.gmail_connected_at).toLocaleDateString()}
                  </p>
                )}
                <p className="gmail-connected-description">
                  Your Gmail is connected and can be searched for school-related emails.
                </p>
                {/* Disconnect Confirmation */}
                {showGmailDisconnectConfirm && (
                  <div className="gmail-disconnect-confirm">
                    <p className="gmail-disconnect-warning"><AlertTriangle size={18} style={{ display: 'inline', marginRight: '4px', color: '#ff9800', verticalAlign: 'middle' }} /> Are you sure?</p>
                    <p className="gmail-disconnect-message">
                      Disconnecting will remove access to your Gmail for searching school-related emails.
                    </p>
                    <div className="gmail-disconnect-buttons">
                      <button
                        onClick={confirmDisconnectGmail}
                        className="gmail-disconnect-yes"
                        onMouseEnter={(e) => e.target.style.background = '#d32f2f'}
                        onMouseLeave={(e) => e.target.style.background = '#f44336'}
                      >
                        Yes, Disconnect
                      </button>
                      <button
                        onClick={cancelDisconnectGmail}
                        className="gmail-disconnect-cancel"
                        onMouseEnter={(e) => {
                          e.target.style.background = '#f5f5f5';
                          e.target.style.borderColor = '#999';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = '#fff';
                          e.target.style.borderColor = '#ddd';
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
                {/* Disconnect Button - Only show if not confirming */}
                {!showGmailDisconnectConfirm && (
                  <button
                    onClick={handleDisconnectGmail}
                    className="gmail-disconnect-button"
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
                )}
              </div>
            ) : showGmailDisconnectSuccess ? (
              <div className="gmail-disconnected-success">
                <p className="gmail-disconnected-title"><CheckCircle size={18} style={{ display: 'inline', marginRight: '4px', color: '#4caf50', verticalAlign: 'middle' }} /> Gmail Disconnected Successfully</p>
                <p className="gmail-disconnected-message">
                  Your Gmail account has been disconnected. Sign in again to reconnect.
                </p>
              </div>
            ) : (
              <p className="setting-description gmail-not-connected">
                <AlertTriangle size={18} style={{ display: 'inline', marginRight: '4px', color: '#ff9800', verticalAlign: 'middle' }} /> Gmail not connected. Please sign out and sign in again to connect Gmail.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
    <div className="settings-section">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ margin: 0 }}><User size={20} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }} /> Children</h3>
        <button 
          className="change-school-button"
          onClick={() => setNeedsChildrenInput(true)}
          style={{ padding: '8px 16px', fontSize: '0.9em' }}
        >
          Manage Children
        </button>
      </div>
      <div className="setting-item">
        <div className="setting-content">
          <div className="setting-label-group">
            {user?.children && user.children.length > 0 ? (
              <>
                <label className="setting-label">
                  Your Children ({user.children.length})
                </label>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
                  {user.children.map((child, index) => (
                  <div 
                    key={index}
                    style={{ 
                      padding: '10px 14px', 
                      background: '#f5f5f5', 
                      borderRadius: '8px', 
                      border: '1px solid #e0e0e0'
                    }}
                  >
                    {childToDelete === child ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <p style={{ margin: 0, color: '#ff9800', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <AlertTriangle size={18} /> Are you sure you want to remove {child}?
                        </p>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          <button
                            onClick={confirmDeleteChild}
                            style={{
                              padding: '6px 12px',
                              background: '#f44336',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '0.9em',
                              fontWeight: '500',
                              transition: 'all 0.2s'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.background = '#d32f2f';
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.background = '#f44336';
                            }}
                          >
                            Yes, Remove
                          </button>
                          <button
                            onClick={cancelDeleteChild}
                            style={{
                              padding: '6px 12px',
                              background: '#fff',
                              color: '#666',
                              border: '1px solid #ddd',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '0.9em',
                              fontWeight: '500',
                              transition: 'all 0.2s'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.background = '#f5f5f5';
                              e.currentTarget.style.borderColor = '#999';
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.background = '#fff';
                              e.currentTarget.style.borderColor = '#ddd';
                            }}
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <p className="setting-description" style={{ margin: 0 }}>
                          <strong>{child}</strong>
                        </p>
                        <button
                          onClick={() => handleDeleteChild(child)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            cursor: 'pointer',
                            padding: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            color: '#f44336',
                            transition: 'all 0.2s'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = '#d32f2f';
                            e.currentTarget.style.transform = 'scale(1.1)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.color = '#f44336';
                            e.currentTarget.style.transform = 'scale(1)';
                          }}
                          title="Delete child"
                        >
                          <Trash2 size={18} />
                        </button>
                      </div>
                    )}
                  </div>
                  ))}
                </div>
              </>
            ) : (
              <>
                <p className="setting-description" style={{ color: '#666', fontStyle: 'italic' }}>
                  You haven't added any children yet. Use the "Manage Children" button above to enter their details.
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
    <div className="settings-section">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ margin: 0 }}><Database size={20} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }} /> School Settings</h3>
        <button 
          className="change-school-button"
          onClick={() => setNeedsSchoolSelection(true)}
          style={{ padding: '8px 16px', fontSize: '0.9em' }}
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
                    style={{ padding: '12px', background: '#f5f5f5', borderRadius: '8px', border: '1px solid #e0e0e0' }}
                  >
                    <p className="setting-description" style={{ marginBottom: '4px' }}>
                      <strong>{school.name}</strong>
                    </p>
                    {school.location && (
                      <p className="setting-description" style={{ color: '#666', fontSize: '0.85em', marginTop: '2px', marginBottom: '2px' }}>
                        <MapPin size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> {school.location}
                      </p>
                    )}
                    {school.website && (
                      <p className="setting-description" style={{ color: '#666', fontSize: '0.85em', marginTop: '2px' }}>
                        <Radio size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> <a href={school.website} target="_blank" rel="noopener noreferrer" style={{ color: '#1976d2', textDecoration: 'none' }}>
                          {school.website.replace(/^https?:\/\/(www\.)?/, '')}
                        </a>
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
      <h3><Wifi size={20} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }} /> Connection Settings</h3>
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
        {useWebSocket ? (
          <div className="setting-status">
            <span className="status-indicator active"></span>
            <span className="status-text">Live updates enabled</span>
          </div>
        ) : (
          <div className="setting-status">
            <span className="status-indicator inactive"></span>
            <span className="status-text">Using standard HTTP requests</span>
          </div>
        )}
      </div>
    </div>
    <div className="settings-section">
      <h3><Bot size={20} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }} /> About</h3>
      <div className="about-info">
        <p><strong>Your School Assistant</strong></p>
        <p>I help you find school events by searching multiple places at once:</p>
        <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
          <li style={{display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px'}}>
            <GmailIcon size={16} />
            Your Gmail inbox for school emails
          </li>
          <li style={{display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px'}}>
            <DatabaseIcon size={16} />
            Our local database of curated events
          </li>
          <li style={{display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px'}}><Radio size={16} /> The web for the newest updates</li>
          <li style={{display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px'}}><Wifi size={16} /> Live updates as you search</li>
        </ul>
      </div>
    </div>
  </div>
);

export default SettingsContainer;
