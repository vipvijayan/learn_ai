import React from 'react';

const EventPopup = ({ event, onClose, onAskAI }) => {
  // Don't render if no event is provided
  if (!event) return null;

  return (
    <div className="event-popup-overlay" onClick={onClose}>
      <div className="event-popup" onClick={(e) => e.stopPropagation()}>
        <button className="popup-close" onClick={onClose}>×</button>
        
        <div className="popup-header">
          <div className="popup-icon">
            {event.type.includes('Camp') ? '🏕️' : 
             event.type.includes('Challenge') ? '🎯' : 
             event.type.includes('Audition') ? '🎭' :
             event.type.includes('Clinic') ? '⚽' :
             event.type.includes('Art') ? '🎨' : '📚'}
          </div>
          <h2>{event.name}</h2>
        </div>

        <div className="popup-content">
          {event.organization && (
            <div className="popup-section">
              <h3>🏢 Organization</h3>
              <p>{event.organization}</p>
            </div>
          )}

          {event.description && (
            <div className="popup-section">
              <h3>📝 Description</h3>
              <p>{event.description}</p>
            </div>
          )}

          <div className="popup-details-grid">
            {event.target_audience && (
              <div className="popup-detail-item">
                <span className="popup-detail-label">👥 Target Audience</span>
                <span className="popup-detail-value">{event.target_audience}</span>
              </div>
            )}

            {event.date && (
              <div className="popup-detail-item">
                <span className="popup-detail-label">📅 Date</span>
                <span className="popup-detail-value">{event.date}</span>
              </div>
            )}

            {event.cost && (
              <div className="popup-detail-item">
                <span className="popup-detail-label">💰 Cost</span>
                <span className="popup-detail-value">{event.cost}</span>
              </div>
            )}

            {event.type && (
              <div className="popup-detail-item">
                <span className="popup-detail-label">🏷️ Type</span>
                <span className="popup-detail-value">{event.type}</span>
              </div>
            )}
          </div>

          <div className="popup-footer">
            <button 
              className="popup-ask-button"
              onClick={() => {
                onClose();
                onAskAI(event);
              }}
            >
              Ask AI About This Event
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EventPopup;
