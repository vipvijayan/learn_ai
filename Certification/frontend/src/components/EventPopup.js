import React from 'react';
import { Tent, Target, Drama, Activity, Palette, BookOpen, Building2, FileText, Users, Calendar, DollarSign, Tag } from 'lucide-react';

const EventPopup = ({ event, onClose, onAskAI }) => {
  // Don't render if no event is provided
  if (!event) return null;

  const getEventIcon = () => {
    if (event.type.includes('Camp')) return <Tent size={32} />;
    if (event.type.includes('Challenge')) return <Target size={32} />;
    if (event.type.includes('Audition')) return <Drama size={32} />;
    if (event.type.includes('Clinic')) return <Activity size={32} />;
    if (event.type.includes('Art')) return <Palette size={32} />;
    return <BookOpen size={32} />;
  };

  return (
    <div className="event-popup-overlay" onClick={onClose}>
      <div className="event-popup" onClick={(e) => e.stopPropagation()}>
        <button className="popup-close" onClick={onClose}>×</button>
        
        <div className="popup-header">
          <div className="popup-icon">
            {getEventIcon()}
          </div>
          <h2>{event.name}</h2>
        </div>

        <div className="popup-content">
          {event.organization && (
            <div className="popup-section">
              <h3><span className="popup-icon-inline"><Building2 size={18} /></span> Organization</h3>
              <p>{event.organization}</p>
            </div>
          )}

          {event.description && (
            <div className="popup-section">
              <h3><span className="popup-icon-inline"><FileText size={18} /></span> Description</h3>
              <p>{event.description}</p>
            </div>
          )}

          <div className="popup-details-grid">
            {event.target_audience && (
              <div className="popup-detail-item">
                <span className="popup-detail-label"><span className="popup-icon-inline-small"><Users size={16} /></span> Target Audience</span>
                <span className="popup-detail-value">{event.target_audience}</span>
              </div>
            )}

            {event.date && (
              <div className="popup-detail-item">
                <span className="popup-detail-label"><span className="popup-icon-inline-small"><Calendar size={16} /></span> Date</span>
                <span className="popup-detail-value">{event.date}</span>
              </div>
            )}

            {event.cost && (
              <div className="popup-detail-item">
                <span className="popup-detail-label"><span className="popup-icon-inline-small"><DollarSign size={16} /></span> Cost</span>
                <span className="popup-detail-value">{event.cost}</span>
              </div>
            )}

            {event.type && (
              <div className="popup-detail-item">
                <span className="popup-detail-label"><span className="popup-icon-inline-small"><Tag size={16} /></span> Type</span>
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
