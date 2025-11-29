import React from 'react';

import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import eventsIcon from '../assets/events.png';

const ChildAttendance = ({
  child,
  user,
  attendanceData,
  loadingAttendance,
  fetchChildAttendance
}) => {
  // Collapsed state for each year (must be before any return)
  const [collapsedYears, setCollapsedYears] = useState({});

  // Compute identifiers safely so hooks run in the same order every render
  const childId = child ? (child.child_id || child.id) : null;
  const childName = child ? (child.child_name || child.name) : 'Child';

  // Trigger fetch when component mounts if not present
  React.useEffect(() => {
    if (!childId) return;
    if (!attendanceData || !attendanceData[childId]) {
      if (user && user.email) {
        fetchChildAttendance(user.email, childId);
      }
    }
  }, [childId, user?.email]);

  // Update collapsedYears when years change
  const childData = attendanceData && attendanceData[childId];
  const byYear = childData?.byYear || {};
  const years = childData?.years || [];
  const items = childData?.all || [];

  React.useEffect(() => {
    // Set all years to collapsed by default when years change
    if (years.length > 0) {
      setCollapsedYears(prev => {
        const updated = { ...prev };
        years.forEach(year => {
          if (!(year in updated)) updated[year] = true;
        });
        // Remove years that no longer exist
        Object.keys(updated).forEach(year => {
          if (!years.includes(year)) delete updated[year];
        });
        return updated;
      });
    }
  }, [years.join(',')]);

  const toggleYear = (year) => {
    setCollapsedYears(prev => ({ ...prev, [year]: !prev[year] }));
  };

  if (!child) return null;

  return (
    <div className="attendance-section">
      <h3 className="attendance-title">
        <span role="img" aria-label="Attendance">📝</span> Attendance
      </h3>
      {loadingAttendance && loadingAttendance[childId] ? (
        <p className="loading-message">Loading attendance emails...</p>
      ) : items.length > 0 ? (
        <div className="attendance-year-list">
          {years.map((year, idx) => (
            <div key={year} className={`year-group-section${idx === 0 ? ' first' : ''}${collapsedYears[year] ? ' collapsed' : ' expanded'}`}>
              <button
                className="year-group-header modern-expand-btn"
                onClick={() => toggleYear(year)}
                aria-label={collapsedYears[year] ? `Expand ${year}` : `Collapse ${year}`}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'none'}
              >
                <div className="year-group-header-content">
                  <img src={eventsIcon} alt="Events" className="year-group-icon" />
                  <span className="year-group-title">{year}</span>
                  <span className="year-group-count">{byYear[year]?.length || 0} email{byYear[year]?.length !== 1 ? 's' : ''}</span>
                </div>
                <span className="year-group-chevron">
                  <span className={`chevron-icon${collapsedYears[year] ? '' : ' rotated'}`}>
                    <ChevronRight size={26} />
                  </span>
                </span>
              </button>
              {!collapsedYears[year] && (
                <div className="year-group-details">
                  {(byYear[year] || []).map((att, idx) => (
                    <div key={att.id || idx} className="email-report-card compact">
                      <div className="email-report-header">
                        <div>
                          <strong>📧 {att.subject || 'Attendance'}</strong>
                          <div className="email-date-info">{att.date}</div>
                        </div>
                        <span className="email-sender">{att.from}</span>
                      </div>
                      {/* Show only a short preview/summary instead of full body */}
                      <div className="email-summary-preview">
                        {att.summary ? (
                          <p>{att.summary}</p>
                        ) : (
                          <p>{(att.preview || (att.trimmed_body || '')).slice(0, 250) + ((att.preview || att.trimmed_body || '').length > 250 ? '...' : '')}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="no-reports">No attendance emails found for {childName}. Make sure Gmail is connected and attendance emails exist.</p>
      )}
    </div>
  );
};

export default ChildAttendance;
