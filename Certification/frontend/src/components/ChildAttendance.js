import React from 'react';

const ChildAttendance = ({
  child,
  user,
  attendanceData,
  loadingAttendance,
  fetchChildAttendance
}) => {
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

  if (!child) return null;

  const items = (attendanceData && attendanceData[childId]) || [];

  return (
    <div className="child-section attendance-section">
      <h3>📝 Attendance</h3>
      {loadingAttendance && loadingAttendance[childId] ? (
        <p className="loading-message">Loading attendance emails...</p>
      ) : items.length > 0 ? (
        <>
          {(() => {
            const grouped = {};
            items.forEach(item => {
              const dateKey = item.date || 'Unknown Date';
              if (!grouped[dateKey]) grouped[dateKey] = [];
              grouped[dateKey].push(item);
            });
            const sorted = Object.keys(grouped).sort((a,b) => {
              if (a === 'Unknown Date') return 1;
              if (b === 'Unknown Date') return -1;
              return new Date(b) - new Date(a);
            });
            return sorted.map(dateKey => (
              <div key={dateKey} className="date-group-card">
                <div className="date-group-header">
                  <h4>📅 {dateKey}</h4>
                  <span className="report-count">{grouped[dateKey].length} email{grouped[dateKey].length !== 1 ? 's' : ''}</span>
                </div>
                {grouped[dateKey].map((att, idx) => (
                  <div key={att.id || idx} className="email-report-card compact">
                    <div className="email-report-header">
                      <strong>📧 {att.subject || 'Attendance'}</strong>
                      <span className="email-sender">{att.from}</span>
                    </div>
                    {/* Show only a short preview/summary instead of full body */}
                    <div className="email-summary-preview">
                      {att.summary ? (
                        <p style={{ margin: 0 }}>{att.summary}</p>
                      ) : (
                        <p style={{ margin: 0 }}>{(att.preview || (att.trimmed_body || '')).slice(0, 250) + ((att.preview || att.trimmed_body || '').length > 250 ? '...' : '')}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ));
          })()}
        </>
      ) : (
        <p className="no-reports">No attendance emails found for {childName}. Make sure Gmail is connected and attendance emails exist.</p>
      )}
    </div>
  );
};

export default ChildAttendance;
