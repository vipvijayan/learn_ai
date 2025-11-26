import React from 'react';
import { Users } from 'lucide-react';
import ChildAttendance from './ChildAttendance';

const ChildTabContent = ({ 
  user, 
  activeTab, 
  studentReports, 
  loadingReports, 
  expandedReports,
  setExpandedReports,
  fetchStudentReports,
  // attendance props
  attendanceData,
  loadingAttendance,
  fetchChildAttendance
}) => {
  if (!user || !user.children || user.children.length === 0) {
    return null;
  }

  return (
    <>
      {user.children.map((child, idx) => {
        const childName = child.child_name || child || child.name || `Child ${idx + 1}`;
        const childGrade = child.child_grade || child.grade;
        const childAge = child.child_age || child.age;
        const childSchool = child.child_school || child.school_name;
        
        if (activeTab === `child-${childName}`) {
          // Fetch student reports when tab is opened
          if (!studentReports[childName] && !loadingReports[childName] && user && user.email) {
            fetchStudentReports(user.email, childName);
          }
          // Fetch attendance when tab is opened
          const childId = child.child_id || child.id;
          if (fetchChildAttendance && childId && (!attendanceData || !attendanceData[childId]) && (!loadingAttendance || !loadingAttendance[childId]) && user && user.email) {
            fetchChildAttendance(user.email, childId);
          }
          
          const reports = studentReports[childName] || [];
          const isLoadingReports = loadingReports[childName] || false;
          
          return (
            <div key={`content-${childName}-${idx}`} className="child-tab-content">
              <div className="child-profile">
                <div className="child-header" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Users size={48} className="child-icon" />
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <h2 style={{ margin: 0 }}>{childName}'s Profile</h2>
                    <p style={{ margin: 0, fontSize: '0.95rem', color: '#666' }}>
                      {childGrade || 'Grade not specified'}{childGrade && childSchool ? ' • ' : ''}{childSchool || ''}
                    </p>
                  </div>
                </div>
                
                {/* Student Reports Section */}
                <div className="child-section student-reports-section">
                  <h3>📊 Student Reports</h3>
                  {isLoadingReports ? (
                    <p className="loading-message">Loading student reports...</p>
                  ) : reports.length > 0 ? (
                    <>
                      {/* Group reports by date and show email body */}
                      {(() => {
                        const reportsByDate = {};
                        reports.forEach(report => {
                          const dateKey = report.date || 'Unknown Date';
                          if (!reportsByDate[dateKey]) {
                            reportsByDate[dateKey] = [];
                          }
                          reportsByDate[dateKey].push(report);
                        });
                        
                        // Sort dates (most recent first)
                        const sortedDates = Object.keys(reportsByDate).sort((a, b) => {
                          if (a === 'Unknown Date') return 1;
                          if (b === 'Unknown Date') return -1;
                          return new Date(b) - new Date(a);
                        });
                        
                        return sortedDates.map(dateKey => (
                          <div key={dateKey} className="date-group-card">
                            <div className="date-group-header">
                              <h4>📅 {dateKey}</h4>
                              <span className="report-count">{reportsByDate[dateKey].length} email{reportsByDate[dateKey].length !== 1 ? 's' : ''}</span>
                            </div>
                            {/* Show each email for this date */}
                            {reportsByDate[dateKey].map((report, idx) => {
                              const reportId = report.id || `${dateKey}-${idx}`;
                              const isExpanded = Boolean(expandedReports && expandedReports[reportId]);

                              // Helper to strip HTML tags in case the email body is HTML
                              const stripHtml = (html) => {
                                if (!html) return '';
                                try {
                                  // Use DOM parser where available (browser)
                                  const doc = new DOMParser().parseFromString(html, 'text/html');
                                  return doc.body.textContent || doc.body.innerText || '';
                                } catch (e) {
                                  // Fallback to regex
                                  return html.replace(/<[^>]+>/g, '');
                                }
                              };

                              // Prefer structured summary, then preview/trimmed body, then strip HTML from raw body
                              let previewSource = report.summary || report.preview || report.trimmed_body;
                              if (!previewSource) {
                                previewSource = report.email_body ? stripHtml(report.email_body) : '';
                              }
                              const previewText = (previewSource || '').slice(0, 250);

                              return (
                                <div key={reportId} className={`email-report-card ${isExpanded ? 'expanded' : 'compact'}`}>
                                  <div className="email-report-header">
                                    <strong>📧 {report.email_subject || report.subject || 'Student Report'}</strong>
                                    <span className="email-sender">{report.sender}</span>
                                  </div>

                                  {/* Preview / summary */}
                                  <div className="email-summary-preview">
                                    <p style={{ margin: 0 }}>{previewText}</p>
                                  </div>

                                  {/* Expand / Collapse control */}
                                  <div style={{ marginTop: '8px' }}>
                                    <button
                                      onClick={() => setExpandedReports(prev => ({ ...(prev || {}), [reportId]: !prev?.[reportId] }))}
                                      style={{
                                        background: 'transparent',
                                        border: 'none',
                                        color: '#1976d2',
                                        cursor: 'pointer',
                                        padding: 0,
                                        fontSize: '0.95em'
                                      }}
                                    >
                                      {isExpanded ? 'Collapse' : 'View full report'}
                                    </button>
                                  </div>

                                  {/* Full body (shown when expanded) */}
                                  {isExpanded && (
                                    <div className="email-body-content" style={{ marginTop: '10px' }}>
                                      {report.email_body && typeof report.email_body === 'string' && report.email_body.trim().startsWith('<') ? (
                                        <div dangerouslySetInnerHTML={{ __html: report.email_body }} />
                                      ) : (
                                        <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{report.email_body || 'No content available'}</pre>
                                      )}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        ));
                      })()}
                    </>
                  ) : (
                    <p className="no-reports">No student reports found in email. Make sure you have student report emails in your inbox for {childName}.</p>
                  )}
                </div>

                {/* Attendance Section (separate component) */}
                <ChildAttendance
                  child={child}
                  user={user}
                  attendanceData={attendanceData}
                  loadingAttendance={loadingAttendance}
                  fetchChildAttendance={fetchChildAttendance}
                />
                
                <div className="child-section">
                  <h3>Events & Activities</h3>
                  <p className="coming-soon">Coming soon: View events filtered for {childName}</p>
                </div>
                <div className="child-section">
                  <h3>Assignments</h3>
                  <p className="coming-soon">Coming soon: View and manage {childName}'s assignments</p>
                </div>
              </div>
            </div>
          );
        }
        return null;
      })}
    </>
  );
};

export default ChildTabContent;
