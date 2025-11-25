import React from 'react';
import { Users } from 'lucide-react';

const ChildTabContent = ({ 
  user, 
  activeTab, 
  studentReports, 
  loadingReports, 
  expandedReports,
  setExpandedReports,
  fetchStudentReports 
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
          
          const reports = studentReports[childName] || [];
          const isLoadingReports = loadingReports[childName] || false;
          
          return (
            <div key={`content-${childName}-${idx}`} className="child-tab-content">
              <div className="child-profile">
                <div className="child-header">
                  <Users size={48} className="child-icon" />
                  <h2>{childName}'s Profile</h2>
                </div>
                <div className="child-details">
                  <div className="detail-card">
                    <h3>Grade</h3>
                    <p>{childGrade || 'Not specified'}</p>
                  </div>
                  <div className="detail-card">
                    <h3>School</h3>
                    <p>{childSchool || 'Not specified'}</p>
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
                            {reportsByDate[dateKey].map((report, idx) => (
                              <div key={report.id || idx} className="email-report-card">
                                <div className="email-report-header">
                                  <strong>📧 {report.email_subject || report.subject || 'Student Report'}</strong>
                                  <span className="email-sender">{report.sender}</span>
                                </div>
                                <div className="email-body-content">
                                  {/* Render HTML if body looks like HTML, else plain text */}
                                  {report.email_body && report.email_body.trim().startsWith('<') ? (
                                    <div dangerouslySetInnerHTML={{ __html: report.email_body }} />
                                  ) : (
                                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{report.email_body || 'No content available'}</pre>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        ));
                      })()}
                    </>
                  ) : (
                    <p className="no-reports">No student reports found in email. Make sure you have student report emails in your inbox for {childName}.</p>
                  )}
                </div>
                
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
