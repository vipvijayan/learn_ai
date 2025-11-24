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
                    <h3>Age</h3>
                    <p>{childAge || 'Not specified'}</p>
                  </div>
                </div>
                
                {/* Student Reports Section */}
                <div className="child-section student-reports-section">
                  <h3>📊 Student Reports</h3>
                  {isLoadingReports ? (
                    <p className="loading-message">Loading student reports...</p>
                  ) : reports.length > 0 ? (
                    <>
                      {/* Group reports by date */}
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
                            </div>
                            <div className="lessons-table-container">
                              <table className="lessons-table">
                                <thead>
                                  <tr>
                                    <th>Email Subject</th>
                                    <th>Subject Area</th>
                                    <th>Expected Lessons</th>
                                    <th>Completed Lessons</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {reportsByDate[dateKey].map((report, idx) => {
                                    // Extract numeric values, default to 0 if not found
                                    const expectedStr = report.lessons_expected || '0';
                                    const completedStr = report.lessons_completed || '0';
                                    // Parse numbers from strings like "2 lessons" or just "2"
                                    const expectedNum = parseInt(expectedStr.toString().match(/\d+/)?.[0] || '0');
                                    const completedNum = parseInt(completedStr.toString().match(/\d+/)?.[0] || '0');
                                    // Display values
                                    const expected = expectedNum.toString();
                                    const completed = completedNum.toString();
                                    // Determine status
                                    let statusClass = 'status-unknown';
                                    let statusText = '-';
                                    if (expectedNum > 0) {
                                      if (completedNum >= expectedNum) {
                                        statusClass = 'status-met';
                                        statusText = '✓ Met';
                                      } else {
                                        statusClass = 'status-below';
                                        statusText = '⚠ Below';
                                      }
                                    } else if (completedNum > 0) {
                                      // Has completed lessons but no expected value
                                      statusClass = 'status-met';
                                      statusText = '✓ Done';
                                    }
                                    return (
                                      <tr key={report.id || idx}>
                                        <td className="email-subject-cell">{report.subject || report.email_subject || '-'}</td>
                                        <td className="subject-cell">{report.subject_area}</td>
                                        <td className="expected-cell">{expected}</td>
                                        <td className="completed-cell">{completed}</td>
                                        <td className="status-cell">
                                          <span className={`status-badge-small ${statusClass}`}>{statusText}</span>
                                        </td>
                                        <td className="actions-cell">
                                          {report.email_body && (
                                            <button 
                                              className="view-email-btn-small"
                                              onClick={() => setExpandedReports(prev => ({
                                                ...prev,
                                                [report.id]: !prev[report.id]
                                              }))}
                                              title={expandedReports[report.id] ? "Hide email" : "View email"}
                                            >
                                              {expandedReports[report.id] ? '▼' : '▶'}
                                            </button>
                                          )}
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                            {/* Show expanded email content below table */}
                            {reportsByDate[dateKey].map(report => (
                              expandedReports[report.id] && report.email_body && (
                                <div key={`email-${report.id}`} className="expanded-email-section">
                                  <div className="expanded-email-header">
                                    <strong>{report.subject_area} - Full Email</strong>
                                    <button 
                                      className="close-email-btn"
                                      onClick={() => setExpandedReports(prev => ({
                                        ...prev,
                                        [report.id]: false
                                      }))}
                                    >
                                      ✕ Close
                                    </button>
                                  </div>
                                  <div className="email-body-content">
                                    {report.email_body}
                                  </div>
                                </div>
                              )
                            ))}
                          </div>
                        ));
                      })()}
                    </>
                  ) : (
                    <p className="no-reports">No student reports found in email. Make sure you have student report emails in your inbox for {childName}.</p>
                  )}
                  
                  {/* Overall Summary Section */}
                  {!isLoadingReports && reports.length > 0 && (() => {
                    // Calculate metrics
                    const totalLessons = reports.reduce((sum, r) => {
                      const lessonsCompletedStr = (r.lessons_completed !== undefined && r.lessons_completed !== null) ? r.lessons_completed.toString() : "0";
                      const match = lessonsCompletedStr.match(/(\d+)/);
                      return sum + (match ? parseInt(match[1]) : 0);
                    }, 0);

                    const totalTimeMinutes = reports.reduce((sum, r) => {
                      const timeSpentStr = (r.time_spent !== undefined && r.time_spent !== null) ? r.time_spent.toString() : "0";
                      const match = timeSpentStr.match(/(\d+)/);
                      return sum + (match ? parseInt(match[1]) : 0);
                    }, 0);
                    
                    const totalTimeHours = Math.floor(totalTimeMinutes / 60);
                    const remainingMinutes = totalTimeMinutes % 60;
                    
                    const uniqueSubjects = [...new Set(reports.map(r => r.subject_area))];
                    const reportsWithScores = reports.filter(r => r.score).length;
                    const reportsWithLessons = reports.filter(r => r.lessons_completed).length;
                    const reportsWithTime = reports.filter(r => r.time_spent).length;
                    
                    return (
                      <div className="overall-summary">
                        <h4>📈 Overall Progress Summary</h4>
                        <div className="summary-stats">
                          <div className="stat-card">
                            <div className="stat-label">Total Reports</div>
                            <div className="stat-value">{reports.length}</div>
                          </div>
                          <div className="stat-card">
                            <div className="stat-label">Subjects Tracked</div>
                            <div className="stat-value">{uniqueSubjects.length}</div>
                          </div>
                          {totalLessons > 0 && (
                            <div className="stat-card">
                              <div className="stat-label">Lessons Completed</div>
                              <div className="stat-value">{totalLessons}</div>
                              <div className="stat-sublabel">across {reportsWithLessons} report{reportsWithLessons !== 1 ? 's' : ''}</div>
                            </div>
                          )}
                          {totalTimeMinutes > 0 && (
                            <div className="stat-card">
                              <div className="stat-label">Total Time</div>
                              <div className="stat-value">
                                {totalTimeHours > 0 ? `${totalTimeHours}h ${remainingMinutes}m` : `${totalTimeMinutes}m`}
                              </div>
                              <div className="stat-sublabel">across {reportsWithTime} report{reportsWithTime !== 1 ? 's' : ''}</div>
                            </div>
                          )}
                          {reportsWithScores > 0 && (
                            <div className="stat-card">
                              <div className="stat-label">Reports with Scores</div>
                              <div className="stat-value">{reportsWithScores}</div>
                              <div className="stat-sublabel">out of {reports.length}</div>
                            </div>
                          )}
                        </div>
                        <div className="summary-details">
                          <h5>Subject Breakdown:</h5>
                          <div className="subject-breakdown">
                            {[...new Set(reports.map(r => r.subject_area))].map(subject => {
                              const subjectReports = reports.filter(r => r.subject_area === subject);
                              return (
                                <div key={subject + '-' + subjectReports[0]?.id} className="subject-summary-item">
                                  <strong>{subject}:</strong>
                                  <span> {subjectReports.length} report{subjectReports.length !== 1 ? 's' : ''}</span>
                                  {subjectReports.some(r => r.performance) && (
                                    <span className="subject-performance">
                                      {' - '}{subjectReports.find(r => r.performance)?.performance}
                                    </span>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                          {reports.some(r => r.performance) && (
                            <div className="performance-overview">
                              <h5>Performance Status:</h5>
                              <div className="performance-badges">
                                {[...new Set(reports.filter(r => r.performance).map(r => r.performance))].map(perf => (
                                  <span key={perf} className="badge badge-info">{perf}</span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {/* Weekly Lessons Tracking */}
                          {totalLessons > 0 && (() => {
                            const districtExpectation = 2; // District expects 2 lessons per week
                            
                            // Group reports by week and keep individual report details
                            const weeklyData = {};
                            
                            reports.forEach(report => {
                              if (report.lessons_completed && report.date) {
                                const lessonsCompletedStr = report.lessons_completed.toString();
                                const match = lessonsCompletedStr.match(/(\d+)/);
                                if (match) {
                                  const lessons = parseInt(match[1]);
                                  let dateStr = report.date;
                                  let weekKey = dateStr;
                                  
                                  // Try to extract a simple date identifier
                                  const monthMatch = dateStr.toLowerCase().match(/(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d+)/);
                                  if (monthMatch) {
                                    weekKey = `${monthMatch[1]} ${monthMatch[2]}`;
                                  } else {
                                    weekKey = dateStr.substring(0, 20);
                                  }
                                  
                                  if (!weeklyData[weekKey]) {
                                    weeklyData[weekKey] = {
                                      totalLessons: 0,
                                      totalTime: 0,
                                      reports: []
                                    };
                                  }
                                  
                                  const timeMatch = report.time_spent?.match(/(\d+)/);
                                  const time = timeMatch ? parseInt(timeMatch[1]) : 0;
                                  
                                  weeklyData[weekKey].totalLessons += lessons;
                                  weeklyData[weekKey].totalTime += time;
                                  weeklyData[weekKey].reports.push({
                                    subject: report.subject_area,
                                    lessons: lessons,
                                    time: time,
                                    score: report.score,
                                    performance: report.performance,
                                    emailSubject: report.email_subject
                                  });
                                }
                              }
                            });
                            
                            const weeks = Object.entries(weeklyData).sort((a, b) => b[1].reports.length - a[1].reports.length);
                            
                            return weeks.length > 0 ? (
                              <div className="weekly-lessons-section">
                                <h5>📅 Weekly Progress Tracking:</h5>
                                {weeks.map(([week, data]) => (
                                  <div key={week} className="weekly-progress-card">
                                    <div className="progress-card-header">
                                      <h6>{week}</h6>
                                      <div className="header-stats">
                                        <span className="expectation-badge">Expected: {districtExpectation} lessons</span>
                                        <span className={`completed-badge ${data.totalLessons >= districtExpectation ? 'met' : 'below'}`}>
                                          Completed: {data.totalLessons}
                                        </span>
                                      </div>
                                    </div>
                                    <div className="progress-card-body">
                                      <table className="report-details-table">
                                        <thead>
                                          <tr>
                                            <th>Report</th>
                                            <th>Subject</th>
                                            <th>Lessons</th>
                                            <th>Time</th>
                                            <th>Status</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {data.reports.map((report, idx) => (
                                            <tr key={idx}>
                                              <td className="report-name">{report.emailSubject || `Report ${idx + 1}`}</td>
                                              <td className="subject-name">{report.subject}</td>
                                              <td className="lessons-count">{report.lessons}</td>
                                              <td className="time-spent">{report.time > 0 ? `${report.time}m` : '-'}</td>
                                              <td className="status-cell">
                                                {report.performance ? (
                                                  <span className={`status-badge ${report.performance.toLowerCase().replace(/\s+/g, '-')}`}>
                                                    {report.performance}
                                                  </span>
                                                ) : report.score ? (
                                                  <span className="status-badge">{report.score}</span>
                                                ) : '-'}
                                              </td>
                                            </tr>
                                          ))}
                                          <tr className="totals-row">
                                            <td colSpan="2" className="totals-label"><strong>Week Total:</strong></td>
                                            <td className="totals-lessons"><strong>{data.totalLessons}</strong></td>
                                            <td className="totals-time"><strong>{data.totalTime > 0 ? `${data.totalTime}m` : '-'}</strong></td>
                                            <td></td>
                                          </tr>
                                        </tbody>
                                      </table>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : null;
                          })()}
                        </div>
                      </div>
                    );
                  })()}
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
