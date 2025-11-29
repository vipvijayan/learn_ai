import React, { useState } from 'react';
import { Users, ChevronDown, ChevronRight } from 'lucide-react';
import BarChartIcon from '../assets/BarChartIcon.svg';
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
  // State for collapsible year sections in student reports
  const [collapsedReportYears, setCollapsedReportYears] = useState({});

  const toggleReportYear = (childName, year) => {
    const key = `${childName}-${year}`;
    setCollapsedReportYears(prev => ({ ...prev, [key]: !prev[key] }));
  };

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
          
          // Get report data from backend-grouped structure
          const reportData = studentReports[childName] || { all: [], byYear: {}, years: [] };
          const reports = reportData.all || [];
          const reportsByYear = reportData.byYear || {};
          const reportYears = reportData.years || [];
          const isLoadingReports = loadingReports[childName] || false;
          
          return (
            <div key={`content-${childName}-${idx}`} className="child-tab-content">
              <div className="child-profile">
                <div className="child-header">
                  <Users size={48} className="child-icon" />
                  <div className="child-header-details">
                    <h2 className="child-header-title">{childName}'s Profile</h2>
                    <p className="child-header-meta">
                      {childGrade || 'Grade not specified'}{childGrade && childSchool ? ' • ' : ''}{childSchool || ''}
                    </p>
                  </div>
                </div>
                
                {/* Student Reports Section */}
                <div className="student-reports-section">
                  <h3 className="student-reports-title">
                    <img src={BarChartIcon} alt="Student Reports" className="student-reports-icon" />
                    Student Reports
                  </h3>
                  {isLoadingReports ? (
                    <p className="loading-message">Loading student reports...</p>
                  ) : reports.length > 0 ? (
                    <div className="student-reports-year-list">
                      {reportYears.map((year, idx) => {
                        const yearKey = `${childName}-${year}`;
                        const isYearCollapsed = collapsedReportYears[yearKey] !== false; // Default to collapsed
                        const yearReports = reportsByYear[year] || [];

                        return (
                          <div key={year} className={`year-group-section${idx === 0 ? ' first' : ''}${isYearCollapsed ? ' collapsed' : ' expanded'}`}>
                            <button
                              className="year-group-header modern-expand-btn"
                              onClick={() => toggleReportYear(childName, year)}
                              aria-label={isYearCollapsed ? `Expand ${year}` : `Collapse ${year}`}
                              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                              onMouseLeave={e => e.currentTarget.style.transform = 'none'}
                            >
                              <div className="year-group-header-content">
                                <img src={BarChartIcon} alt="Student Reports" className="year-group-icon" />
                                <span className="year-group-title">{year}</span>
                                <span className="year-group-count">{yearReports.length} report{yearReports.length !== 1 ? 's' : ''}</span>
                              </div>
                              <span className="year-group-chevron">
                                <span className={`chevron-icon${isYearCollapsed ? '' : ' rotated'}`}>
                                  <ChevronRight size={26} />
                                </span>
                              </span>
                            </button>
                            {!isYearCollapsed && (
                              <div className="year-group-details">
                                {yearReports.map((report, idx) => {
                                  const reportId = report.id || `${year}-${idx}`;
                                  const isExpanded = Boolean(expandedReports && expandedReports[reportId]);

                                  // Helper to strip HTML tags in case the email body is HTML
                                  const stripHtml = (html) => {
                                    if (!html) return '';
                                    try {
                                      const doc = new DOMParser().parseFromString(html, 'text/html');
                                      return doc.body.textContent || doc.body.innerText || '';
                                    } catch (e) {
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
                                        <div>
                                          <strong>📧 {report.email_subject || report.subject || 'Student Report'}</strong>
                                          <div className="email-date-info">{report.date}</div>
                                        </div>
                                        <span className="email-sender">{report.sender}</span>
                                      </div>

                                      {/* Preview / summary */}
                                      <div className="email-summary-preview">
                                        <p>{previewText}</p>
                                      </div>

                                      {/* Expand / Collapse control */}
                                      <div className="email-report-expand">
                                        <button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            setExpandedReports(prev => ({ ...(prev || {}), [reportId]: !prev?.[reportId] }));
                                          }}
                                          className="email-report-expand-btn"
                                        >
                                          {isExpanded ? 'Collapse' : 'View full report'}
                                        </button>
                                      </div>

                                      {/* Full body (shown when expanded) */}
                                      {isExpanded && (
                                        <div className="email-body-content">
                                          {report.email_body && typeof report.email_body === 'string' && report.email_body.trim().startsWith('<') ? (
                                            <div dangerouslySetInnerHTML={{ __html: report.email_body }} />
                                          ) : (
                                            <pre>{report.email_body || 'No content available'}</pre>
                                          )}
                                        </div>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
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
