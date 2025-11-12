import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Determine API URL based on LOCAL_MODE flag
const isLocalMode = process.env.REACT_APP_LOCAL_MODE === 'true';
const API_BASE_URL = isLocalMode 
  ? 'http://localhost:8000' 
  : (process.env.REACT_APP_API_URL || 'https://school-assistant-production.up.railway.app');

const SchoolSelection = ({ user, onSchoolSelected }) => {
  const [schools, setSchools] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedSchoolIds, setSelectedSchoolIds] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchSchools();
    // Pre-select user's current schools if they have any
    if (user?.schools && user.schools.length > 0) {
      setSelectedSchoolIds(user.schools.map(s => s.id));
    } else if (user?.selected_school_id) {
      // Legacy single school support
      setSelectedSchoolIds([user.selected_school_id]);
    }
  }, [user]);

  const fetchSchools = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/auth/schools`);
      if (response.data.success) {
        setSchools(response.data.schools);
      }
    } catch (err) {
      console.error('Error fetching schools:', err);
      setError('Failed to load schools. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const toggleSchool = (schoolId) => {
    setSelectedSchoolIds(prev => {
      if (prev.includes(schoolId)) {
        return prev.filter(id => id !== schoolId);
      } else {
        return [...prev, schoolId];
      }
    });
  };

  const handleSchoolSelect = async () => {
    if (selectedSchoolIds.length === 0) {
      setError('Please select at least one school');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      console.log('Sending school selection request:');
      console.log('  Email:', user.email);
      console.log('  School IDs:', selectedSchoolIds);
      console.log('  School IDs type:', typeof selectedSchoolIds);
      console.log('  Is array:', Array.isArray(selectedSchoolIds));
      
      const payload = {
        email: user.email,
        school_ids: selectedSchoolIds
      };
      console.log('  Payload:', JSON.stringify(payload));
      
      const response = await axios.post(`${API_BASE_URL}/api/auth/select-school`, payload);

      if (response.data.success) {
        console.log('✅ School selection successful');
        onSchoolSelected(response.data.user);
      }
    } catch (err) {
      console.error('Error selecting schools:', err);
      console.error('Error response:', err.response?.data);
      const errorDetail = err.response?.data?.detail;
      const errorMessage = typeof errorDetail === 'string' 
        ? errorDetail 
        : 'Failed to select schools. Please try again.';
      setError(errorMessage);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        padding: '48px',
        maxWidth: '600px',
        width: '100%'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{
            fontSize: '1.8em',
            fontWeight: '700',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '8px'
          }}>
            {selectedSchoolIds.length > 0 ? 'Manage Your Schools' : 'Select Your Schools'}
          </h1>
          <p style={{ color: '#666', fontSize: '0.95em', marginBottom: '4px' }}>
            Welcome, <strong>{user.email}</strong>
          </p>
          {selectedSchoolIds.length > 0 && (
            <p style={{ 
              color: '#667eea', 
              fontSize: '0.9em',
              marginTop: '8px',
              padding: '8px 16px',
              background: 'rgba(102, 126, 234, 0.1)',
              borderRadius: '6px',
              display: 'inline-block'
            }}>
              {selectedSchoolIds.length} school{selectedSchoolIds.length !== 1 ? 's' : ''} selected
            </p>
          )}
          <p style={{ color: '#999', fontSize: '0.85em', marginTop: '8px' }}>
            Select one or more schools to get personalized event information
          </p>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <div style={{ fontSize: '2em', marginBottom: '16px' }}>⏳</div>
            <p style={{ color: '#666' }}>Loading schools...</p>
          </div>
        ) : (
          <>
            <div style={{ marginBottom: '24px' }}>
              <label style={{
                display: 'block',
                marginBottom: '12px',
                fontWeight: '600',
                color: '#333',
                fontSize: '0.95em'
              }}>
                Available Schools
              </label>
              <div style={{
                display: 'grid',
                gap: '12px',
                maxHeight: '400px',
                overflowY: 'auto',
                padding: '4px'
              }}>
                {schools.map((school) => {
                  const isSelected = selectedSchoolIds.includes(school.id);
                  return (
                  <div
                    key={school.id}
                    onClick={() => toggleSchool(school.id)}
                    style={{
                      padding: '16px 20px',
                      border: isSelected 
                        ? '3px solid #667eea' 
                        : '2px solid #e0e0e0',
                      borderRadius: '12px',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      background: isSelected 
                        ? 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)'
                        : 'white'
                    }}
                    onMouseOver={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.borderColor = '#ccc';
                        e.currentTarget.style.transform = 'translateX(4px)';
                      }
                    }}
                    onMouseOut={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.borderColor = '#e0e0e0';
                        e.currentTarget.style.transform = 'translateX(0)';
                      }
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <div>
                        <div style={{
                          fontWeight: '600',
                          color: '#333',
                          fontSize: '1.05em',
                          marginBottom: '4px'
                        }}>
                          {school.name}
                        </div>
                        {school.email_suffix && (
                          <div style={{
                            color: '#667eea',
                            fontSize: '0.85em',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            marginBottom: '2px',
                            fontFamily: 'monospace'
                          }}>
                            📧 @{school.email_suffix}
                          </div>
                        )}
                        {school.location && (
                          <div style={{
                            color: '#666',
                            fontSize: '0.9em',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px'
                          }}>
                            📍 {school.location}
                          </div>
                        )}
                      </div>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                      }}>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => {}} 
                          style={{
                            width: '20px',
                            height: '20px',
                            cursor: 'pointer',
                            accentColor: '#667eea'
                          }}
                        />
                        {isSelected && (
                          <div style={{
                            color: '#667eea',
                            fontSize: '1.3em'
                          }}>
                            ✓
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  );
                })}
              </div>
            </div>

            {error && (
              <div style={{
                background: '#fee',
                color: '#c33',
                padding: '12px 16px',
                borderRadius: '8px',
                marginBottom: '20px',
                fontSize: '0.9em',
                border: '1px solid #fcc'
              }}>
                {typeof error === 'string' ? error : JSON.stringify(error)}
              </div>
            )}

            <button
              onClick={handleSchoolSelect}
              disabled={selectedSchoolIds.length === 0 || submitting}
              style={{
                width: '100%',
                padding: '16px',
                fontSize: '1.05em',
                fontWeight: '600',
                color: 'white',
                background: (selectedSchoolIds.length === 0 || submitting)
                  ? '#ccc'
                  : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none',
                borderRadius: '8px',
                cursor: (selectedSchoolIds.length === 0 || submitting) ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
                boxShadow: '0 4px 12px rgba(102, 126, 234, 0.4)'
              }}
              onMouseOver={(e) => {
                if (selectedSchoolIds.length > 0 && !submitting) {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.5)';
                }
              }}
              onMouseOut={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.4)';
              }}
            >
              {submitting 
                ? 'Saving...' 
                : (selectedSchoolIds.length > 0 ? `Continue with ${selectedSchoolIds.length} School${selectedSchoolIds.length !== 1 ? 's' : ''}` : 'Select Schools')}
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default SchoolSelection;
