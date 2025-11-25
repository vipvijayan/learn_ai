import React, { useState } from 'react';
import axios from 'axios';
import './ChildrenInput.css';

const isLocalMode = process.env.REACT_APP_LOCAL_MODE === 'true';
const API_BASE_URL = isLocalMode 
  ? 'http://localhost:8000' 
  : (process.env.REACT_APP_API_URL || 'https://school-assistant-production.up.railway.app');

const GRADE_OPTIONS = [
  'Pre-K', 'Kindergarten', 
  '1st Grade', '2nd Grade', '3rd Grade', '4th Grade', '5th Grade',
  '6th Grade', '7th Grade', '8th Grade',
  '9th Grade', '10th Grade', '11th Grade', '12th Grade'
];

const ChildrenInput = ({ email, onChildrenAdded, onSkip, existingChildren = [] }) => {
  const [children, setChildren] = useState([{ name: '', grade: '', school_name: '' }]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleChildChange = (idx, field, value) => {
    setChildren(prev => prev.map((c, i) => (i === idx ? { ...c, [field]: value } : c)));
  };

  const handleAddChild = () => {
    setChildren(prev => [...prev, { name: '', grade: '', school_name: '' }]);
  };

  const handleRemoveChild = (idx) => {
    if (children.length > 1) {
      setChildren(prev => prev.filter((_, i) => i !== idx));
    }
  };

  const handleFinish = async (e) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    
    // Filter and format children data
    const filtered = children
      .map(c => ({
        child_name: c.name.trim(),
        grade: c.grade?.trim() || null,
        school_name: c.school_name.trim() || null
      }))
      .filter(c => c.child_name);
    
    if (filtered.length === 0) {
      setError('Please add at least one child or click "Skip for now" below.');
      setSubmitting(false);
      return;
    }
    
    try {
      const addedChildren = [];

      for (const child of filtered) {
        try {
          const response = await axios.post(`${API_BASE_URL}/api/auth/children`, {
            email,
            child_name: child.child_name,
            grade: child.grade,
            school_name: child.school_name
          });
          if (response.data?.child) {
            addedChildren.push(response.data.child);
          }
        } catch (err) {
          const detail = err?.response?.data?.detail || 'Failed to add child.';
          setError(`Could not add ${child.child_name}: ${detail}`);
          setSubmitting(false);
          return;
        }
      }

      setSubmitting(false);
      setChildren([{ name: '', grade: '', school_name: '' }]);
      if (addedChildren.length > 0 && onChildrenAdded) {
        onChildrenAdded(addedChildren);
      }
    } catch (err) {
      const detail = err?.response?.data?.detail || 'Failed to save children.';
      setError(detail);
      setSubmitting(false);
    }
  };

  const handleSkip = () => {
    if (onSkip) onSkip();
  };

  return (
    <div className="children-input-bg">
      <div className="children-input-card">
        <div className="children-input-header">
          <h2 className="children-input-title">
            👶 Add Your Children
          </h2>
          <p className="children-input-desc">
            Please add your children's information. You can skip this step if you prefer.
          </p>
        </div>

        {error && (
          <div className="children-input-error">
            {error}
          </div>
        )}

        <form onSubmit={handleFinish}>
          {existingChildren.length > 0 && (
            <div className="children-input-existing">
              <p>
                You already have {existingChildren.length} child{existingChildren.length === 1 ? '' : 'ren'} saved. Use the edit button in settings to update them. Add new children below if needed.
              </p>
            </div>
          )}

          {children.map((child, idx) => (
            <div key={idx} className="children-input-group">
              <div className="children-input-group-header">
                <span className="children-input-group-title">Child {idx + 1}</span>
                {children.length > 1 && (
                  <button
                    type="button"
                    onClick={() => handleRemoveChild(idx)}
                    className="children-input-remove"
                    title="Remove child"
                  >
                    ✕
                  </button>
                )}
              </div>
              
              <div className="children-input-field-row">
                <label className="children-input-label">Name *</label>
                <input
                  type="text"
                  value={child.name}
                  onChange={e => handleChildChange(idx, 'name', e.target.value)}
                  placeholder="Enter child's name"
                  className="children-input-field"
                  required
                />
              </div>
              
              <div className="children-input-field-row">
                <label className="children-input-label">Grade</label>
                <select
                  value={child.grade}
                  onChange={e => handleChildChange(idx, 'grade', e.target.value)}
                  className="children-input-field children-input-select"
                >
                  <option value="">Select grade...</option>
                  {GRADE_OPTIONS.map(grade => (
                    <option key={grade} value={grade}>{grade}</option>
                  ))}
                </select>
              </div>
              
              <div className="children-input-field-row">
                <label className="children-input-label">School Name</label>
                <input
                  type="text"
                  value={child.school_name}
                  onChange={e => handleChildChange(idx, 'school_name', e.target.value)}
                  placeholder="Enter school name"
                  className="children-input-field"
                />
              </div>
            </div>
          ))}

          <button
            type="button"
            onClick={handleAddChild}
            className="children-input-add"
          >
            ➕ Add Another Child
          </button>

          <button
            type="submit"
            disabled={submitting}
            className="children-input-submit"
          >
            {submitting ? '⏳ Saving Children...' : '✅ Save & Continue'}
          </button>

          <button
            type="button"
            onClick={handleSkip}
            className="children-input-skip"
          >
            Skip for now
          </button>
        </form>

        <div className="children-input-footer">
          <p style={{ margin: 0 }}>
            You can always add or manage your children later in settings
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChildrenInput;