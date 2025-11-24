import React, { useState } from 'react';
import axios from 'axios';
import './ChildrenInput.css';

const isLocalMode = process.env.REACT_APP_LOCAL_MODE === 'true';
const API_BASE_URL = isLocalMode 
  ? 'http://localhost:8000' 
  : (process.env.REACT_APP_API_URL || 'https://school-assistant-production.up.railway.app');

const ChildrenInput = ({ email, onChildrenAdded, onSkip }) => {
  const [children, setChildren] = useState(['']);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleChildChange = (idx, value) => {
    setChildren(prev => prev.map((c, i) => (i === idx ? value : c)));
  };

  const handleAddChild = () => {
    setChildren(prev => [...prev, '']);
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
    const filtered = children.map(c => c.trim()).filter(Boolean);
    if (filtered.length === 0) {
      setError('Please add at least one child or click "Skip for now" below.');
      setSubmitting(false);
      return;
    }
    try {
      await axios.post(`${API_BASE_URL}/api/auth/add-children`, {
        email,
        children: filtered
      });
      setSubmitting(false);
      if (onChildrenAdded) onChildrenAdded(filtered);
    } catch (err) {
      setError('Failed to save children.');
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
            Please add the names of your children. You can skip this step if you prefer.
          </p>
        </div>

        {error && (
          <div className="children-input-error">
            {error}
          </div>
        )}

        <form onSubmit={handleFinish}>
          {children.map((child, idx) => (
            <div key={idx} className="children-input-row">
              <input
                type="text"
                value={child}
                onChange={e => handleChildChange(idx, e.target.value)}
                placeholder={`Child ${idx + 1} Name`}
                className="children-input-field"
                required
              />
              {children.length > 1 && (
                <button
                  type="button"
                  onClick={() => handleRemoveChild(idx)}
                  className="children-input-remove"
                  title="Remove child"
                >
                  −
                </button>
              )}
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
            {submitting ? '⏳ Saving Children...' : '✅ Finish & Continue'}
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