import React from 'react';

const CopyToast = () => (
  <div className="copy-toast">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
    <span>Copied to clipboard!</span>
  </div>
);

export default CopyToast;
