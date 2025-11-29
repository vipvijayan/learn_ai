import React from 'react';

const ComparisonHeader = ({ comparisonResults, isRunningComparison, runComparison, error }) => (
  <div className="evaluation-container">
    <div className="evaluation-header">
      <h2>📈 Retrieval Methods Comparison</h2>
      <p>Compare Original RAG (k=4) vs Naive Retrieval (k=10) using RAGAS metrics</p>
      {comparisonResults.original && comparisonResults.naive && (
        <button 
          onClick={runComparison} 
          disabled={isRunningComparison}
          className="run-evaluation-button"
        >
          {isRunningComparison ? '⏳ Re-running Comparison...' : '🔄 Re-run Comparison'}
        </button>
      )}
    </div>
    {error && (
      <div className="error-message">
        {error}
      </div>
    )}
    {isRunningComparison && (
      <div className="evaluation-loading">
        <div className="spinner"></div>
        <p>Running comparison evaluation... This may take 3-5 minutes</p>
        <p className="evaluation-details">
          Evaluating both Original (k=4) and Naive (k=10) methods
        </p>
      </div>
    )}
  </div>
);

export default ComparisonHeader;
