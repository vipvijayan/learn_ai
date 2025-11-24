import React from 'react';

const ComparisonInsights = () => (
  <div className="comparison-insights">
    <h4>💡 Key Insights</h4>
    <div className="insights-grid">
      <div className="insight-card">
        <h5>Original RAG (k=4)</h5>
        <ul>
          <li>Retrieves 4 most relevant documents</li>
          <li>Faster query processing</li>
          <li>More focused context</li>
        </ul>
      </div>
      <div className="insight-card">
        <h5>Naive Retrieval (k=10)</h5>
        <ul>
          <li>Retrieves 10 most relevant documents</li>
          <li>LCEL chain pattern from Advanced Retrieval</li>
          <li>More comprehensive context</li>
        </ul>
      </div>
    </div>
  </div>
);

export default ComparisonInsights;
