import React from 'react';
import { CheckCircle, Target, FileText, AlertTriangle } from 'lucide-react';

const ComparisonResults = ({ comparisonResults }) => {
  if (!comparisonResults.original || !comparisonResults.naive) return null;
  return (
    <div className="comparison-results">
      <div className="comparison-summary">
        <h3><CheckCircle size={20} style={{ display: 'inline', marginRight: '8px', color: '#4caf50', verticalAlign: 'middle' }} /> Comparison Complete</h3>
        <p>Both retrieval methods have been evaluated with {comparisonResults.original.test_questions_count} test questions</p>
      </div>
      <div className="comparison-table-container">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Original RAG (k=4)</th>
              <th>Naive Retrieval (k=10)</th>
              <th>Improvement</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong><Target size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> Faithfulness</strong><br/><span className="metric-desc">Factual accuracy</span></td>
              <td className="metric-cell">
                <div className="metric-value-large">{(comparisonResults.original.metrics.faithfulness * 100).toFixed(1)}%</div>
                <div className="mini-bar">
                  <div className="mini-bar-fill" style={{width: `${comparisonResults.original.metrics.faithfulness * 100}%`}}></div>
                </div>
              </td>
              <td className="metric-cell">
                <div className="metric-value-large">{(comparisonResults.naive.metrics.faithfulness * 100).toFixed(1)}%</div>
                <div className="mini-bar">
                  <div className="mini-bar-fill" style={{width: `${comparisonResults.naive.metrics.faithfulness * 100}%`}}></div>
                </div>
              </td>
              <td className={`improvement-cell ${comparisonResults.naive.metrics.faithfulness >= comparisonResults.original.metrics.faithfulness ? 'positive' : 'negative'}`}>
                {comparisonResults.naive.metrics.faithfulness >= comparisonResults.original.metrics.faithfulness ? <CheckCircle size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> : <AlertTriangle size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />}
                {((comparisonResults.naive.metrics.faithfulness - comparisonResults.original.metrics.faithfulness) * 100).toFixed(1)}%
              </td>
            </tr>
            <tr>
              <td><strong><FileText size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> Answer Relevancy</strong><br/><span className="metric-desc">Relevance to question</span></td>
              <td className="metric-cell">
                <div className="metric-value-large">{(comparisonResults.original.metrics.answer_relevancy * 100).toFixed(1)}%</div>
                <div className="mini-bar">
                  <div className="mini-bar-fill" style={{width: `${comparisonResults.original.metrics.answer_relevancy * 100}%`}}></div>
                </div>
              </td>
              <td className="metric-cell">
                <div className="metric-value-large">{(comparisonResults.naive.metrics.answer_relevancy * 100).toFixed(1)}%</div>
                <div className="mini-bar">
                  <div className="mini-bar-fill" style={{width: `${comparisonResults.naive.metrics.answer_relevancy * 100}%`}}></div>
                </div>
              </td>
              <td className={`improvement-cell ${comparisonResults.naive.metrics.answer_relevancy >= comparisonResults.original.metrics.answer_relevancy ? 'positive' : 'negative'}`}>
                {comparisonResults.naive.metrics.answer_relevancy >= comparisonResults.original.metrics.answer_relevancy ? <CheckCircle size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> : <AlertTriangle size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />}
                {((comparisonResults.naive.metrics.answer_relevancy - comparisonResults.original.metrics.answer_relevancy) * 100).toFixed(1)}%
              </td>
            </tr>
            <tr>
              <td><strong><Target size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> Context Precision</strong><br/><span className="metric-desc">Precision of contexts</span></td>
              <td className="metric-cell">
                <div className="metric-value-large">{(comparisonResults.original.metrics.context_precision * 100).toFixed(1)}%</div>
                <div className="mini-bar">
                  <div className="mini-bar-fill" style={{width: `${comparisonResults.original.metrics.context_precision * 100}%`}}></div>
                </div>
              </td>
              <td className="metric-cell">
                <div className="metric-value-large">{(comparisonResults.naive.metrics.context_precision * 100).toFixed(1)}%</div>
                <div className="mini-bar">
                  <div className="mini-bar-fill" style={{width: `${comparisonResults.naive.metrics.context_precision * 100}%`}}></div>
                </div>
              </td>
              <td className={`improvement-cell ${comparisonResults.naive.metrics.context_precision >= comparisonResults.original.metrics.context_precision ? 'positive' : 'negative'}`}>
                {comparisonResults.naive.metrics.context_precision >= comparisonResults.original.metrics.context_precision ? <CheckCircle size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> : <AlertTriangle size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />}
                {((comparisonResults.naive.metrics.context_precision - comparisonResults.original.metrics.context_precision) * 100).toFixed(1)}%
              </td>
            </tr>
            <tr>
              <td><strong><Target size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> Context Recall</strong><br/><span className="metric-desc">Completeness of context</span></td>
              <td className="metric-cell">
                <div className="metric-value-large">{(comparisonResults.original.metrics.context_recall * 100).toFixed(1)}%</div>
                <div className="mini-bar">
                  <div className="mini-bar-fill" style={{width: `${comparisonResults.original.metrics.context_recall * 100}%`}}></div>
                </div>
              </td>
              <td className="metric-cell">
                <div className="metric-value-large">{(comparisonResults.naive.metrics.context_recall * 100).toFixed(1)}%</div>
                <div className="mini-bar">
                  <div className="mini-bar-fill" style={{width: `${comparisonResults.naive.metrics.context_recall * 100}%`}}></div>
                </div>
              </td>
              <td className={`improvement-cell ${comparisonResults.naive.metrics.context_recall >= comparisonResults.original.metrics.context_recall ? 'positive' : 'negative'}`}>
                {comparisonResults.naive.metrics.context_recall >= comparisonResults.original.metrics.context_recall ? <CheckCircle size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> : <AlertTriangle size={16} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />}
                {((comparisonResults.naive.metrics.context_recall - comparisonResults.original.metrics.context_recall) * 100).toFixed(1)}%
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ComparisonResults;
