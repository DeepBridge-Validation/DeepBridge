/**
 * Component for rendering metrics tables
 * 
 * Handles the rendering of model performance metrics tables,
 * including comparisons between models
 */
import { formatNumber, formatPercent } from '../../utils/Formatters.js';

class MetricsTable {
  /**
   * Initialize the metrics table renderer
   */
  constructor() {
    // Initialize any needed properties
  }
  
  /**
   * Render metrics table
   * @param {HTMLElement} container - The container element
   * @param {Object} perturbationData - Perturbation data
   */
  render(container, perturbationData) {
    if (!container || !perturbationData) {
      return;
    }
    
    // Create table for primary model
    const tableData = [];
    
    // Add primary model
    tableData.push({
      model: perturbationData.modelName,
      baseScore: perturbationData.baseScore,
      avgScore: this.calculateAvgScore(perturbationData.scores),
      worstScore: Math.min(...perturbationData.worstScores),
      impact: this.calculateImpact(perturbationData.scores, perturbationData.baseScore)
    });
    
    // Add alternative models if available
    if (perturbationData.alternativeModels) {
      Object.entries(perturbationData.alternativeModels).forEach(([modelName, modelData]) => {
        if (!modelData.scores || modelData.scores.length === 0) return;
        
        tableData.push({
          model: modelName,
          baseScore: modelData.baseScore,
          avgScore: this.calculateAvgScore(modelData.scores.filter(score => score !== null)),
          worstScore: Math.min(...modelData.worstScores.filter(score => score !== null)),
          impact: this.calculateImpact(
            modelData.scores.filter(score => score !== null), 
            modelData.baseScore
          )
        });
      });
    }
    
    const html = this.renderMetricsTableHtml(tableData);
    container.innerHTML = html;
  }
  
  /**
   * Calculate average score from an array of scores
   * @param {Array} scores - Array of score values
   * @return {number} Average score
   */
  calculateAvgScore(scores) {
    if (!scores || scores.length === 0) return 0;
    return scores.reduce((sum, score) => sum + score, 0) / scores.length;
  }
  
  /**
   * Calculate impact as the proportional decrease from base score
   * @param {Array} scores - Array of score values
   * @param {number} baseScore - Base score value
   * @return {number} Impact value (0-1)
   */
  calculateImpact(scores, baseScore) {
    if (!scores || scores.length === 0 || !baseScore) return 0;
    const avgScore = this.calculateAvgScore(scores);
    return 1 - (avgScore / baseScore);
  }
  
  /**
   * Render the HTML for metrics table
   * @param {Array} tableData - Array of model metrics data
   * @return {string} HTML for the metrics table
   */
  renderMetricsTableHtml(tableData) {
    return `
      <h4>Model Performance Metrics</h4>
      <table class="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Base Score</th>
            <th>Average Score</th>
            <th>Worst Score</th>
            <th>Impact (%)</th>
            <th>Robustness Rating</th>
          </tr>
        </thead>
        <tbody>
          ${tableData.map(row => this.renderTableRow(row)).join('')}
        </tbody>
      </table>
      <p class="mt-4"><small>Robustness rating is calculated based on the average impact of perturbations on model performance.</small></p>
    `;
  }
  
  /**
   * Render a table row for a model
   * @param {Object} rowData - Data for the model row
   * @return {string} HTML for the table row
   */
  renderTableRow(rowData) {
    return `
      <tr>
        <td>${rowData.model}</td>
        <td>${formatNumber(rowData.baseScore)}</td>
        <td>${formatNumber(rowData.avgScore)}</td>
        <td>${formatNumber(rowData.worstScore)}</td>
        <td>${formatPercent(rowData.impact)}</td>
        <td>${this.getRobustnessRating(1 - rowData.impact)}</td>
      </tr>
    `;
  }
  
  /**
   * Get a robustness rating based on the robustness score
   * @param {number} score - The robustness score (0-1)
   * @return {string} HTML for the rating
   */
  getRobustnessRating(score) {
    if (score >= 0.9) {
      return '<span class="rating excellent">Excellent</span>';
    } else if (score >= 0.8) {
      return '<span class="rating good">Good</span>';
    } else if (score >= 0.7) {
      return '<span class="rating fair">Fair</span>';
    } else if (score >= 0.6) {
      return '<span class="rating moderate">Moderate</span>';
    } else {
      return '<span class="rating poor">Poor</span>';
    }
  }
}

export default MetricsTable;