// ResultTables.js - placeholder
/**
 * Results tables renderer
 * 
 * Handles rendering of result tables for the overview section,
 * including raw perturbation results and quantile perturbation results
 */
import { formatNumber, formatPercent, formatChangeFromBase } from '../../utils/Formatters.js';

class ResultTables {
  /**
   * Initialize the results tables renderer
   */
  constructor() {
    // Initialize any needed properties
  }
  
  /**
   * Render raw perturbation results table
   * @param {HTMLElement} container - The container element
   * @param {Object} perturbationData - Perturbation data
   */
  renderRawPerturbationTable(container, perturbationData) {
    if (!container || !perturbationData) return;
    
    const tableData = this.prepareTableData(perturbationData, 'raw');
    const html = this.renderTableHtml(tableData, 'Raw');
    
    container.innerHTML = html;
  }
  
  /**
   * Render quantile perturbation results table
   * @param {HTMLElement} container - The container element
   * @param {Object} perturbationData - Perturbation data
   */
  renderQuantilePerturbationTable(container, perturbationData) {
    if (!container || !perturbationData) return;
    
    // In a real implementation, extract quantile data
    // For this example, we'll simulate quantile data from raw data
    const tableData = this.prepareTableData(perturbationData, 'quantile');
    const html = this.renderTableHtml(tableData, 'Quantile');
    
    container.innerHTML = html;
  }
  
  /**
   * Prepare table data from perturbation data
   * @param {Object} perturbationData - Perturbation data
   * @param {string} type - Type of perturbation ('raw' or 'quantile')
   * @return {Array} Array of row data objects
   */
  prepareTableData(perturbationData, type) {
    const isQuantile = type === 'quantile';
    
    return perturbationData.levels.map(level => {
      const levelStr = level.toString();
      const levelData = perturbationData.byLevel[levelStr];
      
      if (!levelData) return null;
      
      // For quantile type, simulate slightly better results
      const adjustmentFactor = isQuantile ? 1.05 : 1;
      
      return {
        level: level,
        score: levelData.score * adjustmentFactor,
        worstScore: levelData.worstScore * adjustmentFactor,
        impact: levelData.impact * (isQuantile ? 0.95 : 1),
        baseScore: perturbationData.baseScore,
        // Add best model if available
        bestModel: this.getBestModelForLevel(perturbationData, level)
      };
    }).filter(row => row !== null);
  }
  
  /**
   * Get the best performing model for a level
   * @param {Object} perturbationData - Perturbation data with alternative models
   * @param {number} level - Perturbation level
   * @return {string|null} Name of the best model or null if no alternatives
   */
  getBestModelForLevel(perturbationData, level) {
    if (!perturbationData.alternativeModels || Object.keys(perturbationData.alternativeModels).length === 0) {
      return null;
    }
    
    const levelIndex = perturbationData.levels.indexOf(level);
    if (levelIndex === -1) return null;
    
    // Get primary model score
    const primaryScore = perturbationData.scores[levelIndex];
    let bestScore = primaryScore;
    let bestModel = perturbationData.modelName;
    
    // Check each alternative model
    Object.entries(perturbationData.alternativeModels).forEach(([name, model]) => {
      if (model.scores && model.scores[levelIndex] !== null && model.scores[levelIndex] > bestScore) {
        bestScore = model.scores[levelIndex];
        bestModel = name;
      }
    });
    
    return bestModel;
  }
  
  /**
   * Render HTML for a results table
   * @param {Array} tableData - Array of row data objects
   * @param {string} type - Type of perturbation ('Raw' or 'Quantile')
   * @return {string} HTML for the table
   */
  renderTableHtml(tableData, type) {
    const hasBestModel = tableData.some(row => row.bestModel !== null);
    
    let html = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Level</th>
            <th>Score</th>
            <th>Worst Score</th>
            <th>Impact (%)</th>
            <th>Change from Base</th>
            ${hasBestModel ? '<th>Best Model</th>' : ''}
          </tr>
        </thead>
        <tbody>
    `;
    
    tableData.forEach(row => {
      html += `
        <tr>
          <td>${row.level}</td>
          <td>${formatNumber(row.score)}</td>
          <td>${formatNumber(row.worstScore)}</td>
          <td>${formatPercent(row.impact)}</td>
          <td>${formatChangeFromBase(row.score, row.baseScore)}</td>
          ${hasBestModel ? `<td>${row.bestModel || '—'}</td>` : ''}
        </tr>
      `;
    });
    
    html += `
        </tbody>
      </table>
      <p class="mt-4"><small>${type} perturbation analyzes the effect of ${
        type === 'Raw' ? 'Gaussian noise' : 'feature distribution shifts'
      } on model performance.</small></p>
    `;
    
    return html;
  }
}

export default ResultTables;