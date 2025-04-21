/**
 * Feature importance table renderer
 * 
 * Handles rendering of feature importance tables for robustness importance,
 * model importance, and feature comparison
 */
import { formatNumber, formatSubsetIndicator } from '../../utils/Formatters.js';

class TableRenderers {
  /**
   * Initialize the table renderers
   */
  constructor() {
    // Initialize any needed properties
  }
  
  /**
   * Render the robustness importance table
   * @param {HTMLElement} container - The container element
   * @param {Object} featureData - Feature importance data
   */
  renderRobustnessImportanceTable(container, featureData) {
    if (!container || !featureData || 
        !featureData.featureImportance || Object.keys(featureData.featureImportance).length === 0) {
      this.showNoDataMessage(container, 'Robustness feature importance data not available');
      return;
    }
    
    // Prepare data for table - convert object to sorted array
    const featureItems = Object.entries(featureData.featureImportance)
      .map(([feature, importance]) => ({
        feature,
        importance,
        inSubset: featureData.featureSubset.includes(feature)
      }))
      .sort((a, b) => b.importance - a.importance);
    
    let html = `
      <h4>Robustness Feature Importance Ranking</h4>
      <table class="feature-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Robustness Importance Score</th>
            <th>Rank</th>
            <th>In Optimal Subset</th>
          </tr>
        </thead>
        <tbody>
    `;
    
    featureItems.forEach((item, index) => {
      const rowClass = item.inSubset ? 'class="highlight-row"' : '';
      
      html += `
        <tr ${rowClass}>
          <td>${item.feature}</td>
          <td>${formatNumber(item.importance)}</td>
          <td>${index + 1}</td>
          <td>${formatSubsetIndicator(item.inSubset)}</td>
        </tr>
      `;
    });
    
    html += `
        </tbody>
      </table>
      <p class="mt-4"><small>Features with higher robustness importance scores have greater impact on model behavior when perturbed. 
      Features in the optimal subset are highlighted.</small></p>
    `;
    
    container.innerHTML = html;
  }
  
  /**
   * Render the model importance table
   * @param {HTMLElement} container - The container element
   * @param {Object} featureData - Feature importance data
   */
  renderModelImportanceTable(container, featureData) {
    if (!container || !featureData || 
        !featureData.modelFeatureImportance || Object.keys(featureData.modelFeatureImportance).length === 0) {
      this.showNoDataMessage(container, 'Model feature importance data not available');
      return;
    }
    
    // Prepare data for table - convert object to sorted array
    const featureItems = Object.entries(featureData.modelFeatureImportance)
      .map(([feature, importance]) => ({
        feature,
        importance,
        inSubset: featureData.featureSubset.includes(feature)
      }))
      .sort((a, b) => b.importance - a.importance);
    
    let html = `
      <h4>Model Feature Importance Ranking</h4>
      <table class="feature-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Model Importance Score</th>
            <th>Rank</th>
            <th>In Optimal Subset</th>
          </tr>
        </thead>
        <tbody>
    `;
    
    featureItems.forEach((item, index) => {
      const rowClass = item.inSubset ? 'class="highlight-row"' : '';
      
      html += `
        <tr ${rowClass}>
          <td>${item.feature}</td>
          <td>${formatNumber(item.importance)}</td>
          <td>${index + 1}</td>
          <td>${formatSubsetIndicator(item.inSubset)}</td>
        </tr>
      `;
    });
    
    html += `
        </tbody>
      </table>
      <p class="mt-4"><small>Features with higher model importance scores have greater impact on the model's predictions according to the model's internal metrics. 
      Features in the optimal subset are highlighted.</small></p>
    `;
    
    container.innerHTML = html;
  }
  
  /**
   * Render comparison table
   * @param {HTMLElement} container - The container element
   * @param {Object} featureData - Feature importance data
   */
  renderComparisonTable(container, featureData) {
    if (!container || !featureData || 
        !featureData.featureImportance || !featureData.modelFeatureImportance) {
      this.showNoDataMessage(container, 'Feature importance comparison data not available');
      return;
    }
    
    // Get combined data for common features
    const combinedData = this.getCombinedFeatureData(featureData);
    
    // Calculate rankings
    const withRankings = combinedData.map(item => ({
      ...item,
      modelRank: 0,
      robustnessRank: 0
    }));
    
    // Assign model ranks
    withRankings
      .sort((a, b) => b.modelImportance - a.modelImportance)
      .forEach((item, index) => {
        item.modelRank = index + 1;
      });
    
    // Assign robustness ranks
    withRankings
      .sort((a, b) => b.robustnessImportance - a.robustnessImportance)
      .forEach((item, index) => {
        item.robustnessRank = index + 1;
      });
    
    // Calculate rank differences
    const withDifferences = withRankings.map(item => ({
      ...item,
      rankDiff: Math.abs(item.modelRank - item.robustnessRank)
    }));
    
    // Sort by model importance for the table
    const sortedData = withDifferences.sort((a, b) => b.modelImportance - a.modelImportance);
    
    let html = `
      <h4>Feature Importance Comparison Table</h4>
      <table class="feature-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Model Importance</th>
            <th>Robustness Importance</th>
            <th>Model Rank</th>
            <th>Robustness Rank</th>
            <th>Rank Difference</th>
            <th>In Optimal Subset</th>
          </tr>
        </thead>
        <tbody>
    `;
    
    sortedData.forEach(item => {
      const rowClass = item.inSubset ? 'class="highlight-row"' : '';
      
      html += `
        <tr ${rowClass}>
          <td>${item.feature}</td>
          <td>${formatNumber(item.modelImportance)}</td>
          <td>${formatNumber(item.robustnessImportance)}</td>
          <td>${item.modelRank}</td>
          <td>${item.robustnessRank}</td>
          <td>${item.rankDiff}</td>
          <td>${formatSubsetIndicator(item.inSubset)}</td>
        </tr>
      `;
    });
    
    html += `
        </tbody>
      </table>
      <p class="mt-4">
        <small>
          Features with large rank differences indicate discrepancies between model importance and robustness importance.
          <br>Features in the optimal subset (highlighted) represent the best balance between predictive power and robustness.
        </small>
      </p>
    `;
    
    container.innerHTML = html;
  }
  
  /**
   * Get combined feature data for features present in both importance sets
   * @param {Object} featureData - Feature importance data
   * @return {Array} Array of combined feature data objects
   */
  getCombinedFeatureData(featureData) {
    const featureImportance = featureData.featureImportance || {};
    const modelFeatureImportance = featureData.modelFeatureImportance || {};
    const featureSubset = featureData.featureSubset || [];
    
    // Find common features between both sets
    const allFeatures = new Set([
      ...Object.keys(featureImportance),
      ...Object.keys(modelFeatureImportance)
    ]);
    
    // Combine the data
    return Array.from(allFeatures).map(feature => ({
      feature,
      robustnessImportance: featureImportance[feature] || 0,
      modelImportance: modelFeatureImportance[feature] || 0,
      inSubset: featureSubset.includes(feature)
    }));
  }
  
  /**
   * Show a no data message in a container
   * @param {HTMLElement} container - The container element
   * @param {string} message - The message to display
   */
  showNoDataMessage(container, message) {
    if (!container) return;
    
    container.innerHTML = `
      <div class="alert alert-info">
        <strong>No data available</strong><br>
        ${message}
      </div>
    `;
  }
}

export default TableRenderers;