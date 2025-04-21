// SummaryTables.js - placeholder
/**
 * Summary tables renderer
 * 
 * Handles rendering of summary tables for the overview section,
 * including shift results and metrics details
 */

class SummaryTables {
    /**
     * Initialize the summary tables renderer
     */
    constructor() {
      // Initialize any needed properties
    }
    
    /**
     * Render the shift results table
     * @param {HTMLElement} container - The container element
     * @param {Object} resilienceData - Resilience data with shift results
     */
    renderShiftResultsTable(container, resilienceData) {
      if (!container || !resilienceData) return;
      
      let html = `
        <h4>Shift Results Summary</h4>
        <table class="data-table">
          <thead>
            <tr>
              <th>Shift Type</th>
              <th>Score</th>
              <th>Base Score</th>
              <th>Gap</th>
              <th>Impact (%)</th>
              <th>Intensity</th>
            </tr>
          </thead>
          <tbody>
      `;
      
      // Add a row for each shift
      resilienceData.shifts.forEach(shift => {
        const gap = resilienceData.baseScore - shift.score;
        const impact = (gap / resilienceData.baseScore) * 100;
        
        html += `
          <tr>
            <td>${shift.type}</td>
            <td>${this.formatNumber(shift.score)}</td>
            <td>${this.formatNumber(resilienceData.baseScore)}</td>
            <td>${this.formatNumber(gap)}</td>
            <td>${this.formatNumber(impact, 2)}%</td>
            <td>${this.formatIntensity(shift.intensity || 0.5)}</td>
          </tr>
        `;
      });
      
      html += `
          </tbody>
        </table>
        <p class="mt-4"><small>Performance gaps show how much model performance drops under different distribution shifts. 
        Lower values indicate higher resilience.</small></p>
      `;
      
      container.innerHTML = html;
    }
    
    /**
     * Render the metrics details table
     * @param {HTMLElement} container - The container element
     * @param {Object} resilienceData - Resilience data with metrics details
     */
    renderMetricsDetailsTable(container, resilienceData) {
      if (!container || !resilienceData) return;
      
      // Calculate aggregate metrics
      const avgGap = this.calculateAverageGap(resilienceData);
      const maxGap = this.calculateMaxGap(resilienceData);
      const resilienceScore = resilienceData.resilienceScore || 
                            this.calculateResilienceScore(resilienceData);
      
      let html = `
        <h4>Resilience Metrics Details</h4>
        <table class="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
              <th>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Resilience Score</td>
              <td>${this.formatNumber(resilienceScore)}</td>
              <td>${this.getResilienceInterpretation(resilienceScore)}</td>
            </tr>
            <tr>
              <td>Average Performance Gap</td>
              <td>${this.formatNumber(avgGap)}</td>
              <td>${this.getGapInterpretation(avgGap)}</td>
            </tr>
            <tr>
              <td>Maximum Performance Gap</td>
              <td>${this.formatNumber(maxGap)}</td>
              <td>${this.getGapInterpretation(maxGap)}</td>
            </tr>
            <tr>
              <td>Base Performance</td>
              <td>${this.formatNumber(resilienceData.baseScore)}</td>
              <td>Baseline score on original distribution</td>
            </tr>
            <tr>
              <td>Worst Shift Type</td>
              <td>${this.getWorstShiftType(resilienceData)}</td>
              <td>Shift type with the largest performance gap</td>
            </tr>
          </tbody>
        </table>
      `;
      
      container.innerHTML = html;
    }
    
    /**
     * Calculate average performance gap
     * @param {Object} resilienceData - Resilience data with shift results
     * @return {number} Average gap
     */
    calculateAverageGap(resilienceData) {
      if (!resilienceData || !resilienceData.shifts || resilienceData.shifts.length === 0) return 0;
      
      const baseScore = resilienceData.baseScore || 0;
      const totalGap = resilienceData.shifts.reduce((sum, shift) => sum + (baseScore - shift.score), 0);
      return totalGap / resilienceData.shifts.length;
    }
    
    /**
     * Calculate maximum performance gap
     * @param {Object} resilienceData - Resilience data with shift results
     * @return {number} Maximum gap
     */
    calculateMaxGap(resilienceData) {
      if (!resilienceData || !resilienceData.shifts || resilienceData.shifts.length === 0) return 0;
      
      const baseScore = resilienceData.baseScore || 0;
      const gaps = resilienceData.shifts.map(shift => baseScore - shift.score);
      return Math.max(...gaps);
    }
    
    /**
     * Calculate resilience score if not provided
     * @param {Object} resilienceData - Resilience data with shift results
     * @return {number} Resilience score (0-1)
     */
    calculateResilienceScore(resilienceData) {
      if (!resilienceData || !resilienceData.shifts || resilienceData.shifts.length === 0) return 0;
      
      const baseScore = resilienceData.baseScore || 0;
      if (baseScore === 0) return 0;
      
      // Calculate average normalized score (scores / baseScore)
      const avgNormalizedScore = resilienceData.shifts.reduce(
        (sum, shift) => sum + (shift.score / baseScore), 0
      ) / resilienceData.shifts.length;
      
      return avgNormalizedScore;
    }
    
    /**
     * Get the worst shift type
     * @param {Object} resilienceData - Resilience data with shift results
     * @return {string} Worst shift type
     */
    getWorstShiftType(resilienceData) {
      if (!resilienceData || !resilienceData.shifts || resilienceData.shifts.length === 0) return 'N/A';
      
      const baseScore = resilienceData.baseScore || 0;
      let worstShift = resilienceData.shifts[0];
      let worstGap = baseScore - worstShift.score;
      
      resilienceData.shifts.forEach(shift => {
        const gap = baseScore - shift.score;
        if (gap > worstGap) {
          worstGap = gap;
          worstShift = shift;
        }
      });
      
      return worstShift.type;
    }
    
    /**
     * Format a number for display
     * @param {number} value - The value to format
     * @param {number} decimals - Number of decimal places (default: 4)
     * @return {string} Formatted value
     */
    formatNumber(value, decimals = 4) {
      if (value === null || value === undefined) return 'N/A';
      return value.toFixed(decimals);
    }
    
    /**
     * Format intensity value
     * @param {number} intensity - Intensity value (0-1)
     * @return {string} Formatted intensity
     */
    formatIntensity(intensity) {
      if (intensity < 0.25) {
        return 'Low';
      } else if (intensity < 0.75) {
        return 'Medium';
      } else {
        return 'High';
      }
    }
    
    /**
     * Get interpretation text for resilience score
     * @param {number} score - Resilience score (0-1)
     * @return {string} Interpretation text
     */
    getResilienceInterpretation(score) {
      if (score >= 0.9) {
        return 'Excellent resilience';
      } else if (score >= 0.8) {
        return 'Good resilience';
      } else if (score >= 0.7) {
        return 'Moderate resilience';
      } else if (score >= 0.6) {
        return 'Fair resilience';
      } else {
        return 'Poor resilience';
      }
    }
    
    /**
     * Get interpretation text for performance gap
     * @param {number} gap - Performance gap
     * @return {string} Interpretation text
     */
    getGapInterpretation(gap) {
      if (gap < 0.05) {
        return 'Minimal performance degradation';
      } else if (gap < 0.15) {
        return 'Moderate performance degradation';
      } else if (gap < 0.25) {
        return 'Significant performance degradation';
      } else {
        return 'Severe performance degradation';
      }
    }
  }
  
  export default SummaryTables;