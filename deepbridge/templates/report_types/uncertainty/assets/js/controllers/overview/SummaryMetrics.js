// SummaryMetrics.js - placeholder
/**
 * Summary metrics renderer
 * 
 * Handles rendering of summary metrics and key performance 
 * indicators for the uncertainty quantification model
 */
class SummaryMetrics {
    /**
     * Initialize the summary metrics renderer
     */
    constructor() {
      // Initialize any needed properties
    }
    
    /**
     * Render uncertainty summary metrics
     * @param {HTMLElement} container - The container element
     * @param {Object} uncertaintyData - The uncertainty data
     */
    render(container, uncertaintyData) {
      if (!container || !uncertaintyData) {
        return;
      }
      
      const summary = uncertaintyData.summary || {};
      
      let html = `
        <div class="metrics-overview">
          <h3>Uncertainty Quantification Performance</h3>
          <p class="metrics-description">
            ${this.generateSummaryDescription(summary)}
          </p>
          
          <div class="metrics-grid">
            ${this.renderMetricCards(summary)}
          </div>
        </div>
        
        <div class="key-findings">
          <h3>Key Findings</h3>
          ${this.renderKeyFindings(uncertaintyData)}
        </div>
      `;
      
      container.innerHTML = html;
    }
    
    /**
     * Generate a descriptive summary based on metrics
     * @param {Object} summary - Summary metrics
     * @return {string} Description text
     */
    generateSummaryDescription(summary) {
      const uncertaintyScore = summary.uncertaintyScore || 0;
      const averageCoverage = summary.averageCoverage || 0;
      const expectedCoverage = summary.expectedCoverage || 0.9;
      const averageWidth = summary.averageWidth || 0;
      
      let description = 'This overview presents the performance of the uncertainty quantification model. ';
      
      // Add description based on uncertainty score
      if (uncertaintyScore >= 0.9) {
        description += 'The model demonstrates excellent uncertainty quantification with well-calibrated prediction intervals. ';
      } else if (uncertaintyScore >= 0.8) {
        description += 'The model shows good uncertainty quantification capabilities with generally reliable prediction intervals. ';
      } else if (uncertaintyScore >= 0.7) {
        description += 'The model provides moderately effective uncertainty quantification with reasonably calibrated prediction intervals. ';
      } else if (uncertaintyScore >= 0.6) {
        description += 'The model exhibits fair uncertainty quantification with some calibration issues in its prediction intervals. ';
      } else {
        description += 'The model has limited uncertainty quantification performance with poorly calibrated prediction intervals. ';
      }
      
      // Add coverage comparison
      const coverageDiff = Math.abs(averageCoverage - expectedCoverage);
      if (coverageDiff <= 0.01) {
        description += 'Coverage matches the expected confidence levels exceptionally well. ';
      } else if (coverageDiff <= 0.05) {
        description += 'Coverage is reasonably close to the expected confidence levels. ';
      } else if (coverageDiff <= 0.1) {
        description += 'Coverage shows moderate deviation from expected confidence levels. ';
      } else {
        description += 'Coverage significantly deviates from expected confidence levels. ';
      }
      
      return description;
    }
    
    /**
     * Render metric cards for summary data
     * @param {Object} summary - Summary metrics
     * @return {string} HTML for metric cards
     */
    renderMetricCards(summary) {
      let html = '';
      
      // Uncertainty Score
      html += this.createMetricCard(
        'Uncertainty Score',
        summary.uncertaintyScore,
        'Overall quality of uncertainty quantification',
        this.getScoreColorClass(summary.uncertaintyScore)
      );
      
      // Average Coverage
      html += this.createMetricCard(
        'Average Coverage',
        summary.averageCoverage,
        'Fraction of test samples within prediction intervals',
        this.getCoverageColorClass(summary.averageCoverage, summary.expectedCoverage)
      );
      
      // Average Width
      html += this.createMetricCard(
        'Average Width',
        summary.averageWidth,
        'Average prediction interval width',
        'neutral',
        false
      );
      
      // Calibration Error
      if (summary.expectedCalibrationError !== undefined) {
        html += this.createMetricCard(
          'Calibration Error',
          summary.expectedCalibrationError,
          'Average discrepancy between expected and observed coverage',
          this.getErrorColorClass(summary.expectedCalibrationError),
          true,
          true
        );
      }
      
      return html;
    }
    
    /**
     * Create a single metric card
     * @param {string} title - Metric title
     * @param {number} value - Metric value
     * @param {string} description - Metric description
     * @param {string} colorClass - Color class for the metric
     * @param {boolean} isPercentage - Whether to format as percentage
     * @param {boolean} lowerIsBetter - Whether lower values are better
     * @return {string} HTML for the metric card
     */
    createMetricCard(title, value, description, colorClass, isPercentage = true, lowerIsBetter = false) {
      const formattedValue = value !== undefined ? 
        (isPercentage ? `${(value * 100).toFixed(1)}%` : value.toFixed(4)) : 
        'N/A';
      
      return `
        <div class="metric-card">
          <div class="metric-icon">
            <i class="metric-icon-${colorClass}"></i>
          </div>
          <div class="metric-content">
            <div class="metric-title">${title}</div>
            <div class="metric-value ${colorClass}">${formattedValue}</div>
            <div class="metric-description">${description}</div>
          </div>
          <div class="metric-indicator">
            <span class="indicator ${colorClass}"></span>
          </div>
        </div>
      `;
    }
    
    /**
     * Render key findings based on uncertainty data
     * @param {Object} uncertaintyData - The uncertainty data
     * @return {string} HTML for key findings
     */
    renderKeyFindings(uncertaintyData) {
      const summary = uncertaintyData.summary || {};
      const alphaLevels = uncertaintyData.alphaLevels || [];
      
      // Finding 1: Best performing confidence level
      let bestAlpha = null;
      let bestCoverageDiff = Infinity;
      
      alphaLevels.forEach(alpha => {
        const levelData = uncertaintyData.byAlpha[alpha] || {};
        const expectedCoverage = 1 - alpha;
        const observedCoverage = levelData.coverage || 0;
        const coverageDiff = Math.abs(observedCoverage - expectedCoverage);
        
        if (coverageDiff < bestCoverageDiff) {
          bestCoverageDiff = coverageDiff;
          bestAlpha = alpha;
        }
      });
      
      // Finding 2: Most challenging confidence level
      let worstAlpha = null;
      let worstCoverageDiff = -Infinity;
      
      alphaLevels.forEach(alpha => {
        const levelData = uncertaintyData.byAlpha[alpha] || {};
        const expectedCoverage = 1 - alpha;
        const observedCoverage = levelData.coverage || 0;
        const coverageDiff = Math.abs(observedCoverage - expectedCoverage);
        
        if (coverageDiff > worstCoverageDiff) {
          worstCoverageDiff = coverageDiff;
          worstAlpha = alpha;
        }
      });
      
      // Finding 3: Width assessment
      const widthAssessment = this.assessIntervalWidths(uncertaintyData);
      
      // Compile findings
      let html = '<ul class="findings-list">';
      
      if (bestAlpha !== null) {
        html += `
          <li class="finding-item">
            <span class="finding-highlight">Best calibration achieved</span> at confidence level ${((1 - bestAlpha) * 100).toFixed(0)}% 
            (α=${bestAlpha}) with a coverage difference of ${bestCoverageDiff.toFixed(4)}.
          </li>
        `;
      }
      
      if (worstAlpha !== null) {
        html += `
          <li class="finding-item">
            <span class="finding-highlight">Most challenging calibration</span> at confidence level ${((1 - worstAlpha) * 100).toFixed(0)}% 
            (α=${worstAlpha}) with a coverage difference of ${worstCoverageDiff.toFixed(4)}.
          </li>
        `;
      }
      
      html += `
        <li class="finding-item">
          <span class="finding-highlight">Prediction interval widths:</span> ${widthAssessment}
        </li>
      `;
      
      // Add overall assessment
      const overallAssessment = this.getOverallAssessment(summary);
      html += `
        <li class="finding-item">
          <span class="finding-highlight">Overall assessment:</span> ${overallAssessment}
        </li>
      `;
      
      html += '</ul>';
      return html;
    }
    
    /**
     * Assess interval widths
     * @param {Object} uncertaintyData - The uncertainty data
     * @return {string} Assessment text
     */
    assessIntervalWidths(uncertaintyData) {
      const alphaLevels = uncertaintyData.alphaLevels || [];
      const widths = [];
      
      alphaLevels.forEach(alpha => {
        const levelData = uncertaintyData.byAlpha[alpha] || {};
        if (levelData.averageWidth !== undefined) {
          widths.push(levelData.averageWidth);
        }
      });
      
      if (widths.length === 0) {
        return 'No width data available.';
      }
      
      const avgWidth = widths.reduce((sum, width) => sum + width, 0) / widths.length;
      
      if (avgWidth < 0.1) {
        return 'Prediction intervals are very narrow, indicating high precision.';
      } else if (avgWidth < 0.2) {
        return 'Prediction intervals show good precision with moderate widths.';
      } else if (avgWidth < 0.3) {
        return 'Prediction intervals have medium width, balancing precision and coverage.';
      } else if (avgWidth < 0.4) {
        return 'Prediction intervals are somewhat wide, prioritizing coverage over precision.';
      } else {
        return 'Prediction intervals are quite wide, indicating conservative uncertainty estimates.';
      }
    }
    
    /**
     * Get overall assessment based on summary metrics
     * @param {Object} summary - Summary metrics
     * @return {string} Assessment text
     */
    getOverallAssessment(summary) {
      const uncertaintyScore = summary.uncertaintyScore || 0;
      
      if (uncertaintyScore >= 0.9) {
        return 'The model provides excellent uncertainty quantification with well-calibrated prediction intervals, suitable for critical applications.';
      } else if (uncertaintyScore >= 0.8) {
        return 'The model demonstrates good uncertainty quantification capabilities, reliable for most applications.';
      } else if (uncertaintyScore >= 0.7) {
        return 'The model shows moderate uncertainty quantification performance, acceptable for many applications but with some limitations.';
      } else if (uncertaintyScore >= 0.6) {
        return 'The model exhibits fair uncertainty quantification with noticeable calibration issues, requiring cautious interpretation.';
      } else {
        return 'The model has limited uncertainty quantification capabilities, requiring significant improvement before deployment in applications where uncertainty matters.';
      }
    }
    
    /**
     * Get color class for a score
     * @param {number} score - Score value (0-1)
     * @return {string} Color class name
     */
    getScoreColorClass(score) {
      if (score >= 0.9) return 'excellent';
      if (score >= 0.8) return 'good';
      if (score >= 0.7) return 'moderate';
      if (score >= 0.6) return 'fair';
      return 'poor';
    }
    
    /**
     * Get color class for coverage
     * @param {number} observed - Observed coverage
     * @param {number} expected - Expected coverage
     * @return {string} Color class name
     */
    getCoverageColorClass(observed, expected) {
      const diff = Math.abs(observed - expected);
      
      if (diff <= 0.01) return 'excellent';
      if (diff <= 0.05) return 'good';
      if (diff <= 0.1) return 'moderate';
      if (diff <= 0.15) return 'fair';
      return 'poor';
    }
    
    /**
     * Get color class for error metrics (lower is better)
     * @param {number} error - Error value
     * @return {string} Color class name
     */
    getErrorColorClass(error) {
      if (error <= 0.01) return 'excellent';
      if (error <= 0.05) return 'good';
      if (error <= 0.1) return 'moderate';
      if (error <= 0.15) return 'fair';
      return 'poor';
    }
  }
  
  export default SummaryMetrics;