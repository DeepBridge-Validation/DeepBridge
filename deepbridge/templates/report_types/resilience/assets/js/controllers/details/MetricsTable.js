// MetricsTable.js - placeholder
/**
 * Component for rendering metrics tables
 * 
 * Handles the rendering of model performance metrics tables,
 * including summary metrics and detailed performance analysis
 */
class MetricsTable {
    /**
     * Initialize the metrics table renderer
     */
    constructor() {
        // Default formatting options
        this.defaultOptions = {
            precision: 4,    // Decimal places for numeric results
            percentPrecision: 2  // Decimal places for percent values
        };
    }
    
    /**
     * Render metrics table
     * @param {HTMLElement} container - The container element
     * @param {Object} shiftData - Distribution shift data
     */
    render(container, shiftData) {
        if (!container || !shiftData) {
            return;
        }
        
        // Create HTML for metrics table
        const html = this.createMetricsTableHtml(shiftData);
        container.innerHTML = html;
        
        // Initialize any interactive elements
        this.initializeInteractiveElements(container);
    }
    
    /**
     * Create HTML for metrics table
     * @param {Object} shiftData - Distribution shift data
     * @return {string} HTML for metrics table
     */
    createMetricsTableHtml(shiftData) {
        // Extract metrics and performance data
        const baseScore = shiftData.baseScore || 0;
        const resilienceScore = shiftData.resilienceScore || 0;
        const shiftResults = shiftData.shiftResults || [];
        
        // Calculate summary metrics
        const avgShiftScore = this.calculateAvgShiftScore(shiftResults);
        const worstShiftScore = this.calculateWorstShiftScore(shiftResults);
        const avgPerformanceGap = this.calculateAvgPerformanceGap(shiftResults, baseScore);
        const maxPerformanceGap = this.calculateMaxPerformanceGap(shiftResults, baseScore);
        
        // Format percentages
        const resilienceScorePercent = this.formatPercent(resilienceScore);
        const avgPerformanceGapPercent = this.formatPercent(avgPerformanceGap);
        const maxPerformanceGapPercent = this.formatPercent(maxPerformanceGap);
        
        return `
            <div class="metrics-details-wrapper">
                <h4>Model Performance Metrics</h4>
                <table class="data-table metrics-details-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Base Score</td>
                            <td>${this.formatNumber(baseScore)}</td>
                            <td>Model score on original distribution</td>
                        </tr>
                        <tr>
                            <td>Resilience Score</td>
                            <td>${resilienceScorePercent}</td>
                            <td>Overall measure of model resilience to distribution shifts</td>
                        </tr>
                        <tr>
                            <td>Average Shift Score</td>
                            <td>${this.formatNumber(avgShiftScore)}</td>
                            <td>Average performance across all distribution shifts</td>
                        </tr>
                        <tr>
                            <td>Worst Shift Score</td>
                            <td>${this.formatNumber(worstShiftScore)}</td>
                            <td>Worst performance across all distribution shifts</td>
                        </tr>
                        <tr>
                            <td>Average Performance Gap</td>
                            <td>${avgPerformanceGapPercent}</td>
                            <td>Average drop in performance under distribution shifts</td>
                        </tr>
                        <tr>
                            <td>Maximum Performance Gap</td>
                            <td>${maxPerformanceGapPercent}</td>
                            <td>Maximum drop in performance under distribution shifts</td>
                        </tr>
                    </tbody>
                </table>
                
                <div class="metrics-analysis mt-4">
                    <h4>Performance Analysis</h4>
                    ${this.createPerformanceAnalysisHtml(shiftData)}
                </div>
            </div>
        `;
    }
    
    /**
     * Create HTML for performance analysis
     * @param {Object} shiftData - Distribution shift data
     * @return {string} HTML for performance analysis
     */
    createPerformanceAnalysisHtml(shiftData) {
        const resilienceScore = shiftData.resilienceScore || 0;
        let analysisHtml = '';
        
        if (resilienceScore >= 0.9) {
            analysisHtml = `
                <p>The model demonstrates <strong class="excellent">excellent resilience</strong> to distribution shifts, 
                maintaining consistent performance across different data distributions. The high resilience score indicates 
                that the model generalizes well beyond its training distribution.</p>
            `;
        } else if (resilienceScore >= 0.8) {
            analysisHtml = `
                <p>The model shows <strong class="good">good resilience</strong> to distribution shifts, 
                with only minor performance degradation across different data distributions. The model 
                maintains reliable performance in most shifted scenarios.</p>
            `;
        } else if (resilienceScore >= 0.7) {
            analysisHtml = `
                <p>The model demonstrates <strong class="moderate">moderate resilience</strong> to distribution shifts. 
                While it maintains acceptable performance across most shifts, there is noticeable degradation 
                in some scenarios that may require attention.</p>
            `;
        } else if (resilienceScore >= 0.6) {
            analysisHtml = `
                <p>The model shows <strong class="fair">fair resilience</strong> to distribution shifts, 
                with significant performance degradation in some scenarios. Improvements to model robustness 
                should be considered, especially for critical applications.</p>
            `;
        } else {
            analysisHtml = `
                <p>The model demonstrates <strong class="poor">limited resilience</strong> to distribution shifts, 
                with substantial performance degradation across many scenarios. Consider retraining with more 
                diverse data or implementing specific robustness techniques.</p>
            `;
        }
        
        // Add shift-specific insights if available
        if (shiftData.insights && shiftData.insights.length > 0) {
            analysisHtml += `
                <h5 class="mt-3">Key Insights:</h5>
                <ul class="metrics-insights">
                    ${shiftData.insights.map(insight => `<li>${insight}</li>`).join('')}
                </ul>
            `;
        }
        
        return analysisHtml;
    }
    
    /**
     * Initialize interactive elements in the metrics table
     * @param {HTMLElement} container - The container element
     */
    initializeInteractiveElements(container) {
        // Add any event listeners or interactivity to the table
        // For example, tooltips, expandable sections, etc.
    }
    
    /**
     * Calculate average shift score
     * @param {Array} shiftResults - Array of shift result objects
     * @return {number} Average shift score
     */
    calculateAvgShiftScore(shiftResults) {
        if (!shiftResults || shiftResults.length === 0) {
            return 0;
        }
        
        const scores = shiftResults.map(result => result.score).filter(score => !isNaN(score));
        if (scores.length === 0) {
            return 0;
        }
        
        return scores.reduce((sum, score) => sum + score, 0) / scores.length;
    }
    
    /**
     * Calculate worst shift score
     * @param {Array} shiftResults - Array of shift result objects
     * @return {number} Worst shift score
     */
    calculateWorstShiftScore(shiftResults) {
        if (!shiftResults || shiftResults.length === 0) {
            return 0;
        }
        
        const scores = shiftResults.map(result => result.score).filter(score => !isNaN(score));
        if (scores.length === 0) {
            return 0;
        }
        
        return Math.min(...scores);
    }
    
    /**
     * Calculate average performance gap
     * @param {Array} shiftResults - Array of shift result objects
     * @param {number} baseScore - Base score for comparison
     * @return {number} Average performance gap
     */
    calculateAvgPerformanceGap(shiftResults, baseScore) {
        if (!shiftResults || shiftResults.length === 0 || !baseScore) {
            return 0;
        }
        
        const gaps = shiftResults.map(result => (baseScore - result.score) / baseScore)
            .filter(gap => !isNaN(gap));
            
        if (gaps.length === 0) {
            return 0;
        }
        
        return gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length;
    }
    
    /**
     * Calculate maximum performance gap
     * @param {Array} shiftResults - Array of shift result objects
     * @param {number} baseScore - Base score for comparison
     * @return {number} Maximum performance gap
     */
    calculateMaxPerformanceGap(shiftResults, baseScore) {
        if (!shiftResults || shiftResults.length === 0 || !baseScore) {
            return 0;
        }
        
        const gaps = shiftResults.map(result => (baseScore - result.score) / baseScore)
            .filter(gap => !isNaN(gap));
            
        if (gaps.length === 0) {
            return 0;
        }
        
        return Math.max(...gaps);
    }
    
    /**
     * Format a number with specified precision
     * @param {number} value - Number to format
     * @param {number} precision - Decimal precision (optional)
     * @return {string} Formatted number
     */
    formatNumber(value, precision) {
        if (value === undefined || value === null || isNaN(value)) {
            return 'N/A';
        }
        
        const p = precision !== undefined ? precision : this.defaultOptions.precision;
        return value.toFixed(p);
    }
    
    /**
     * Format a value as a percentage
     * @param {number} value - Value to format (0-1)
     * @param {number} precision - Decimal precision (optional)
     * @return {string} Formatted percentage
     */
    formatPercent(value, precision) {
        if (value === undefined || value === null || isNaN(value)) {
            return 'N/A';
        }
        
        const p = precision !== undefined ? precision : this.defaultOptions.percentPrecision;
        return (value * 100).toFixed(p) + '%';
    }
}

export default MetricsTable;