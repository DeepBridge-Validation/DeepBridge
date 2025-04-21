// DistanceMetrics.js - placeholder
/**
 * Component for rendering distance metrics visualizations
 * 
 * Handles the visualization and explanation of different distance metrics
 * used to quantify distribution shifts
 */
class DistanceMetrics {
    /**
     * Initialize the distance metrics renderer
     * @param {ChartFactory} chartFactory - Factory for creating charts
     */
    constructor(chartFactory) {
        this.chartFactory = chartFactory;
        
        // Default formatting options
        this.defaultOptions = {
            precision: 4,    // Decimal places for numeric results
            percentPrecision: 2  // Decimal places for percent values
        };
        
        // Chart configuration
        this.chartConfig = {
            height: 350,
            colors: [
                '#1b78de', '#2ecc71', '#f39c12', '#e74c3c', 
                '#9b59b6', '#3498db', '#1abc9c', '#f1c40f'
            ]
        };
    }
    
    /**
     * Render distance metrics visualizations
     * @param {HTMLElement} container - The container element
     * @param {Object} distributionData - Distribution shift data
     */
    render(container, distributionData) {
        if (!container || !distributionData) {
            return;
        }
        
        // Create HTML structure
        const html = this.createDistanceMetricsHtml(distributionData);
        container.innerHTML = html;
        
        // Render distance correlation chart
        this.renderDistanceCorrelationChart(
            container.querySelector('#distance-correlation-chart'), 
            distributionData
        );
        
        // Initialize interactive elements
        this.initializeInteractiveElements(container);
    }
    
    /**
     * Create HTML structure for distance metrics
     * @param {Object} distributionData - Distribution shift data
     * @return {string} HTML for distance metrics
     */
    createDistanceMetricsHtml(distributionData) {
        const distanceMetrics = distributionData.distanceMetrics || [];
        
        let metricsHtml = '';
        if (distanceMetrics.length > 0) {
            metricsHtml = `
                <div class="distance-metrics-info">
                    <h4>Distribution Distance Metrics</h4>
                    <p>The following metrics were used to quantify the difference between distributions:</p>
                    <div class="metrics-grid">
                        ${distanceMetrics.map(metric => `
                            <div class="metric-card">
                                <h5>${metric.name}</h5>
                                <p>${metric.description}</p>
                                ${metric.formula ? `<div class="metric-formula">${metric.formula}</div>` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        return `
            <div class="distance-metrics-wrapper">
                <h3>Distribution Distance Analysis</h3>
                <p>This section explores how different metrics capture the distance between 
                the original distribution and shifted distributions, and how these distances 
                correlate with model performance.</p>
                
                ${metricsHtml}
                
                <div id="distance-correlation-chart" class="chart-container mt-4">
                    <!-- Distance correlation chart will be rendered here -->
                </div>
                
                <div class="distribution-explanation mt-4">
                    <h4>Understanding Distribution Shifts</h4>
                    <p>Distribution shifts occur when the statistical properties of the input data 
                    change from what the model was trained on. These shifts can be measured using 
                    various distance metrics that quantify how different two distributions are.</p>
                    
                    <p>The correlation between distribution distance and model performance degradation 
                    provides insight into which types of shifts most affect the model.</p>
                </div>
            </div>
        `;
    }
    
    /**
     * Render distance correlation chart
     * @param {HTMLElement} container - The chart container element
     * @param {Object} distributionData - Distribution shift data
     */
    renderDistanceCorrelationChart(container, distributionData) {
        if (!container) {
            return;
        }
        
        const baseScore = distributionData.baseScore || 0;
        const shiftResults = distributionData.shiftResults || [];
        
        if (shiftResults.length === 0) {
            container.innerHTML = this.getNoDataHtml('No distance data available for chart');
            return;
        }
        
        // Calculate performance gap for each result
        const distances = shiftResults.map(result => result.distance || 0);
        const performanceGaps = shiftResults.map(result => {
            return (baseScore - result.score) / baseScore;
        });
        
        // Calculate regression line for trend
        const { slope, intercept } = this.calculateLinearRegression(distances, performanceGaps);
        
        // Create regression line data
        const minDistance = Math.min(...distances);
        const maxDistance = Math.max(...distances);
        const regressionLine = [
            { x: minDistance, y: slope * minDistance + intercept },
            { x: maxDistance, y: slope * maxDistance + intercept }
        ];
        
        // Create chart configuration
        const chartConfig = {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Performance Gap vs Distance',
                        data: shiftResults.map((result, index) => ({
                            x: result.distance || 0,
                            y: performanceGaps[index],
                            shiftType: result.shiftType,
                            intensity: result.intensity,
                            score: result.score
                        })),
                        backgroundColor: shiftResults.map((result, index) => 
                            this.chartConfig.colors[index % this.chartConfig.colors.length] + '80'  // 50% opacity
                        ),
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Trend Line',
                        data: regressionLine,
                        type: 'line',
                        borderColor: 'rgba(128, 128, 128, 0.7)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Distribution Distance'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Performance Gap'
                        },
                        ticks: {
                            callback: value => this.formatPercent(value)
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Correlation Between Distribution Distance and Performance Gap',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                if (context.datasetIndex === 0) {
                                    const dataPoint = context.raw;
                                    return [
                                        `Shift Type: ${dataPoint.shiftType}`,
                                        `Intensity: ${this.formatPercent(dataPoint.intensity)}`,
                                        `Distance: ${this.formatNumber(dataPoint.x)}`,
                                        `Performance Gap: ${this.formatPercent(dataPoint.y)}`,
                                        `Score: ${this.formatNumber(dataPoint.score)}`
                                    ];
                                } else {
                                    return `Trend Line: ${this.formatPercent(context.parsed.y)}`;
                                }
                            }
                        }
                    }
                }
            }
        };
        
        // Render chart using ChartFactory
        this.chartFactory.createChart(container, chartConfig);
    }
    
    /**
     * Initialize interactive elements
     * @param {HTMLElement} container - The container element
     */
    initializeInteractiveElements(container) {
        // Add any interaction logic for distance metrics section here
    }
    
    /**
     * Calculate linear regression (simple least squares)
     * @param {Array} x - X values
     * @param {Array} y - Y values
     * @return {Object} Slope and intercept of regression line
     */
    calculateLinearRegression(x, y) {
        if (!x || !y || x.length !== y.length || x.length < 2) {
            return { slope: 0, intercept: 0 };
        }
        
        // Calculate means
        const n = x.length;
        const meanX = x.reduce((sum, val) => sum + val, 0) / n;
        const meanY = y.reduce((sum, val) => sum + val, 0) / n;
        
        // Calculate slope
        let numerator = 0;
        let denominator = 0;
        
        for (let i = 0; i < n; i++) {
            numerator += (x[i] - meanX) * (y[i] - meanY);
            denominator += (x[i] - meanX) ** 2;
        }
        
        const slope = denominator === 0 ? 0 : numerator / denominator;
        
        // Calculate intercept
        const intercept = meanY - slope * meanX;
        
        return { slope, intercept };
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
    
    /**
     * Get HTML for a no data message
     * @param {string} message - The message to display
     * @return {string} HTML for the message
     */
    getNoDataHtml(message) {
        return `
            <div class="alert alert-info">
                <strong>No data available</strong><br>
                ${message}
            </div>
        `;
    }
}

export default DistanceMetrics;