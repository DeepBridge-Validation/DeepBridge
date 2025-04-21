// ShiftResults.js - placeholder
/**
 * Component for rendering distribution shift results
 * 
 * Handles the rendering of detailed shift results including
 * tables and visualizations of performance across different shifts
 */
class ShiftResults {
    /**
     * Initialize the shift results renderer
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
            height: 300,
            colors: [
                '#1b78de', '#2ecc71', '#f39c12', '#e74c3c', 
                '#9b59b6', '#3498db', '#1abc9c', '#f1c40f'
            ]
        };
    }
    
    /**
     * Render shift results
     * @param {HTMLElement} container - The container element
     * @param {Object} shiftData - Distribution shift data
     */
    render(container, shiftData) {
        if (!container || !shiftData) {
            return;
        }
        
        // Create HTML for shift results
        const html = this.createShiftResultsHtml(shiftData);
        container.innerHTML = html;
        
        // Initialize and render charts
        this.initializeCharts(container, shiftData);
        
        // Initialize interactive elements
        this.initializeInteractiveElements(container);
    }
    
    /**
     * Create HTML for shift results
     * @param {Object} shiftData - Distribution shift data
     * @return {string} HTML for shift results
     */
    createShiftResultsHtml(shiftData) {
        const baseScore = shiftData.baseScore || 0;
        const shiftResults = shiftData.shiftResults || [];
        const distanceMetrics = shiftData.distanceMetrics || [];
        
        // Format results table
        let resultsTableHtml = `
            <h4>Distribution Shift Results</h4>
            <div class="table-container">
                <table class="data-table shift-results-table">
                    <thead>
                        <tr>
                            <th>Shift Type</th>
                            <th>Intensity</th>
                            <th>Distance</th>
                            <th>Score</th>
                            <th>Performance Gap</th>
                            <th>Relative to Base</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        if (shiftResults.length === 0) {
            resultsTableHtml += `
                <tr>
                    <td colspan="6">No shift results available</td>
                </tr>
            `;
        } else {
            // Add rows for each shift result
            shiftResults.forEach(result => {
                const performanceGap = (baseScore - result.score) / baseScore;
                const relativePerformance = result.score / baseScore;
                
                let performanceClass = '';
                if (relativePerformance >= 0.9) performanceClass = 'excellent';
                else if (relativePerformance >= 0.8) performanceClass = 'good';
                else if (relativePerformance >= 0.7) performanceClass = 'moderate';
                else if (relativePerformance >= 0.6) performanceClass = 'fair';
                else performanceClass = 'poor';
                
                resultsTableHtml += `
                    <tr>
                        <td>${result.shiftType || 'Unknown'}</td>
                        <td>${this.formatIntensity(result.intensity)}</td>
                        <td>${this.formatNumber(result.distance)}</td>
                        <td>${this.formatNumber(result.score)}</td>
                        <td>${this.formatPercent(performanceGap)}</td>
                        <td class="${performanceClass}">${this.formatPercent(relativePerformance)}</td>
                    </tr>
                `;
            });
        }
        
        resultsTableHtml += `
                    </tbody>
                </table>
            </div>
        `;
        
        // Add distance metrics section if available
        let distanceMetricsHtml = '';
        if (distanceMetrics.length > 0) {
            distanceMetricsHtml = `
                <div class="distance-metrics-section mt-4">
                    <h4>Distance Metrics Analysis</h4>
                    <p>The following distance metrics were used to quantify the difference between 
                    the original distribution and the shifted distributions:</p>
                    <ul class="metrics-list">
                        ${distanceMetrics.map(metric => `
                            <li><strong>${metric.name}:</strong> ${metric.description}</li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Add chart containers
        const chartsHtml = `
            <div class="shift-charts-section mt-4">
                <h4>Visual Analysis</h4>
                <div class="chart-tabs">
                    <button class="chart-tab active" data-chart="performance-chart">Performance</button>
                    <button class="chart-tab" data-chart="distance-chart">Distance Impact</button>
                </div>
                <div class="chart-containers">
                    <div id="shift-performance-chart" class="chart-container active" data-chart-id="performance-chart">
                        <!-- Performance chart will be rendered here -->
                    </div>
                    <div id="shift-distance-chart" class="chart-container" data-chart-id="distance-chart">
                        <!-- Distance chart will be rendered here -->
                    </div>
                </div>
            </div>
        `;
        
        // Combine all sections
        return `
            <div class="shift-results-wrapper">
                ${resultsTableHtml}
                ${distanceMetricsHtml}
                ${chartsHtml}
            </div>
        `;
    }
    
    /**
     * Initialize and render charts
     * @param {HTMLElement} container - The container element
     * @param {Object} shiftData - Distribution shift data
     */
    initializeCharts(container, shiftData) {
        // Get chart containers
        const performanceChartContainer = container.querySelector('#shift-performance-chart');
        const distanceChartContainer = container.querySelector('#shift-distance-chart');
        
        if (!performanceChartContainer || !distanceChartContainer) {
            return;
        }
        
        // Render performance chart
        this.renderPerformanceChart(performanceChartContainer, shiftData);
        
        // Render distance chart
        this.renderDistanceChart(distanceChartContainer, shiftData);
    }
    
    /**
     * Render performance chart
     * @param {HTMLElement} container - The chart container element
     * @param {Object} shiftData - Distribution shift data
     */
    renderPerformanceChart(container, shiftData) {
        const baseScore = shiftData.baseScore || 0;
        const shiftResults = shiftData.shiftResults || [];
        
        if (shiftResults.length === 0) {
            container.innerHTML = this.getNoDataHtml('No performance data available for chart');
            return;
        }
        
        // Prepare data for chart
        const shiftTypes = [...new Set(shiftResults.map(result => result.shiftType))];
        const chartData = {
            labels: shiftResults.map(result => 
                `${result.shiftType} (${this.formatIntensity(result.intensity)})`
            ),
            datasets: [{
                label: 'Performance Score',
                data: shiftResults.map(result => result.score),
                backgroundColor: shiftResults.map((result, index) => 
                    this.chartConfig.colors[index % this.chartConfig.colors.length] + '80'  // 50% opacity
                ),
                borderColor: shiftResults.map((result, index) => 
                    this.chartConfig.colors[index % this.chartConfig.colors.length]
                ),
                borderWidth: 1
            }]
        };
        
        // Create chart configuration
        const chartConfig = {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Across Distribution Shifts',
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
                                const result = shiftResults[context.dataIndex];
                                return [
                                    `Score: ${this.formatNumber(result.score)}`,
                                    `Gap: ${this.formatPercent((baseScore - result.score) / baseScore)}`,
                                    `Distance: ${this.formatNumber(result.distance)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: Math.max(baseScore * 1.1, 1),
                        title: {
                            display: true,
                            text: 'Performance Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Distribution Shift'
                        }
                    }
                },
                annotation: {
                    annotations: [{
                        type: 'line',
                        mode: 'horizontal',
                        scaleID: 'y',
                        value: baseScore,
                        borderColor: 'rgba(46, 204, 113, 0.8)',
                        borderWidth: 2,
                        borderDash: [6, 4],
                        label: {
                            content: `Base Score: ${this.formatNumber(baseScore)}`,
                            enabled: true,
                            position: 'right'
                        }
                    }]
                }
            }
        };
        
        // Render chart using ChartFactory
        this.chartFactory.createChart(container, chartConfig);
    }
    
    /**
     * Render distance chart
     * @param {HTMLElement} container - The chart container element
     * @param {Object} shiftData - Distribution shift data
     */
    renderDistanceChart(container, shiftData) {
        const baseScore = shiftData.baseScore || 0;
        const shiftResults = shiftData.shiftResults || [];
        
        if (shiftResults.length === 0) {
            container.innerHTML = this.getNoDataHtml('No distance data available for chart');
            return;
        }
        
        // Prepare data for chart
        const chartData = {
            datasets: [{
                label: 'Performance Gap vs Distance',
                data: shiftResults.map(result => ({
                    x: result.distance,
                    y: (baseScore - result.score) / baseScore,
                    r: result.intensity * 10 + 5,  // Size based on intensity
                })),
                backgroundColor: shiftResults.map((result, index) => 
                    this.chartConfig.colors[index % this.chartConfig.colors.length] + '80'  // 50% opacity
                ),
                borderColor: shiftResults.map((result, index) => 
                    this.chartConfig.colors[index % this.chartConfig.colors.length]
                ),
                borderWidth: 1
            }]
        };
        
        // Create chart configuration
        const chartConfig = {
            type: 'bubble',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Gap vs Distribution Distance',
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
                                const result = shiftResults[context.dataIndex];
                                return [
                                    `Shift: ${result.shiftType} (${this.formatIntensity(result.intensity)})`,
                                    `Distance: ${this.formatNumber(result.distance)}`,
                                    `Gap: ${this.formatPercent((baseScore - result.score) / baseScore)}`,
                                    `Score: ${this.formatNumber(result.score)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Performance Gap'
                        },
                        ticks: {
                            callback: value => this.formatPercent(value)
                        }
                    },
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Distribution Distance'
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
        // Set up chart tab navigation
        const chartTabs = container.querySelectorAll('.chart-tab');
        const chartContainers = container.querySelectorAll('.chart-container');
        
        chartTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Deactivate all tabs and containers
                chartTabs.forEach(t => t.classList.remove('active'));
                chartContainers.forEach(c => c.classList.remove('active'));
                
                // Activate selected tab and container
                tab.classList.add('active');
                const chartId = tab.getAttribute('data-chart');
                const chartContainer = container.querySelector(`.chart-container[data-chart-id="${chartId}"]`);
                if (chartContainer) {
                    chartContainer.classList.add('active');
                }
                
                // Trigger resize event to ensure proper chart rendering
                window.dispatchEvent(new Event('resize'));
            });
        });
    }
    
    /**
     * Format intensity value
     * @param {number} intensity - Intensity value (0-1)
     * @return {string} Formatted intensity
     */
    formatIntensity(intensity) {
        if (intensity === undefined || intensity === null || isNaN(intensity)) {
            return 'N/A';
        }
        
        // Format as percentage
        return this.formatPercent(intensity);
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

export default ShiftResults;