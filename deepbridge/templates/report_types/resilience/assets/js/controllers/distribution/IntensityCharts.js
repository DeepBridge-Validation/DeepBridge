// IntensityCharts.js - placeholder
/**
 * Component for rendering intensity charts
 * 
 * Handles the visualization of how intensity levels of distribution shifts
 * affect model performance
 */
class IntensityCharts {
    /**
     * Initialize the intensity charts renderer
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
            height: 400,
            colors: [
                '#1b78de', '#2ecc71', '#f39c12', '#e74c3c', 
                '#9b59b6', '#3498db', '#1abc9c', '#f1c40f'
            ]
        };
    }
    
    /**
     * Render intensity charts
     * @param {HTMLElement} container - The container element
     * @param {Object} distributionData - Distribution shift data
     */
    render(container, distributionData) {
        if (!container || !distributionData) {
            return;
        }
        
        // Create HTML structure for charts
        const html = this.createIntensityChartsHtml();
        container.innerHTML = html;
        
        // Render the main intensity chart
        this.renderIntensityChart(
            container.querySelector('#intensity-impact-chart'), 
            distributionData
        );
        
        // Render the shift type comparison chart
        this.renderShiftTypeChart(
            container.querySelector('#shift-type-chart'), 
            distributionData
        );
        
        // Initialize interactive elements
        this.initializeInteractiveElements(container, distributionData);
    }
    
    /**
     * Create HTML structure for intensity charts
     * @return {string} HTML for intensity charts
     */
    createIntensityChartsHtml() {
        return `
            <div class="intensity-charts-wrapper">
                <h3>Performance Across Shift Intensities</h3>
                <p>These charts show how model performance changes with different intensities of distribution shift.</p>
                
                <div class="chart-tabs">
                    <button class="chart-tab active" data-chart="intensity-impact-chart">Intensity Impact</button>
                    <button class="chart-tab" data-chart="shift-type-chart">Shift Type Comparison</button>
                </div>
                
                <div class="chart-containers">
                    <div id="intensity-impact-chart" class="chart-container active" data-chart-id="intensity-impact-chart">
                        <!-- Intensity impact chart will be rendered here -->
                    </div>
                    <div id="shift-type-chart" class="chart-container" data-chart-id="shift-type-chart">
                        <!-- Shift type chart will be rendered here -->
                    </div>
                </div>
                
                <div class="chart-description mt-3">
                    <h4>Understanding Distribution Shift Intensity</h4>
                    <p>Intensity represents the magnitude of the shift from the original distribution. 
                    Higher intensity values indicate greater deviation from the training distribution.</p>
                </div>
            </div>
        `;
    }
    
    /**
     * Render intensity impact chart
     * @param {HTMLElement} container - The chart container element
     * @param {Object} distributionData - Distribution shift data
     */
    renderIntensityChart(container, distributionData) {
        if (!container) {
            return;
        }
        
        const baseScore = distributionData.baseScore || 0;
        const shiftResults = distributionData.shiftResults || [];
        
        if (shiftResults.length === 0) {
            container.innerHTML = this.getNoDataHtml('No intensity data available for chart');
            return;
        }
        
        // Group results by shift type
        const shiftTypes = [...new Set(shiftResults.map(result => result.shiftType))];
        const datasets = [];
        
        // Create a dataset for each shift type
        shiftTypes.forEach((shiftType, index) => {
            const typeResults = shiftResults.filter(result => result.shiftType === shiftType);
            // Sort by intensity
            typeResults.sort((a, b) => a.intensity - b.intensity);
            
            datasets.push({
                label: shiftType,
                data: typeResults.map(result => ({
                    x: result.intensity,
                    y: result.score
                })),
                borderColor: this.chartConfig.colors[index % this.chartConfig.colors.length],
                backgroundColor: this.chartConfig.colors[index % this.chartConfig.colors.length] + '20',  // 12.5% opacity
                borderWidth: 2,
                fill: false,
                tension: 0.4
            });
        });
        
        // Create chart configuration
        const chartConfig = {
            type: 'line',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Shift Intensity'
                        },
                        ticks: {
                            callback: value => this.formatPercent(value)
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Performance Score'
                        },
                        max: Math.max(baseScore * 1.1, 1)
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance vs Shift Intensity',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return [
                                    `${context.dataset.label}`,
                                    `Intensity: ${this.formatPercent(context.parsed.x)}`,
                                    `Score: ${this.formatNumber(context.parsed.y)}`,
                                    `Gap: ${this.formatPercent((baseScore - context.parsed.y) / baseScore)}`
                                ];
                            }
                        }
                    }
                },
                annotation: {
                    annotations: [
                        {
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
                        }
                    ]
                }
            }
        };
        
        // Render chart using ChartFactory
        this.chartFactory.createChart(container, chartConfig);
    }
    
    /**
     * Render shift type comparison chart
     * @param {HTMLElement} container - The chart container element
     * @param {Object} distributionData - Distribution shift data
     */
    renderShiftTypeChart(container, distributionData) {
        if (!container) {
            return;
        }
        
        const baseScore = distributionData.baseScore || 0;
        const shiftResults = distributionData.shiftResults || [];
        
        if (shiftResults.length === 0) {
            container.innerHTML = this.getNoDataHtml('No shift type data available for chart');
            return;
        }
        
        // Group results by shift type and calculate averages
        const shiftTypes = [...new Set(shiftResults.map(result => result.shiftType))];
        
        // Calculate average score and performance gap for each shift type
        const averageScores = [];
        const performanceGaps = [];
        
        shiftTypes.forEach(shiftType => {
            const typeResults = shiftResults.filter(result => result.shiftType === shiftType);
            const avgScore = typeResults.reduce((sum, result) => sum + result.score, 0) / typeResults.length;
            const avgGap = (baseScore - avgScore) / baseScore;
            
            averageScores.push(avgScore);
            performanceGaps.push(avgGap);
        });
        
        // Create chart configuration
        const chartConfig = {
            type: 'bar',
            data: {
                labels: shiftTypes,
                datasets: [
                    {
                        label: 'Average Score',
                        data: averageScores,
                        backgroundColor: this.chartConfig.colors.map(color => color + '80'),  // 50% opacity
                        borderColor: this.chartConfig.colors,
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Performance Gap',
                        data: performanceGaps,
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgba(231, 76, 60, 1)',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Average Score'
                        },
                        max: Math.max(baseScore * 1.1, 1)
                    },
                    y1: {
                        beginAtZero: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Performance Gap'
                        },
                        max: 1,
                        grid: {
                            drawOnChartArea: false
                        },
                        ticks: {
                            callback: value => this.formatPercent(value)
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Shift Type'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Shift Type Comparison',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                if (context.dataset.label === 'Average Score') {
                                    return `Average Score: ${this.formatNumber(context.parsed.y)}`;
                                } else {
                                    return `Performance Gap: ${this.formatPercent(context.parsed.y)}`;
                                }
                            }
                        }
                    }
                },
                annotation: {
                    annotations: [
                        {
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
                        }
                    ]
                }
            }
        };
        
        // Render chart using ChartFactory
        this.chartFactory.createChart(container, chartConfig);
    }
    
    /**
     * Initialize interactive elements
     * @param {HTMLElement} container - The container element
     * @param {Object} distributionData - Distribution shift data
     */
    initializeInteractiveElements(container, distributionData) {
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

export default IntensityCharts;