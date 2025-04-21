// DistributionController.js
/**
 * Main controller for the distribution shift analysis section
 * 
 * Coordinates all interactions and data flow for the distribution shift analysis,
 * delegating specific rendering tasks to specialized components
 */
class DistributionController {
    /**
     * Initialize the distribution controller
     */
    constructor() {
        // Track state
        this.activeFeature = null;
        this.activeMetric = 'PSI';
        this.activeChartType = 'shift-magnitude';
        
        // Bind event handlers
        this.switchIntensityChart = this.switchIntensityChart.bind(this);
        this.filterFeatures = this.filterFeatures.bind(this);
        this.updateFeatureDisplay = this.updateFeatureDisplay.bind(this);
        
        // Initialize when data is ready
        document.addEventListener('DOMContentLoaded', () => {
            // Initialize on tab activation
            document.addEventListener('tab-changed', (event) => {
                if (event.detail && event.detail.tabId === 'distribution') {
                    this.initialize();
                }
            });
            
            // Check if this tab is already active
            if (document.getElementById('distribution').classList.contains('active')) {
                this.initialize();
            }
        });
    }
    
    /**
     * Initialize the controller and load data
     */
    initialize() {
        // Get the report data
        const reportData = window.reportData || {};
        
        // Set up data model if not already done
        if (!window.dataModel) {
            window.dataModel = new DataModel(reportData);
            document.dispatchEvent(new CustomEvent('dataModelInitialized'));
        }
        
        // Set up feature selector
        this.initializeFeatureSelector();
        
        // Set up chart selectors
        this.initializeChartSelectors();
        
        // Initial render of intensity chart
        this.renderIntensityChart();
        
        // Load distance metrics table
        this.loadDistanceMetricsTable();
    }
    
    /**
     * Initialize the feature selector dropdown
     */
    initializeFeatureSelector() {
        const featureSelector = document.getElementById('distribution-feature-selector');
        if (!featureSelector) return;
        
        // Clear existing options
        while (featureSelector.options.length > 1) {
            featureSelector.remove(1);
        }
        
        // Get features from data model
        const features = window.dataModel.getFeatures();
        if (!features || features.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.text = 'No features available';
            featureSelector.add(option);
            return;
        }
        
        // Add features to selector
        features.forEach(feature => {
            const option = document.createElement('option');
            option.value = feature;
            option.text = feature;
            featureSelector.add(option);
        });
        
        // Add event listener
        featureSelector.addEventListener('change', this.updateFeatureDisplay);
    }
    
    /**
     * Initialize chart selector buttons
     */
    initializeChartSelectors() {
        // Intensity chart selector
        const intensitySelector = document.getElementById('intensity_charts_selector');
        if (intensitySelector) {
            const chartButtons = intensitySelector.querySelectorAll('.chart-option');
            chartButtons.forEach(button => {
                button.addEventListener('click', () => {
                    this.switchIntensityChart(button.dataset.option);
                });
            });
        }
    }
    
    /**
     * Switch between different intensity chart types
     * @param {string} chartType - The chart type to switch to
     */
    switchIntensityChart(chartType) {
        // Update active chart type
        this.activeChartType = chartType;
        
        // Update container visibility
        const containers = document.querySelectorAll('.intensity-charts-container .chart-container');
        containers.forEach(container => {
            container.classList.toggle('active', container.dataset.chartType === chartType);
        });
        
        // Render the selected chart
        this.renderIntensityChart();
    }
    
    /**
     * Apply filters to the feature list
     * @param {Object} filters - The filters to apply
     */
    filterFeatures(filters) {
        // This would filter the features based on user criteria
        console.log('Filtering features with:', filters);
        
        // Re-render intensity chart with filtered features
        this.renderIntensityChart();
    }
    
    /**
     * Update the feature display when a feature is selected
     */
    updateFeatureDisplay() {
        const featureSelector = document.getElementById('distribution-feature-selector');
        if (!featureSelector) return;
        
        const selectedFeature = featureSelector.value;
        if (!selectedFeature) return;
        
        // Update active feature
        this.activeFeature = selectedFeature;
        
        // Render distribution chart
        this.renderDistributionChart(selectedFeature);
        
        // Update statistical metrics
        this.updateDistributionMetrics(selectedFeature);
    }
    
    /**
     * Render the intensity chart based on current state
     */
    renderIntensityChart() {
        const container = document.getElementById(`${this.activeChartType}-container`);
        if (!container) return;
        
        // Get data from the model
        const dataModel = window.dataModel;
        if (!dataModel) {
            container.innerHTML = '<div class="no-data-message">Data model not initialized</div>';
            return;
        }
        
        // Different rendering based on chart type
        switch (this.activeChartType) {
            case 'shift-magnitude':
                this.renderShiftMagnitudeChart(container);
                break;
            case 'feature-ranking':
                this.renderFeatureRankingChart(container);
                break;
            case 'shift-heatmap':
                this.renderShiftHeatmapChart(container);
                break;
        }
    }
    
    /**
     * Render the shift magnitude chart
     * @param {HTMLElement} container - The container element
     */
    renderShiftMagnitudeChart(container) {
        // Get data from model
        const dataModel = window.dataModel;
        const features = dataModel.getFeatureShiftMagnitudes();
        
        if (!features || Object.keys(features).length === 0) {
            container.innerHTML = '<div class="no-data-message">No feature shift data available</div>';
            return;
        }
        
        // Sort features by magnitude
        const sortedFeatures = Object.entries(features)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10); // Top 10 features
            
        // Create chart with Plotly.js
        const featureNames = sortedFeatures.map(f => f[0]);
        const shiftValues = sortedFeatures.map(f => f[1]);
        
        Plotly.newPlot(container, [
            {
                x: shiftValues,
                y: featureNames,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: shiftValues.map(v => {
                        return v > 0.5 ? 'rgba(255, 99, 71, 0.7)' :
                               v > 0.2 ? 'rgba(255, 165, 0, 0.7)' : 
                               'rgba(46, 139, 87, 0.7)';
                    })
                }
            }
        ], {
            title: 'Top 10 Features by Shift Magnitude',
            margin: { l: 150, r: 30, t: 50, b: 50 },
            xaxis: { title: 'Shift Magnitude' },
            yaxis: { title: 'Feature' }
        }, { responsive: true });
    }
    
    /**
     * Render the feature ranking chart
     * @param {HTMLElement} container - The container element
     */
    renderFeatureRankingChart(container) {
        container.innerHTML = '<div class="no-data-message">Feature ranking visualization will be available in a future update</div>';
        // Will implement in the future
    }
    
    /**
     * Render the shift heatmap chart
     * @param {HTMLElement} container - The container element
     */
    renderShiftHeatmapChart(container) {
        container.innerHTML = '<div class="no-data-message">Shift heatmap visualization will be available in a future update</div>';
        // Will implement in the future
    }
    
    /**
     * Render the distribution chart for a specific feature
     * @param {string} feature - The feature to display
     */
    renderDistributionChart(feature) {
        const container = document.getElementById('distribution-chart-container');
        if (!container) return;
        
        const dataModel = window.dataModel;
        if (!dataModel) {
            container.innerHTML = '<div class="no-data-message">Data model not initialized</div>';
            return;
        }
        
        // Placeholder for actual distribution data
        // In a real implementation, this would get the baseline and target distributions for the feature
        const baselineData = dataModel.getBaselineDistribution(feature);
        const targetData = dataModel.getTargetDistribution(feature);
        
        if (!baselineData || !targetData) {
            container.innerHTML = '<div class="no-data-message">Distribution data not available for this feature</div>';
            return;
        }
        
        // Create distribution chart
        Plotly.newPlot(container, [
            {
                x: baselineData,
                type: 'histogram',
                name: 'Baseline',
                opacity: 0.7,
                marker: { color: 'blue' }
            },
            {
                x: targetData,
                type: 'histogram',
                name: 'Target',
                opacity: 0.7,
                marker: { color: 'red' }
            }
        ], {
            title: `Distribution Comparison: ${feature}`,
            xaxis: { title: 'Value' },
            yaxis: { title: 'Frequency' },
            barmode: 'overlay'
        }, { responsive: true });
    }
    
    /**
     * Update the distribution metrics for a specific feature
     * @param {string} feature - The feature to display metrics for
     */
    updateDistributionMetrics(feature) {
        // Get metrics from data model
        const dataModel = window.dataModel;
        if (!dataModel) return;
        
        const metrics = dataModel.getDistributionMetrics(feature);
        if (!metrics) {
            document.getElementById('kl-divergence').textContent = 'data not available';
            document.getElementById('js-distance').textContent = 'data not available';
            document.getElementById('wasserstein').textContent = 'data not available';
            document.getElementById('hellinger').textContent = 'data not available';
            
            document.getElementById('baseline-mean').textContent = 'data not available';
            document.getElementById('target-mean').textContent = 'data not available';
            document.getElementById('mean-change').textContent = 'data not available';
            
            document.getElementById('baseline-median').textContent = 'data not available';
            document.getElementById('target-median').textContent = 'data not available';
            document.getElementById('median-change').textContent = 'data not available';
            
            document.getElementById('baseline-std').textContent = 'data not available';
            document.getElementById('target-std').textContent = 'data not available';
            document.getElementById('std-change').textContent = 'data not available';
            
            document.getElementById('baseline-iqr').textContent = 'data not available';
            document.getElementById('target-iqr').textContent = 'data not available';
            document.getElementById('iqr-change').textContent = 'data not available';
            return;
        }
        
        // Update distance metrics
        document.getElementById('kl-divergence').textContent = metrics.klDivergence ? metrics.klDivergence.toFixed(4) : 'data not available';
        document.getElementById('js-distance').textContent = metrics.jsDistance ? metrics.jsDistance.toFixed(4) : 'data not available';
        document.getElementById('wasserstein').textContent = metrics.wasserstein ? metrics.wasserstein.toFixed(4) : 'data not available';
        document.getElementById('hellinger').textContent = metrics.hellinger ? metrics.hellinger.toFixed(4) : 'data not available';
        
        // Update distribution statistics
        document.getElementById('baseline-mean').textContent = metrics.baselineMean ? metrics.baselineMean.toFixed(4) : 'data not available';
        document.getElementById('target-mean').textContent = metrics.targetMean ? metrics.targetMean.toFixed(4) : 'data not available';
        document.getElementById('mean-change').textContent = metrics.meanChange ? metrics.meanChange.toFixed(4) : 'data not available';
        
        document.getElementById('baseline-median').textContent = metrics.baselineMedian ? metrics.baselineMedian.toFixed(4) : 'data not available';
        document.getElementById('target-median').textContent = metrics.targetMedian ? metrics.targetMedian.toFixed(4) : 'data not available';
        document.getElementById('median-change').textContent = metrics.medianChange ? metrics.medianChange.toFixed(4) : 'data not available';
        
        document.getElementById('baseline-std').textContent = metrics.baselineStd ? metrics.baselineStd.toFixed(4) : 'data not available';
        document.getElementById('target-std').textContent = metrics.targetStd ? metrics.targetStd.toFixed(4) : 'data not available';
        document.getElementById('std-change').textContent = metrics.stdChange ? metrics.stdChange.toFixed(4) : 'data not available';
        
        document.getElementById('baseline-iqr').textContent = metrics.baselineIQR ? metrics.baselineIQR.toFixed(4) : 'data not available';
        document.getElementById('target-iqr').textContent = metrics.targetIQR ? metrics.targetIQR.toFixed(4) : 'data not available';
        document.getElementById('iqr-change').textContent = metrics.iqrChange ? metrics.iqrChange.toFixed(4) : 'data not available';
    }
    
    /**
     * Load the distance metrics table
     */
    loadDistanceMetricsTable() {
        const tableContainer = document.querySelector('.distance-metrics-container');
        if (!tableContainer) return;
        
        const dataModel = window.dataModel;
        if (!dataModel) {
            tableContainer.innerHTML = '<div class="no-data-message">Data model not initialized</div>';
            return;
        }
        
        const metrics = dataModel.getAllFeatureDistanceMetrics();
        if (!metrics || metrics.length === 0) {
            tableContainer.innerHTML = '<div class="no-data-message">No distance metrics data available</div>';
            return;
        }
        
        // Create table
        const tableHtml = `
            <table class="data-table distance-metrics-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>PSI</th>
                        <th>KS</th>
                        <th>WD1</th>
                        <th>Impact</th>
                    </tr>
                </thead>
                <tbody>
                    ${metrics.map(m => `
                        <tr>
                            <td>${m.feature}</td>
                            <td>${m.PSI ? m.PSI.toFixed(4) : 'data not available'}</td>
                            <td>${m.KS ? m.KS.toFixed(4) : 'data not available'}</td>
                            <td>${m.WD1 ? m.WD1.toFixed(4) : 'data not available'}</td>
                            <td>${m.impact ? m.impact.toFixed(4) : 'data not available'}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        tableContainer.innerHTML = tableHtml;
    }
}

// Initialize the controller when the document is ready
const distributionController = new DistributionController();