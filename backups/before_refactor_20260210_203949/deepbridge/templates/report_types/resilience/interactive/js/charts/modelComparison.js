// Model Comparison Manager
const ModelComparisonManager = {
    // State variables
    state: {
        expandedRow: null,
        selectedMetric: 'roc_auc',
        highlightBest: true,
        modelData: {}
    },
    
    /**
     * Initialize the manager with model data
     * @param {Object} modelData - Model comparison data
     */
    initialize: function(modelData) {
        // Store model data
        this.state.modelData = modelData || {};
        
        // Render initial tables
        this.renderTable('overview-tab');
    },
    
    /**
     * Update the selected metrics display
     * @param {string} metric - Selected metric
     */
    updateMetricsDisplay: function(metric) {
        this.state.selectedMetric = metric;
        this.renderTable('metrics-tab');
    },
    
    /**
     * Set highlight best flag
     * @param {boolean} highlight - Whether to highlight best values
     */
    setHighlightBest: function(highlight) {
        this.state.highlightBest = highlight;
    },
    
    /**
     * Toggle row expansion
     * @param {string} modelKey - Key of the model to toggle
     */
    toggleRowExpansion: function(modelKey) {
        if (this.state.expandedRow === modelKey) {
            this.state.expandedRow = null;
        } else {
            this.state.expandedRow = modelKey;
        }
        
        // Re-render active table
        const activeTab = document.querySelector('.model-comparison-tab.active');
        if (activeTab) {
            const tabId = activeTab.getAttribute('data-tab');
            this.renderTable(tabId);
        }
    },
    
    /**
     * Format number with specified precision
     * @param {number} num - Number to format
     * @param {number} precision - Decimal precision
     * @returns {string} - Formatted number
     */
    formatNumber: function(num, precision = 4) {
        return Number(num).toFixed(precision);
    },
    
    /**
     * Get CSS class for robustness score
     * @param {number} score - Robustness score
     * @returns {string} - CSS class
     */
    getRobustnessColor: function(score) {
        if (score >= 0.95) return 'bg-green-500';
        if (score >= 0.85) return 'bg-blue-500';
        if (score >= 0.75) return 'bg-yellow-500';
        return 'bg-red-500';
    },
    
    /**
     * Get text color for impact value
     * @param {number} impact - Impact value
     * @returns {string} - CSS class
     */
    getImpactTextColor: function(impact) {
        if (impact < 0) return 'text-green-600'; // Improvement
        if (impact < 0.05) return 'text-blue-600'; // Small degradation
        if (impact < 0.1) return 'text-yellow-600'; // Medium degradation
        return 'text-red-600'; // Large degradation
    },
    
    /**
     * Find best value for a given metric across all models
     * @param {string} metric - Metric to compare
     * @returns {number} - Best value
     */
    getBestValue: function(metric) {
        const values = Object.values(this.state.modelData).map(model => 
            metric === 'robustness_score' ? model[metric] : 
            metric === 'raw_impact' ? model[metric] : 
            model.metrics[metric]
        );
        return Math.max(...values);
    },
    
    /**
     * Determine if a value is the best for its metric
     * @param {string} modelKey - Model key
     * @param {string} metric - Metric to compare
     * @returns {boolean} - Whether value is best
     */
    isBestValue: function(modelKey, metric) {
        if (!this.state.highlightBest) return false;
        
        const bestValue = this.getBestValue(metric);
        const modelValue = metric === 'robustness_score' ? this.state.modelData[modelKey][metric] : 
                           metric === 'raw_impact' ? this.state.modelData[modelKey][metric] : 
                           this.state.modelData[modelKey].metrics[metric];
        
        // For raw_impact, lower is better (with negative being best)
        if (metric === 'raw_impact') {
            return modelValue === Math.min(...Object.values(this.state.modelData).map(m => m[metric]));
        }
        
        return Math.abs(modelValue - bestValue) < 0.0001; // Account for floating point issues
    },
    
    /**
     * Render table based on tab ID
     * @param {string} tabId - Tab ID
     */
    renderTable: function(tabId) {
        const tableContainer = document.getElementById(tabId);
        if (!tableContainer) return;
        
        let tableHTML = '';
        
        switch (tabId) {
            case 'overview-tab':
                tableHTML = this.renderOverviewTable();
                break;
            case 'robustness-tab':
                tableHTML = this.renderRobustnessTable();
                break;
            case 'metrics-tab':
                tableHTML = this.renderMetricsTable();
                break;
            default:
                tableHTML = '<div class="text-center p-4">No data available</div>';
        }
        
        tableContainer.innerHTML = tableHTML;
    },
    
    /**
     * Render overview comparison table
     * @returns {string} - HTML for overview table
     */
    renderOverviewTable: function() {
        let html = `
        <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Model
                    </th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Base Score
                    </th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Robustness
                    </th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Impact
                    </th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Key Features
                    </th>
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">`;
            
        // Add rows for each model
        for (const [modelKey, model] of Object.entries(this.state.modelData)) {
            // Create the main model row
            html += `
                <tr 
                    data-model-key="${modelKey}"
                    class="${this.state.expandedRow === modelKey ? 'bg-blue-50' : ''} hover:bg-gray-50 cursor-pointer"
                >
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex items-center">
                            <div class="flex-shrink-0 h-10 w-10 rounded-full flex items-center justify-center ${
                                modelKey === 'primary_model' ? 'bg-blue-100' : 'bg-gray-100'
                            }">
                                <span class="text-lg font-bold">${model.name.charAt(0)}</span>
                            </div>
                            <div class="ml-4">
                                <div class="text-sm font-medium text-gray-900">${model.name}</div>
                                <div class="text-xs text-gray-500">${model.type}</div>
                            </div>
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 ${
                        this.isBestValue(modelKey, 'base_score') ? 'font-bold text-green-600' : ''
                    }">
                        ${this.formatNumber(model.base_score)}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex items-center">
                            <span class="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full text-white ${
                                this.getRobustnessColor(model.robustness_score)
                            }">
                                ${this.formatNumber(model.robustness_score * 100, 1)}%
                            </span>
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <span class="text-sm ${this.getImpactTextColor(model.raw_impact)}">
                            ${model.raw_impact < 0 ? '+' : ''}
                            ${this.formatNumber(Math.abs(model.raw_impact) * 100, 2)}%
                        </span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        ${Object.keys(model.featureImportance).slice(0, 3).join(', ')}
                    </td>
                </tr>`;
                
            // Add expanded row if model is expanded
            if (this.state.expandedRow === modelKey) {
                html += `
                <tr class="bg-blue-50">
                    <td colspan="5" class="px-6 py-4">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <h3 class="font-medium text-gray-900 mb-2">Performance Metrics</h3>
                                <div class="grid grid-cols-2 gap-2">`;
                                
                for (const [metricName, value] of Object.entries(model.metrics)) {
                    html += `
                    <div class="flex justify-between p-2 bg-white rounded shadow-sm">
                        <span class="text-sm font-medium text-gray-500 capitalize">
                            ${metricName.replace('_', ' ')}:
                        </span>
                        <span class="text-sm font-medium text-gray-900">
                            ${this.formatNumber(value)}
                        </span>
                    </div>`;
                }
                
                html += `
                                </div>
                            </div>
                            
                            <div>
                                <h3 class="font-medium text-gray-900 mb-2">Key Features Importance</h3>
                                <div class="space-y-2">`;
                                
                for (const [feature, value] of Object.entries(model.featureImportance)) {
                    html += `
                    <div class="flex items-center">
                        <span class="text-sm font-medium text-gray-500 w-24">${feature}:</span>
                        <div class="flex-grow bg-gray-200 rounded-full h-2">
                            <div 
                                class="h-2 rounded-full ${value < 0 ? 'bg-red-500' : 'bg-blue-500'}"
                                style="width: ${Math.min(Math.abs(value) * 100 * 3, 100)}%"
                            ></div>
                        </div>
                        <span class="text-sm text-gray-900 ml-2">
                            ${this.formatNumber(value, 4)}
                        </span>
                    </div>`;
                }
                
                html += `
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>`;
            }
        }
        
        html += `
            </tbody>
        </table>`;
        
        return html;
    },
    
    /**
     * Render robustness comparison table
     * @returns {string} - HTML for robustness table
     */
    renderRobustnessTable: function() {
        // Define perturbation levels - typically 0 to 1
        const perturbationLevels = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
        
        // For demo, we'll generate mock perturbation scores
        // In a real implementation, this would come from model data
        const perturbationScores = this.generatePerturbationScores(perturbationLevels);
        
        let html = `
        <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Model
                    </th>`;
                    
        // Add column for each perturbation level
        for (const level of perturbationLevels) {
            html += `
                    <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                        ${level * 100}%
                    </th>`;
        }
        
        html += `
                    <th class="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Impact
                    </th>
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">`;
            
        // Add rows for each model
        for (const [modelKey, model] of Object.entries(this.state.modelData)) {
            html += `
                <tr 
                    data-model-key="${modelKey}"
                    class="${this.state.expandedRow === modelKey ? 'bg-blue-50' : ''} hover:bg-gray-50 cursor-pointer"
                >
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex items-center">
                            <div class="flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${
                                modelKey === 'primary_model' ? 'bg-blue-100' : 'bg-gray-100'
                            }">
                                <span class="text-md font-bold">${model.name.charAt(0)}</span>
                            </div>
                            <div class="ml-3 text-sm font-medium text-gray-900">${model.name}</div>
                        </div>
                    </td>`;
                    
            // Add cell for each perturbation level
            for (let i = 0; i < perturbationLevels.length; i++) {
                const level = perturbationLevels[i];
                const score = perturbationScores[modelKey][i];
                const baseScore = model.base_score;
                const diff = score - baseScore;
                const percentChange = (diff / baseScore) * 100;
                
                // Get background color based on performance change
                let bgColor = 'bg-white';
                if (level === 0) {
                    bgColor = 'bg-gray-100';
                } else if (diff > 0) {
                    bgColor = 'bg-green-50';
                } else if (diff < -0.1) {
                    bgColor = 'bg-red-50';
                } else if (diff < -0.05) {
                    bgColor = 'bg-orange-50';
                } else if (diff < 0) {
                    bgColor = 'bg-yellow-50';
                }
                
                html += `
                    <td class="px-4 py-3 text-center text-sm ${bgColor}">
                        <div class="font-medium text-gray-900">
                            ${this.formatNumber(score)}
                        </div>
                        ${level > 0 ? `
                        <div class="text-xs ${
                            diff >= 0 ? 'text-green-600' : diff < -0.1 ? 'text-red-600' : 'text-yellow-600'
                        }">
                            ${diff >= 0 ? '+' : ''}
                            ${this.formatNumber(percentChange, 1)}%
                        </div>
                        ` : ''}
                    </td>`;
            }
            
            html += `
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-center font-medium ${
                        this.getImpactTextColor(model.raw_impact)
                    }">
                        ${model.raw_impact < 0 ? 'Improved' : this.formatNumber(model.raw_impact * 100, 1) + '%'}
                    </td>
                </tr>`;
        }
        
        html += `
            </tbody>
        </table>`;
        
        return html;
    },
    
    /**
     * Generate perturbation scores for each model
     * @param {Array} levels - Perturbation levels
     * @returns {Object} - Scores by model and level
     */
    generatePerturbationScores: function(levels) {
        const scores = {};
        
        for (const [modelKey, model] of Object.entries(this.state.modelData)) {
            scores[modelKey] = [];
            
            // Base score at level 0
            scores[modelKey].push(model.base_score);
            
            // Generate scores for other levels based on robustness score
            // Models with higher robustness score will degrade more slowly
            for (let i = 1; i < levels.length; i++) {
                const level = levels[i];
                
                // Simulate degradation based on robustness score
                // Better robustness means less degradation
                let degradation;
                if (model.robustness_score > 1) {
                    // Unusual case: model improves with perturbation
                    degradation = -0.05 * level;
                } else {
                    // Normal case: model degrades with perturbation
                    degradation = (1 - model.robustness_score) * level;
                }
                
                const score = model.base_score * (1 - degradation);
                scores[modelKey].push(score);
            }
        }
        
        return scores;
    },
    
    /**
     * Render metrics comparison table
     * @returns {string} - HTML for metrics table
     */
    renderMetricsTable: function() {
        const metric = this.state.selectedMetric;
        
        let html = `
        <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Model
                    </th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        ${metric === 'roc_auc' ? 'ROC AUC' : 
                          metric === 'f1' ? 'F1 Score' : 
                          metric.charAt(0).toUpperCase() + metric.slice(1)}
                    </th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Robustness
                    </th>
                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Trade-off
                    </th>
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">`;
        
        // Add rows for each model
        for (const [modelKey, model] of Object.entries(this.state.modelData)) {
            // Calculate the trade-off metric (performance x robustness)
            const metricValue = metric === 'robustness_score' ? model.robustness_score : 
                                metric === 'raw_impact' ? model.raw_impact : 
                                model.metrics[metric];
                                
            const tradeoffValue = metricValue * model.robustness_score;
            
            // Check if this model has the best trade-off
            const isBestTradeoff = this.isHighestTradeoff(modelKey, metric);
            
            html += `
                <tr 
                    class="${isBestTradeoff && this.state.highlightBest ? 'bg-green-50' : ''} hover:bg-gray-50"
                    data-model-key="${modelKey}"
                >
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="flex items-center">
                            <div class="flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${
                                modelKey === 'primary_model' ? 'bg-blue-100' : 'bg-gray-100'
                            }">
                                <span class="text-md font-bold">${model.name.charAt(0)}</span>
                            </div>
                            <div class="ml-3 text-sm font-medium text-gray-900">${model.name}</div>
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm ${
                        this.isBestValue(modelKey, metric) ? 'font-bold text-blue-600' : 'text-gray-500'
                    }">
                        ${this.formatNumber(metricValue)}
                        
                        <div class="mt-1 w-32 bg-gray-200 rounded-full h-2">
                            <div 
                                class="h-2 rounded-full bg-blue-500"
                                style="width: ${Math.min(metricValue * 100, 100)}%"
                            ></div>
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm ${
                        this.isBestValue(modelKey, 'robustness_score') ? 'font-bold text-blue-600' : 'text-gray-500'
                    }">
                        ${this.formatNumber(model.robustness_score, 2)}
                        
                        <div class="mt-1 w-32 bg-gray-200 rounded-full h-2">
                            <div 
                                class="h-2 rounded-full ${this.getRobustnessColor(model.robustness_score)}"
                                style="width: ${Math.min(model.robustness_score * 100, 100)}%"
                            ></div>
                        </div>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <div class="text-sm text-gray-900 font-medium">
                            ${this.formatNumber(tradeoffValue, 3)}
                        </div>
                        <div class="text-xs text-gray-500">
                            Performance × Robustness
                        </div>
                    </td>
                </tr>`;
        }
        
        html += `
            </tbody>
        </table>`;
        
        return html;
    },
    
    /**
     * Check if model has the highest trade-off value
     * @param {string} modelKey - Model key
     * @param {string} metric - Metric name
     * @returns {boolean} - Whether model has highest trade-off
     */
    isHighestTradeoff: function(modelKey, metric) {
        if (!this.state.highlightBest) return false;
        
        // Calculate trade-off for each model
        const tradeoffs = {};
        for (const [key, model] of Object.entries(this.state.modelData)) {
            const metricValue = metric === 'robustness_score' ? model.robustness_score : 
                                metric === 'raw_impact' ? model.raw_impact : 
                                model.metrics[metric];
            tradeoffs[key] = metricValue * model.robustness_score;
        }
        
        // Find the highest trade-off value
        const highestValue = Math.max(...Object.values(tradeoffs));
        
        // Check if this model has the highest trade-off
        return Math.abs(tradeoffs[modelKey] - highestValue) < 0.0001;
    },
    
    /**
     * Render perturbation chart
     * @param {string} elementId - Chart element ID
     */
    renderPerturbationChart: function(elementId) {
        const chartElement = document.getElementById(elementId);
        if (!chartElement || typeof Plotly === 'undefined') return;
        
        try {
            // Define perturbation levels
            const perturbationLevels = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
            
            // Generate perturbation scores
            const perturbationScores = this.generatePerturbationScores(perturbationLevels);
            
            // Create plot traces
            const traces = [];
            
            for (const [modelKey, model] of Object.entries(this.state.modelData)) {
                traces.push({
                    x: perturbationLevels.map(l => l * 100), // Convert to percentages
                    y: perturbationScores[modelKey],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: model.name,
                    line: {
                        width: modelKey === 'primary_model' ? 3 : 2,
                        dash: modelKey === 'primary_model' ? 'solid' : 'dot'
                    },
                    marker: {
                        size: modelKey === 'primary_model' ? 8 : 6
                    }
                });
            }
            
            // Layout configuration
            const layout = {
                title: 'Model Performance under Perturbation',
                xaxis: {
                    title: 'Perturbation Level (%)',
                    ticksuffix: '%'
                },
                yaxis: {
                    title: 'Performance Score'
                },
                legend: {
                    orientation: 'h',
                    y: -0.2
                },
                margin: {
                    l: 60,
                    r: 30,
                    t: 50,
                    b: 80
                },
                hovermode: 'closest'
            };
            
            // Create the plot
            Plotly.newPlot(chartElement, traces, layout, {responsive: true});
            
        } catch (error) {
            console.error("Error creating perturbation chart:", error);
            chartElement.innerHTML = `<div style='padding: 20px; color: red;'>Error creating chart: ${error.message}</div>`;
        }
    }
};