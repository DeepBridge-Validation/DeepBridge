/**
 * Safe Chart Manager
 * Provides safe replacements for chart functions that might have syntax errors
 * Version 2.0 - May 7, 2024
 */

// Define a safe version of the ChartManager with properly structured functions
window.SafeChartManager = {
    /**
     * Initialize perturbation chart safely
     * @param {string} containerId - ID of container element
     */
    initializePerturbationChart: function(containerId) {
        console.log("Safe version of initializePerturbationChart called");
        
        try {
            // Get chart data from reportData
            const chartData = this.extractPerturbationChartData();
            if (!chartData || !chartData.levels || chartData.levels.length === 0) {
                this.showNoDataMessage(containerId, "No perturbation data available");
                return;
            }
            
            // Define colors for chart traces
            const colors = {
                primary: 'rgba(31, 119, 180, 0.7)',
                perturbed: 'rgba(44, 160, 44, 0.7)',
                worst: 'rgba(214, 39, 40, 0.7)',
                subset: 'rgba(148, 103, 189, 0.7)'
            };
            
            // Create traces for chart
            const traces = [];
            
            // Base score reference line
            if (chartData.baseScore) {
                traces.push({
                    x: chartData.levels,
                    y: Array(chartData.levels.length).fill(chartData.baseScore),
                    mode: 'lines',
                    name: 'Base Score',
                    line: {
                        dash: 'dash',
                        width: 2,
                        color: colors.primary
                    }
                });
            }
            
            // Perturbed scores
            if (chartData.perturbedScores && chartData.perturbedScores.length > 0) {
                traces.push({
                    x: chartData.levels,
                    y: chartData.perturbedScores,
                    mode: 'lines+markers',
                    name: 'Mean Score',
                    line: { color: colors.perturbed },
                    marker: {
                        size: 8,
                        color: colors.perturbed
                    }
                });
            }
            
            // Worst scores
            if (chartData.worstScores && chartData.worstScores.length > 0) {
                traces.push({
                    x: chartData.levels,
                    y: chartData.worstScores,
                    mode: 'lines+markers',
                    name: 'Worst Score',
                    line: { color: colors.worst },
                    marker: {
                        size: 8,
                        color: colors.worst
                    }
                });
            }
            
            // Feature subset scores if available
            if (chartData.featureSubsetScores && chartData.featureSubsetScores.some(s => s !== null)) {
                traces.push({
                    x: chartData.levels,
                    y: chartData.featureSubsetScores,
                    mode: 'lines+markers',
                    name: 'Feature Subset',
                    line: {
                        dash: 'dot',
                        color: colors.subset
                    },
                    marker: {
                        size: 8,
                        color: colors.subset
                    }
                });
            }
            
            // Layout for the chart
            const layout = {
                title: 'Model Performance under Perturbation',
                xaxis: {
                    title: 'Perturbation Level',
                    tickvals: chartData.levels,
                    ticktext: chartData.levels.map(String)
                },
                yaxis: {
                    title: chartData.metricName || 'Score',
                    autorange: true
                },
                legend: {
                    orientation: "h",
                    yanchor: "bottom",
                    y: -0.2,
                    xanchor: "center",
                    x: 0.5
                },
                margin: {
                    l: 60,
                    r: 20,
                    t: 40,
                    b: 80
                }
            };
            
            // Plot chart if Plotly is available
            if (typeof Plotly !== 'undefined') {
                Plotly.newPlot(containerId, traces, layout, {
                    responsive: true,
                    displayModeBar: false
                });
            } else {
                this.showNoDataMessage(containerId, "Plotly library not available");
            }
        } catch (error) {
            console.error("Error initializing perturbation chart:", error);
            this.showErrorMessage(containerId, error.message);
        }
    },
    
    /**
     * Extract perturbation chart data safely
     * @returns {Object} Chart data object
     */
    extractPerturbationChartData: function() {
        console.log("Safe version of extractPerturbationChartData called");
        
        try {
            // If perturbation_chart_data is already available, use it
            if (window.reportData && window.reportData.perturbation_chart_data) {
                return window.reportData.perturbation_chart_data;
            }
            
            // Otherwise, extract data from raw results
            if (!window.reportData || !window.reportData.raw) {
                console.warn("No raw data available for perturbation chart");
                return null;
            }
            
            // Extract basic metadata
            const result = {
                baseScore: window.reportData.base_score || 0,
                metricName: window.reportData.metric || 'Score',
                modelName: window.reportData.model_name || 'Model'
            };
            
            // Try to get perturbation levels and scores from raw data
            if (window.reportData.raw.by_level) {
                const rawData = window.reportData.raw.by_level;
                
                // Get all perturbation levels
                const levels = Object.keys(rawData)
                    .map(level => parseFloat(level))
                    .filter(level => !isNaN(level))
                    .sort((a, b) => a - b);
                
                result.levels = levels;
                
                // Extract scores for each level
                const perturbedScores = [];
                const worstScores = [];
                const featureSubsetScores = [];
                
                levels.forEach(level => {
                    const levelStr = level.toString();
                    const levelData = rawData[levelStr];
                    
                    let meanScore = null;
                    let worstScore = null;
                    let subsetScore = null;
                    
                    // Try to get scores from overall_result
                    if (levelData && levelData.overall_result) {
                        const overall = levelData.overall_result;
                        
                        if (overall.all_features) {
                            meanScore = overall.all_features.mean_score;
                            worstScore = overall.all_features.worst_score;
                        }
                        
                        if (overall.feature_subset) {
                            subsetScore = overall.feature_subset.mean_score;
                        }
                    }
                    
                    perturbedScores.push(meanScore);
                    worstScores.push(worstScore);
                    featureSubsetScores.push(subsetScore);
                });
                
                result.perturbedScores = perturbedScores;
                result.worstScores = worstScores;
                result.featureSubsetScores = featureSubsetScores;
            }
            
            return result;
        } catch (error) {
            console.error("Error extracting perturbation chart data:", error);
            return null;
        }
    },
    
    /**
     * Initialize model comparison chart safely
     * No synthetic data
     * @param {string} containerId - ID of container element
     */
    initializeModelComparisonChart: function(containerId) {
        console.log("Safe version of initializeModelComparisonChart called");
        
        try {
            // Extract data safely
            const chartData = this.extractModelComparisonData();
            
            if (!chartData || !chartData.levels || chartData.levels.length === 0 ||
                !chartData.modelScores || Object.keys(chartData.modelScores).length === 0) {
                this.showNoDataMessage(containerId, "No model comparison data available");
                return;
            }
            
            // Define some colors for the chart
            const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];
            let colorIndex = 0;
            
            // Get all model IDs
            const allModelIds = Object.keys(chartData.modelScores);
            
            // Create plot data array
            const plotData = [];
            
            // Add primary model first if available
            if (chartData.modelScores['primary']) {
                console.log("Adding primary model to chart");
                plotData.push({
                    x: chartData.levels,
                    y: chartData.modelScores['primary'],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: chartData.modelNames['primary'] || 'Primary Model',
                    line: {
                        width: 3,
                        color: colors[colorIndex % colors.length]
                    },
                    marker: {
                        size: 8,
                        color: colors[colorIndex % colors.length]
                    }
                });
                colorIndex++;
            }
            
            // Add all other models, only real data
            for (const modelId of allModelIds) {
                // Skip primary model as we've already added it
                if (modelId === 'primary') {
                    continue;
                }
                
                console.log(`Adding model ${modelId} to chart`);
                
                // Check if scores are valid - only use real scores
                const validScores = chartData.modelScores[modelId] && 
                                   chartData.modelScores[modelId].some(score => score !== null);
                
                if (!validScores) {
                    console.log(`Model ${modelId} has no valid scores, skipping`);
                    continue;
                }
                
                // Add to chart if we have valid scores
                plotData.push({
                    x: chartData.levels,
                    y: chartData.modelScores[modelId],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: chartData.modelNames[modelId] || modelId,
                    line: {
                        width: 2.5,
                        color: colors[colorIndex % colors.length]
                    },
                    marker: {
                        size: 7,
                        color: colors[colorIndex % colors.length]
                    }
                });
                colorIndex++;
            }
            
            // If no models have been added, show a message
            if (plotData.length === 0) {
                this.showNoDataMessage(containerId, "No models with valid data available");
                return;
            }
            
            // Layout
            const layout = {
                title: 'Model Comparison: Performance by Perturbation Level',
                xaxis: {
                    title: 'Perturbation Level',
                    tickvals: chartData.levels,
                    ticktext: chartData.levels.map(String)
                },
                yaxis: {
                    title: `${chartData.metricName} Score`,
                    autorange: true
                },
                legend: {
                    orientation: "h",
                    yanchor: "top",
                    y: 1,
                    xanchor: "right",
                    x: 1
                },
                hovermode: 'closest',
                margin: {
                    l: 60,
                    r: 20,
                    t: 40,
                    b: 40
                }
            };
            
            // Create the plot if Plotly is available
            if (typeof Plotly !== 'undefined') {
                Plotly.newPlot(containerId, plotData, layout, {
                    responsive: true,
                    displayModeBar: false
                });
            } else {
                this.showNoDataMessage(containerId, "Plotly library not available");
            }
        } catch (error) {
            console.error("Error initializing model comparison chart:", error);
            this.showErrorMessage(containerId, error.message);
        }
    },
    
    /**
     * Extract model comparison data safely without synthetic data
     * @returns {Object} Chart data object
     */
    extractModelComparisonData: function() {
        console.log("Safe version of extractModelComparisonData called");
        
        try {
            // Check if we have valid report data
            if (!window.reportData) {
                console.warn("No reportData available");
                return null;
            }
            
            // Check for alternative models
            const hasAlternativeModels = window.reportData.alternative_models && 
                                        Object.keys(window.reportData.alternative_models).length > 0;
            
            if (!hasAlternativeModels) {
                console.warn("No alternative models found - no comparison needed");
                return null;
            }
            
            // Get primary model name
            const primaryModelName = window.reportData.model_name || "Primary Model";
            const metricName = window.reportData.metric || "Score";
            
            // Determine available levels
            const rawData = window.reportData.raw && window.reportData.raw.by_level || {};
            const perturbationData = window.reportData.perturbation_chart_data || {};
            
            // Use perturbation chart data levels if available, otherwise extract from raw data
            let levels = [];
            if (perturbationData && perturbationData.levels) {
                levels = perturbationData.levels;
                console.log("Using levels from perturbation chart data:", levels);
            } else {
                // Extract levels from raw data
                levels = Object.keys(rawData)
                    .map(level => parseFloat(level))
                    .filter(level => !isNaN(level))
                    .sort((a, b) => a - b);
                console.log("Extracted levels from raw data:", levels);
            }
            
            if (levels.length === 0) {
                console.warn("No valid perturbation levels found");
                return null;
            }
            
            // Initialize model scores and names
            const modelScores = {};
            const modelNames = {};
            
            // Add primary model scores if available
            if (perturbationData.scores && perturbationData.scores.length > 0) {
                modelScores["primary"] = perturbationData.scores;
                modelNames["primary"] = primaryModelName;
                console.log("Added primary model scores from perturbation_chart_data");
            } 
            // Otherwise extract from raw data
            else if (Object.keys(rawData).length > 0) {
                const primaryScores = levels.map(level => {
                    const levelStr = level.toString();
                    if (rawData[levelStr] && 
                        rawData[levelStr].overall_result && 
                        rawData[levelStr].overall_result.all_features) {
                        const score = rawData[levelStr].overall_result.all_features.mean_score;
                        return typeof score === 'number' ? score : null;
                    }
                    return null;
                });
                
                if (primaryScores.some(score => score !== null)) {
                    modelScores["primary"] = primaryScores;
                    modelNames["primary"] = primaryModelName;
                    console.log("Added primary model scores from raw data");
                }
            }
            
            // Process alternative models - ONLY use real data
            const alternativeModels = window.reportData.alternative_models || {};
            
            // Count how many alternative models have real perturbation test data
            let modelsWithRealData = 0;
            
            Object.entries(alternativeModels).forEach(([modelId, modelData]) => {
                // Get model name
                const name = modelData.model_name || modelId;
                
                // Only include models that have actual test data for perturbation levels
                if (modelData.raw && modelData.raw.by_level) {
                    // Extract actual scores for each perturbation level
                    const scores = levels.map(level => {
                        const levelStr = level.toString();
                        if (modelData.raw.by_level[levelStr] && 
                            modelData.raw.by_level[levelStr].overall_result && 
                            modelData.raw.by_level[levelStr].overall_result.all_features) {
                            return modelData.raw.by_level[levelStr].overall_result.all_features.mean_score || null;
                        }
                        return null;
                    });
                    
                    // Only include models with at least one valid score
                    if (scores.some(score => score !== null)) {
                        modelScores[modelId] = scores;
                        modelNames[modelId] = name;
                        modelsWithRealData++;
                        console.log(`Added real data for alternative model: ${name}`);
                    }
                }
            });
            
            // Return the processed data with NO TRAILING COMMAS
            return {
                levels: levels,
                modelScores: modelScores,
                modelNames: modelNames,
                metricName: metricName
            };
        } catch (error) {
            console.error("Error extracting model comparison data:", error);
            return null;
        }
    },
    
    /**
     * Display a message when no data is available
     * @param {string} containerId - ID of container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(containerId, message) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #f8f9fa; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">No Data Available</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">
                    ${message}
                </p>
                <p style="color: #666; margin-top: 20px; font-size: 14px;">
                    Run perturbation tests to generate real data for visualization.
                </p>
            </div>`;
    },
    
    /**
     * Display an error message
     * @param {string} containerId - ID of container element
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(containerId, errorMessage) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Chart Error</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">${errorMessage}</p>
            </div>`;
    }
};

// Install the safe chart manager when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log("Installing SafeChartManager...");
    
    // Replace existing ChartManager methods with safe versions
    if (window.ChartManager) {
        console.log("Found ChartManager, replacing methods with safe versions");
        
        // Replace perturbation chart methods
        if (typeof window.ChartManager.initializePerturbationChart === 'function') {
            window.ChartManager.initializePerturbationChart = SafeChartManager.initializePerturbationChart.bind(SafeChartManager);
            console.log("Replaced initializePerturbationChart with safe version");
        }
        
        if (typeof window.ChartManager.extractPerturbationChartData === 'function') {
            window.ChartManager.extractPerturbationChartData = SafeChartManager.extractPerturbationChartData.bind(SafeChartManager);
            console.log("Replaced extractPerturbationChartData with safe version");
        }
        
        // Replace model comparison chart methods
        if (typeof window.ChartManager.initializeModelComparisonChart === 'function') {
            window.ChartManager.initializeModelComparisonChart = SafeChartManager.initializeModelComparisonChart.bind(SafeChartManager);
            console.log("Replaced initializeModelComparisonChart with safe version");
        }
        
        if (typeof window.ChartManager.extractModelComparisonData === 'function') {
            window.ChartManager.extractModelComparisonData = SafeChartManager.extractModelComparisonData.bind(SafeChartManager);
            console.log("Replaced extractModelComparisonData with safe version");
        }
        
        // Add utility methods if they don't exist
        if (typeof window.ChartManager.showNoDataMessage !== 'function') {
            window.ChartManager.showNoDataMessage = SafeChartManager.showNoDataMessage.bind(SafeChartManager);
            console.log("Added showNoDataMessage to ChartManager");
        }
        
        if (typeof window.ChartManager.showErrorMessage !== 'function') {
            window.ChartManager.showErrorMessage = SafeChartManager.showErrorMessage.bind(SafeChartManager);
            console.log("Added showErrorMessage to ChartManager");
        }
    } else {
        // If ChartManager doesn't exist, create it from SafeChartManager
        console.log("ChartManager not found, creating one from SafeChartManager");
        window.ChartManager = SafeChartManager;
    }
});