/**
 * Overview Charts Manager
 * Handles chart creation and data visualization for the model overview page
 */
window.OverviewChartsManager = {
    /**
     * Initialize all charts on the overview page
     */
    initializeOverviewCharts: function() {
        console.log("Initializing overview charts");
        
        try {
            // Initialize the metrics radar chart
            this.initializeMetricsRadarChart("metrics-radar-chart");
        } catch (error) {
            console.error("Error initializing overview charts:", error);
            this.showErrorMessage(document.getElementById("metrics-radar-chart"), error.message);
        }
    },
    
    /**
     * Initialize metrics radar chart
     * @param {string} elementId - Chart container ID
     */
    initializeMetricsRadarChart: function(elementId) {
        console.log("Initializing metrics radar chart");
        const chartElement = document.getElementById(elementId);
        if (!chartElement) {
            console.error("Chart element not found:", elementId);
            return;
        }
        
        try {
            // Clear any previous content to avoid double rendering
            chartElement.innerHTML = '';
            
            // Extract data for chart with validation
            const chartData = this.extractMetricsRadarData();
            
            if (!chartData || !chartData.models || chartData.models.length === 0) {
                this.showNoDataMessage(chartElement, "No model metrics data available");
                return;
            }
            
            // Create data traces for each model
            const traces = [];
            
            // Get metrics and model names
            const metrics = chartData.metrics;
            const modelNames = chartData.models.map(model => model.name);
            
            // Create a radar trace for each metric
            metrics.forEach(metric => {
                const trace = {
                    type: 'scatterpolar',
                    name: metric.displayName,
                    r: chartData.models.map(model => model.metrics[metric.name]),
                    theta: modelNames,
                    fill: 'toself',
                    opacity: 0.7,
                    hovertemplate: '<b>%{theta}</b><br>' + 
                                   metric.displayName + ': %{r:.4f}<extra></extra>'
                };
                traces.push(trace);
            });
            
            // Layout for the radar chart
            const layout = {
                title: 'Comparação de Métricas por Modelo',
                polar: {
                    radialaxis: {
                        visible: true,
                        range: [0, 1],
                        showticklabels: true,
                        gridcolor: "#d9d9d9",
                        tickformat: ".2f"
                    },
                    angularaxis: {
                        gridcolor: "#d9d9d9"
                    }
                },
                legend: {
                    orientation: 'h',
                    y: -0.2
                },
                margin: {
                    l: 50,
                    r: 50,
                    t: 60,
                    b: 70
                },
                plot_bgcolor: '#fafafa',
                paper_bgcolor: '#fff'
            };
            
            // Create the plot with error handling
            try {
                Plotly.newPlot(chartElement, traces, layout, {
                    responsive: true,
                    displayModeBar: false,
                    displaylogo: false
                });
            } catch (plotlyError) {
                console.error("Plotly rendering error:", plotlyError);
                this.showErrorMessage(chartElement, "Chart rendering failed: " + plotlyError.message);
                return;
            }
            
            // Add resize event listener
            window.addEventListener('resize', () => {
                Plotly.relayout(chartElement, {
                    'autosize': true
                });
            });
            
        } catch (error) {
            console.error("Error creating metrics radar chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Extract data for metrics radar chart
     * @returns {Object} - Metrics and model data
     */
    extractMetricsRadarData: function() {
        let models = [];
        
        // Try multiple sources for model metrics data
        let modelData = {};
        
        console.log("Data sources available:", {
            "reportData": !!window.reportData,
            "chartData": !!window.chartData,
            "config": !!window.config,
            "OverviewController": !!window.OverviewController
        });
        
        // First check if OverviewController has already loaded model data
        if (window.OverviewController && window.OverviewController.modelData && 
            Object.keys(window.OverviewController.modelData).length > 0) {
            console.log("Using data from OverviewController");
            modelData = window.OverviewController.modelData;
        }
        // Try initial_results first
        else if (window.reportData && window.reportData.initial_results && window.reportData.initial_results.models) {
            console.log("Using model data from reportData.initial_results");
            modelData = window.reportData.initial_results.models;
        }
        // Try radar_chart_data
        else if (window.chartData && window.chartData.radar_chart_data && window.chartData.radar_chart_data.models) {
            console.log("Using model data from chartData.radar_chart_data");
            modelData = window.chartData.radar_chart_data.models;
        }
        // Try window.reportData
        else if (window.reportData && window.reportData.models) {
            modelData = window.reportData.models;
            console.log("Using model data from reportData");
        } else if (window.chartData && window.chartData.models) {
            // Try chart data
            modelData = window.chartData.models;
            console.log("Using model data from chartData");
        } else if (window.config && window.config.models) {
            // Try config
            modelData = window.config.models;
            console.log("Using model data from config");
        } else {
            console.warn("No model data found in any data source");
            return null;
        }
        
        // Check if we have any model data
        if (Object.keys(modelData).length === 0) {
            console.warn("Empty model data");
            return null;
        }
        
        console.log("Found model data:", Object.keys(modelData).join(", "));
        
        console.log("Found data for", Object.keys(modelData).length, "models");
        
        try {
            // Convert to array format for plotting
            models = Object.entries(modelData).map(([key, model]) => ({
                id: key,
                name: model.name || key,
                type: model.type || "Unknown",
                metrics: model.metrics || {}
            }));
            
            // Define metrics to display
            const metrics = [
                { name: "accuracy", displayName: "Accuracy" },
                { name: "roc_auc", displayName: "ROC AUC" },
                { name: "f1", displayName: "F1 Score" },
                { name: "precision", displayName: "Precision" },
                { name: "recall", displayName: "Recall" }
            ];
            
            // Validate that each model has the required metrics
            models.forEach(model => {
                metrics.forEach(metric => {
                    if (model.metrics[metric.name] === undefined) {
                        model.metrics[metric.name] = 0;
                        console.warn(`Missing ${metric.name} metric for model ${model.name}`);
                    }
                });
            });
            
            console.log("Prepared", models.length, "models with", metrics.length, "metrics each");
            
            return {
                models,
                metrics
            };
        } catch (error) {
            console.error("Error processing model metrics data:", error);
            return null;
        }
    },
    
    /**
     * Create a custom metric chart for a specific model and metric
     * @param {string} elementId - Chart container ID
     * @param {string} modelId - Model identifier
     * @param {string} metricName - Metric name to display
     */
    createModelMetricChart: function(elementId, modelId, metricName) {
        console.log(`Creating metric chart for ${modelId} - ${metricName}`);
        const chartElement = document.getElementById(elementId);
        if (!chartElement) {
            console.error("Chart element not found:", elementId);
            return;
        }
        
        // Extract data with specific model and metric filtering
        const chartData = this.extractMetricsRadarData();
        
        if (!chartData || !chartData.models || chartData.models.length === 0) {
            this.showNoDataMessage(chartElement, "No model metrics data available");
            return;
        }
        
        try {
            // Clear previous content
            chartElement.innerHTML = '';
            
            // Find the metric display name
            const metricInfo = chartData.metrics.find(m => m.name === metricName);
            if (!metricInfo) {
                this.showErrorMessage(chartElement, `Metric ${metricName} not found`);
                return;
            }
            
            // Filter models if specific model is requested
            let modelsToUse = chartData.models;
            if (modelId !== 'all') {
                modelsToUse = chartData.models.filter(model => model.id === modelId);
                if (modelsToUse.length === 0) {
                    this.showErrorMessage(chartElement, `Model ${modelId} not found`);
                    return;
                }
            }
            
            // Prepare data for the bar chart
            const chartTrace = {
                x: modelsToUse.map(model => model.name),
                y: modelsToUse.map(model => model.metrics[metricName]),
                type: 'bar',
                marker: {
                    color: modelsToUse.map((_, index) => {
                        // Generate colors based on index
                        const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe'];
                        return colors[index % colors.length];
                    })
                },
                hovertemplate: '<b>%{x}</b><br>' + 
                               metricInfo.displayName + ': %{y:.4f}<extra></extra>'
            };
            
            // Layout for the bar chart
            const layout = {
                title: `${metricInfo.displayName} por Modelo`,
                xaxis: {
                    title: 'Modelo',
                    automargin: true
                },
                yaxis: {
                    title: metricInfo.displayName,
                    range: [0, 1.1],
                    tickformat: '.2f'
                },
                margin: {
                    l: 60,
                    r: 30,
                    t: 60,
                    b: 80
                },
                plot_bgcolor: '#fafafa',
                paper_bgcolor: '#fff'
            };
            
            // Create the plot
            Plotly.newPlot(chartElement, [chartTrace], layout, {
                responsive: true,
                displayModeBar: false,
                displaylogo: false
            });
            
        } catch (error) {
            console.error("Error creating model metric chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Show no data message in chart container
     * @param {HTMLElement} element - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(element, message) {
        element.innerHTML = `
            <div class="no-data-container">
                <div class="no-data-icon">📊</div>
                <h3 class="no-data-title">Dados Não Disponíveis</h3>
                <p class="no-data-message">${message}</p>
            </div>`;
    },
    
    /**
     * Show error message in chart container
     * @param {HTMLElement} element - Chart container element
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(element, errorMessage) {
        element.innerHTML = `
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <h3 class="error-title">Erro no Gráfico</h3>
                <p class="error-message">${errorMessage}</p>
                <div class="error-help">
                    <p class="error-help-title">Possíveis soluções:</p>
                    <ul class="error-help-list">
                        <li>Verifique se a biblioteca Plotly está carregada corretamente</li>
                        <li>Confirme que os dados dos modelos estão disponíveis</li>
                        <li>Tente recarregar a página</li>
                    </ul>
                </div>
            </div>`;
    }
};

// PerturbationResultsManager.js
// Updated: May 7, 2024 - Removed synthetic data generation
const PerturbationResultsManager = {
    /**
     * Extract perturbation data from report data
     * @returns {Object} Perturbation data
     */
    extractPerturbationData: function() {
        console.log("Extracting perturbation data from report");
        const perturbationResults = [];
        
        try {
            // Check if window.reportData exists and has necessary data
            if (!window.reportData || (!window.reportData.raw && !window.reportData.perturbation_chart_data)) {
                console.warn("Report data not found or incomplete");
                console.warn("No synthetic data will be generated - will return null");
                return null; // Return null if no report data available
            }
            
            // If the server already prepared perturbation chart data, use it
            if (window.reportData.perturbation_chart_data) {
                console.log("Using server-prepared perturbation chart data");
                return this.processPreparedChartData(window.reportData.perturbation_chart_data);
            }
            
            // Extract from raw data if available
            if (window.reportData.raw && window.reportData.raw.by_level) {
                console.log("Extracting from raw perturbation data");
                
                // Get base score and metric
                const baseScore = window.reportData.base_score || 0.0;
                const metric = window.reportData.metric || 'Score';
                
                // Process each perturbation level
                Object.keys(window.reportData.raw.by_level).forEach(level => {
                    const numericLevel = parseFloat(level);
                    const levelData = window.reportData.raw.by_level[level];
                    
                    const resultItem = {
                        level: numericLevel,
                        allFeatures: this.extractFeatureData(levelData, 'all_features', baseScore),
                        featureSubset: this.extractFeatureData(levelData, 'feature_subset', baseScore)
                    };
                    
                    // If feature subset wasn't found, try to extract from selectedFeatures
                    if (!resultItem.featureSubset.iterations.length && window.reportData.feature_subset) {
                        const featureSubset = window.reportData.feature_subset;
                        if (Array.isArray(featureSubset) && featureSubset.length > 0) {
                            const subsetName = featureSubset.join('_');
                            resultItem.featureSubset = this.extractFeatureData(levelData, subsetName, baseScore);
                        }
                    }
                    
                    perturbationResults.push(resultItem);
                });
                
                // Sort by level
                perturbationResults.sort((a, b) => a.level - b.level);
                
                // Only return if we actually have results
                if (perturbationResults.length > 0) {
                    return {
                        results: perturbationResults,
                        baseScore: baseScore,
                        metric: metric,
                        modelName: window.reportData.model_name || 'Model',
                        modelType: window.reportData.model_type || 'Model',
                        featureSubset: window.reportData.feature_subset || []
                    };
                }
            }
            
            // If no perturbation data found, return null (no synthetic data)
            console.warn("No perturbation data found in report data");
            console.warn("No synthetic data will be generated - will return null");
            return null;
            
        } catch (error) {
            console.error("Error extracting perturbation data:", error);
            console.error("No synthetic data will be generated due to error - will return null");
            return null; // Return null on error
        }
    },
    
    /**
     * Process server-prepared chart data
     * @param {Object} chartData - Server-prepared chart data 
     * @returns {Object} Processed perturbation data
     */
    processPreparedChartData: function(chartData) {
        const perturbationResults = [];
        const baseScore = chartData.baseScore || 0.0;
        const metric = chartData.metric || 'Score';
        
        // Process levels and scores
        if (chartData.levels && chartData.scores) {
            chartData.levels.forEach((level, index) => {
                const resultItem = {
                    level: level,
                    allFeatures: {
                        baseScore: baseScore,
                        meanScore: chartData.scores[index] || 0,
                        impact: (baseScore - (chartData.scores[index] || 0)) / baseScore,
                        worstScore: chartData.worstScores ? chartData.worstScores[index] || 0 : 0,
                        iterations: []
                    },
                    featureSubset: {
                        baseScore: baseScore,
                        meanScore: 0,
                        impact: 0,
                        worstScore: 0,
                        iterations: []
                    }
                };
                
                // Add iterations if available
                if (window.reportData.iterations_by_level && window.reportData.iterations_by_level[level]) {
                    resultItem.allFeatures.iterations = window.reportData.iterations_by_level[level];
                } else if (chartData.iterations && chartData.iterations[index]) {
                    resultItem.allFeatures.iterations = chartData.iterations[index];
                }
                // No synthetic iterations will be created if none are available
                
                // Process feature subset if available
                if (chartData.alternativeModels && Object.keys(chartData.alternativeModels).length > 0) {
                    const subsetName = Object.keys(chartData.alternativeModels)[0];
                    const subsetData = chartData.alternativeModels[subsetName];
                    
                    resultItem.featureSubset = {
                        baseScore: subsetData.baseScore || baseScore,
                        meanScore: subsetData.scores[index] || 0,
                        impact: (baseScore - (subsetData.scores[index] || 0)) / baseScore,
                        worstScore: subsetData.worstScores ? subsetData.worstScores[index] || 0 : 0,
                        iterations: []
                    };
                    
                    // Only add real iteration data, no synthetic data
                    if (subsetData.iterations && subsetData.iterations[index]) {
                        resultItem.featureSubset.iterations = subsetData.iterations[index];
                    }
                }
                
                perturbationResults.push(resultItem);
            });
        }
        
        return {
            results: perturbationResults,
            baseScore: baseScore,
            metric: metric,
            modelName: chartData.modelName || 'Model',
            modelType: window.reportData.model_type || 'Model',
            featureSubset: window.reportData.feature_subset || []
        };
    },
    
    /**
     * Extract feature data from level data
     * @param {Object} levelData - Level data
     * @param {string} featureKey - Feature key to extract
     * @param {number} baseScore - Base score
     * @returns {Object} Extracted feature data
     */
    extractFeatureData: function(levelData, featureKey, baseScore) {
        const result = {
            baseScore: baseScore,
            meanScore: 0,
            impact: 0,
            worstScore: 0,
            iterations: []
        };
        
        try {
            // Check if we have overall_result data
            if (levelData.overall_result && levelData.overall_result[featureKey]) {
                const featureData = levelData.overall_result[featureKey];
                
                result.meanScore = featureData.mean_score || featureData.perturbed_score || 0;
                result.worstScore = featureData.worst_score || featureData.min_score || result.meanScore;
                result.impact = (baseScore - result.meanScore) / baseScore;
                
                // If negative impact (improvement), cap at a reasonable value
                if (result.impact < -0.1) result.impact = -0.1;
            }
            
            // Extract iteration data if available - NO synthetic data generation
            if (levelData.runs && levelData.runs[featureKey] && 
                levelData.runs[featureKey][0] && 
                levelData.runs[featureKey][0].iterations &&
                levelData.runs[featureKey][0].iterations.scores) {
                    
                result.iterations = levelData.runs[featureKey][0].iterations.scores;
            }
            // No synthetic iterations data will be created
        } catch (error) {
            console.error(`Error extracting ${featureKey} data:`, error);
        }
        
        return result;
    },
    
    /**
     * Format number with specified precision
     * @param {number} num - Number to format
     * @param {number} precision - Number of decimal places
     * @returns {string} Formatted number
     */
    formatNumber: function(num, precision = 4) {
        return Number(num).toFixed(precision);
    },
    
    /**
     * Get color class based on impact
     * @param {number} impact - Impact value
     * @returns {string} CSS class for coloring
     */
    getImpactColorClass: function(impact) {
        if (impact < 0) return 'text-green-600'; // Improvement
        if (impact < 0.03) return 'text-blue-600'; // Small degradation
        if (impact < 0.07) return 'text-yellow-600'; // Medium degradation
        return 'text-red-600'; // Large degradation
    },
    
    /**
     * Get background color class based on score comparison
     * @param {number} score - Score to compare
     * @param {number} baseScore - Base score for comparison
     * @returns {string} CSS class for background coloring
     */
    getScoreBgColorClass: function(score, baseScore) {
        const diff = score - baseScore;
        if (diff > 0) return 'bg-green-100';
        if (diff > -0.01) return 'bg-yellow-50';
        if (diff > -0.03) return 'bg-orange-50';
        return 'bg-red-50';
    }
};