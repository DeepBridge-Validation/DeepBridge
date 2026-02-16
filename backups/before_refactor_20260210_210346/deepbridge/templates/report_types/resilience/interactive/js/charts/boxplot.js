// BoxplotChartManager.js
const BoxplotChartManager = {
    /**
     * Initialize model comparison boxplot chart
     * @param {string} elementId - Chart container ID
     */
    initializeBoxplotChart: function(elementId) {
        console.log("Initializing model comparison boxplot chart");
        const chartElement = document.getElementById(elementId);
        if (!chartElement) {
            console.error("Chart element not found:", elementId);
            return;
        }
        
        try {
            // Extract data for boxplot
            const chartData = this.extractBoxplotData();
            
            if (!chartData || !chartData.models || chartData.models.length === 0) {
                this.showNoDataMessage(chartElement, "No boxplot data available");
                return;
            }
            
            // Create the Plotly boxplot visualization
            this.createPlotlyBoxplot(chartElement, chartData);
            
        } catch (error) {
            console.error("Error creating boxplot chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Extract data for boxplot chart from report data
     */
    extractBoxplotData: function() {
        const models = [];
        let allScores = [];
        
        // Extract data from report data
        if (window.reportData) {
            // Check if boxplot data is already prepared by the server
            if (window.reportData.boxplot_data && window.reportData.boxplot_data.models && window.reportData.boxplot_data.models.length > 0) {
                console.log("Using server-prepared boxplot data");
                
                // Use the pre-processed data from the server
                const serverModels = window.reportData.boxplot_data.models;
                
                // Process each model
                serverModels.forEach(model => {
                    if (model.scores && model.scores.length > 0) {
                        models.push(model);
                        allScores.push(...model.scores);
                        allScores.push(model.baseScore);
                    }
                });
            } else {
                console.log("Server boxplot data not found, extracting from raw data");
                
                // Extract primary model data
                const primaryModelData = {
                    name: window.reportData.model_name || 'Primary Model',
                    modelType: window.reportData.model_type || 'Unknown',
                    baseScore: window.reportData.base_score || 0,
                    scores: []
                };
                
                // Extract scores from each perturbation level for primary model
                if (window.reportData.raw && window.reportData.raw.by_level) {
                    Object.keys(window.reportData.raw.by_level).forEach(level => {
                        const levelData = window.reportData.raw.by_level[level];
                        if (levelData.runs && levelData.runs.all_features && levelData.runs.all_features[0]) {
                            const iterationScores = levelData.runs.all_features[0].iterations.scores;
                            if (iterationScores && iterationScores.length) {
                                primaryModelData.scores.push(...iterationScores);
                            }
                        } else if (levelData.overall_result && levelData.overall_result.all_features) {
                            // If no iteration data, try to get overall result
                            const meanScore = levelData.overall_result.all_features.mean_score;
                            if (meanScore !== undefined) {
                                primaryModelData.scores.push(meanScore);
                            }
                        }
                    });
                }
                
                // Add primary model if it has scores
                if (primaryModelData.scores.length > 0) {
                    models.push(primaryModelData);
                    allScores.push(...primaryModelData.scores);
                    allScores.push(primaryModelData.baseScore);
                }
                
                // Extract alternative models data
                if (window.reportData.alternative_models) {
                    Object.keys(window.reportData.alternative_models).forEach(modelName => {
                        const modelData = window.reportData.alternative_models[modelName];
                        const altModelData = {
                            name: modelName,
                            modelType: modelData.model_type || 'Unknown',
                            baseScore: modelData.base_score || 0,
                            scores: []
                        };
                        
                        // Extract scores from each perturbation level
                        if (modelData.raw && modelData.raw.by_level) {
                            Object.keys(modelData.raw.by_level).forEach(level => {
                                const levelData = modelData.raw.by_level[level];
                                if (levelData.runs && levelData.runs.all_features && levelData.runs.all_features[0]) {
                                    const iterationScores = levelData.runs.all_features[0].iterations.scores;
                                    if (iterationScores && iterationScores.length) {
                                        altModelData.scores.push(...iterationScores);
                                    }
                                } else if (levelData.overall_result && levelData.overall_result.all_features) {
                                    // If no iteration data, try to get overall result
                                    const meanScore = levelData.overall_result.all_features.mean_score;
                                    if (meanScore !== undefined) {
                                        altModelData.scores.push(meanScore);
                                    }
                                }
                            });
                        }
                        
                        // Add alternative model if it has scores
                        if (altModelData.scores.length > 0) {
                            models.push(altModelData);
                            allScores.push(...altModelData.scores);
                            allScores.push(altModelData.baseScore);
                        }
                    });
                }
            }
        }
        
        // If we don't have any data, return null
        if (models.length === 0 || allScores.length === 0) {
            return null;
        }
        
        // Calculate min and max values for y-axis scale
        const minValue = Math.min(...allScores);
        const maxValue = Math.max(...models.map(model => model.baseScore), ...allScores);
        
        // Calculate y-axis scale with padding
        const yMin = Math.max(0, minValue - 0.05);
        const yMax = maxValue + 0.05;
        const yRange = yMax - yMin;
        
        return {
            models,
            allScores,
            yMin,
            yMax,
            yRange
        };
    },
    
    /**
     * Calculate boxplot statistics for a set of scores
     * @param {Array} scores - Array of score values
     * @returns {Object} Boxplot statistics
     */
    calculateBoxplotStats: function(scores) {
        if (!scores || scores.length === 0) return null;
        
        // Sort scores for percentile calculations
        const sortedScores = [...scores].sort((a, b) => a - b);
        
        // Calculate statistics
        const min = sortedScores[0];
        const max = sortedScores[sortedScores.length - 1];
        
        // Find quartiles
        const getPercentile = (arr, p) => {
            const index = Math.floor(arr.length * p);
            return arr[index];
        };
        
        const q1 = getPercentile(sortedScores, 0.25);
        const median = getPercentile(sortedScores, 0.5);
        const q3 = getPercentile(sortedScores, 0.75);
        
        // Calculate IQR (Interquartile Range)
        const iqr = q3 - q1;
        
        // Calculate whiskers (using Tukey's method: 1.5 * IQR)
        const lowerWhisker = Math.max(min, q1 - 1.5 * iqr);
        const upperWhisker = Math.min(max, q3 + 1.5 * iqr);
        
        // Find outliers
        const outliers = sortedScores.filter(score => score < lowerWhisker || score > upperWhisker);
        
        return {
            min,
            max,
            q1,
            median,
            q3,
            iqr,
            lowerWhisker,
            upperWhisker,
            outliers
        };
    },
    
    /**
     * Create Plotly boxplot visualization
     * @param {HTMLElement} chartElement - Chart container element
     * @param {Object} chartData - Data for the chart
     */
    createPlotlyBoxplot: function(chartElement, chartData) {
        // Dynamic load Plotly if not available
        if (typeof Plotly === 'undefined') {
            console.log("Plotly not found, loading from CDN");
            const script = document.createElement('script');
            script.src = 'https://cdn.plot.ly/plotly-latest.min.js';
            script.onload = () => this.renderPlotlyBoxplot(chartElement, chartData);
            document.head.appendChild(script);
        } else {
            this.renderPlotlyBoxplot(chartElement, chartData);
        }
    },
    
    /**
     * Render Plotly boxplot
     * @param {HTMLElement} chartElement - Chart container element
     * @param {Object} chartData - Data for the chart
     */
    renderPlotlyBoxplot: function(chartElement, chartData) {
        const models = chartData.models;
        const plotData = [];
        const modelColors = {
            'Primary Model': 'rgba(31, 119, 180, 0.7)',
            'primary_model': 'rgba(31, 119, 180, 0.7)',
            'GLM_CLASSIFIER': 'rgba(255, 127, 14, 0.7)',
            'GAM_CLASSIFIER': 'rgba(44, 160, 44, 0.7)',
            'GBM': 'rgba(214, 39, 40, 0.7)'
        };
        
        // Create base score markers
        const baseScores = {
            x: [],
            y: [],
            text: [],
            type: 'scatter',
            mode: 'markers',
            marker: {
                color: 'rgb(8, 81, 156)',
                size: 10,
                symbol: 'diamond'
            },
            name: 'Base Score'
        };
        
        // Create model name display mapping
        const modelDisplayNames = {};
        models.forEach(model => {
            const displayName = model.name.replace('_', ' ');
            modelDisplayNames[model.name] = displayName;
        });
        
        // Create violin+box trace for each model
        models.forEach((model, index) => {
            const displayName = modelDisplayNames[model.name] || model.name;
            const color = modelColors[model.name] || `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, 0.7)`;
            
            // Create violin + box trace
            plotData.push({
                type: 'violin',
                x: Array(model.scores.length).fill(displayName),
                y: model.scores,
                name: displayName,
                box: {
                    visible: true,
                    width: 0.6
                },
                meanline: {
                    visible: true
                },
                line: {
                    color: 'black'
                },
                fillcolor: color,
                opacity: 0.6,
                points: 'all',
                jitter: 0.3,
                pointpos: 0,
                hoverinfo: 'y+name',
                spanmode: 'soft',
                width: 0.25,
                bandwidth: 0.03
            });
            
            // Add base score marker
            baseScores.x.push(displayName);
            baseScores.y.push(model.baseScore);
            baseScores.text.push(`Base Score: ${model.baseScore.toFixed(4)}`);
        });
        
        // Add base scores as a separate trace
        plotData.push(baseScores);
        
        // Get metric name
        const metricName = window.reportData && window.reportData.metric ? window.reportData.metric : 'Score';
        
        // Create layout
        const layout = {
            title: {
                text: `Model Performance Comparison - ${metricName} Scores`,
                font: { size: 20 }
            },
            xaxis: {
                title: '',
                tickangle: 0,
                automargin: true
            },
            yaxis: {
                title: metricName,
                zeroline: false,
                autorange: true,
                automargin: true
            },
            autosize: true,
            violinmode: 'group',
            violingap: 0.1,
            violingroupgap: 0.05,
            hoverlabel: {
                bgcolor: "#FFF",
                font: { size: 12 },
                bordercolor: "#333"
            },
            showlegend: true,
            legend: {
                orientation: "h",
                yanchor: "top",
                y: 1.1,
                xanchor: "right",
                x: 1
            },
            hovermode: 'closest',
                margin: {
                    l: 50,
                    r: 20,
                    t: 60,
                    b: 100
                },
            annotations: [{
                xref: 'paper',
                yref: 'paper',
                x: 0,
                y: -0.15,
                text: 'The boxplots show model performance distribution under perturbation tests. Higher stability indicates better robustness.',
                showarrow: false,
                font: { size: 12 }
            }]
        };
        
        // Render the visualization
        Plotly.newPlot(chartElement, plotData, layout, {
            responsive: true,
            displayModeBar: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false,
            staticPlot: false,
            toImageButtonOptions: {
                format: 'png',
                filename: 'model_comparison_boxplot',
                height: 700,
                width: 1000,
                scale: 2
            }
        });
        
        // Add legend explanation after the chart
        this.addPlotlyLegend(chartElement);
    },
    
    /**
     * Add legend explanation after the chart
     * @param {HTMLElement} chartElement - Chart container element
     */
    addPlotlyLegend: function(chartElement) {
        const legendDiv = document.createElement('div');
        legendDiv.className = 'boxplot-legend mt-4';
        
        
        // Add the legend after the chart
        chartElement.insertAdjacentElement('afterend', legendDiv);
    },
    
    /**
     * Show no data message in chart container
     * @param {HTMLElement} element - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(element, message) {
        element.innerHTML = `
            <div class="p-8 text-center">
                <div class="inline-block p-4 bg-gray-100 rounded-lg shadow-sm">
                    <span class="text-3xl mb-2">📊</span>
                    <h3 class="text-lg font-semibold">No Data Available</h3>
                    <p class="text-gray-600">${message}</p>
                </div>
            </div>`;
    },
    
    /**
     * Show error message in chart container
     * @param {HTMLElement} element - Chart container element
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(element, errorMessage) {
        element.innerHTML = `
            <div class="p-8 text-center text-red-500">
                Error creating chart: ${errorMessage}
            </div>`;
    }
};