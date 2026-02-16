/**
 * Boxplot Chart Manager
 * Handles boxplot visualization logic for robustness reports
 * Updated: May 7, 2024 - Removed synthetic data generation
 */

const BoxplotChartManager = {
    /**
     * Initialize boxplot chart
     * @param {string} elementId - Chart container ID
     */
    initializeBoxplotChart: function(elementId) {
        console.log("BoxplotChartManager initializing boxplot chart in:", elementId);
        
        const container = document.getElementById(elementId);
        if (!container) {
            console.error("Chart container not found:", elementId);
            return;
        }
        
        try {
            // Extract data for boxplot
            const chartData = this.extractBoxplotData();
            
            if (!chartData || !chartData.models || chartData.models.length === 0 || 
                !chartData.models.some(m => m.scores && m.scores.length > 0)) {
                console.error("No valid chart data available");
                this.showNoDataMessage(container, "Dados de boxplot não disponíveis. Execute testes com iterações múltiplas para visualizar a distribuição dos scores.");
                return;
            }
            
            // Create the Plotly boxplot visualization
            this.createPlotlyBoxplot(container, chartData);
            
        } catch (error) {
            console.error("Error creating boxplot chart:", error);
            this.showErrorMessage(container, error.message);
        }
    },
    
    /**
     * Extract data for boxplot from report data
     * @returns {Object} Data for boxplot chart
     */
    extractBoxplotData: function() {
        try {
            // Get data from window.reportData or window.chartData
            const reportData = window.reportData || {};
            const chartData = window.chartData || {};

            // Check if we have pre-processed boxplot data
            if (chartData.boxplot_data && chartData.boxplot_data.models &&
                chartData.boxplot_data.models.length > 0) {
                console.log("Using pre-processed boxplot data");

                // Filter models to only include those with real scores
                const validModels = chartData.boxplot_data.models.filter(model =>
                    model.scores && model.scores.length > 0
                );

                if (validModels.length === 0) {
                    console.error("No models with valid scores found in boxplot_data");
                    return null;
                }

                return {
                    models: validModels,
                    metricName: reportData.metric || 'Score'
                };
            }

            // If no pre-processed data, try to extract from reportData
            if (reportData.boxplot_data && reportData.boxplot_data.models &&
                reportData.boxplot_data.models.length > 0) {
                console.log("Using reportData.boxplot_data");

                // Filter models to only include those with real scores
                const validModels = reportData.boxplot_data.models.filter(model =>
                    model.scores && model.scores.length > 0
                );

                if (validModels.length === 0) {
                    console.error("No models with valid scores found in reportData.boxplot_data");
                    return null;
                }

                return {
                    models: validModels,
                    metricName: reportData.metric || 'Score'
                };
            }

            // No pre-processed data, try to extract from raw data
            console.log("No pre-processed boxplot data, extracting from raw data");

            if (!reportData.raw || !reportData.raw.by_level) {
                console.error("No raw data available for boxplot extraction");
                return null;
            }

            // Extract primary model data
            const primaryModelData = {
                name: reportData.model_name || 'Primary Model',
                modelType: reportData.model_type || 'Unknown',
                baseScore: reportData.base_score || 0,
                scores: []
            };

            // Extract scores from perturbation levels
            Object.keys(reportData.raw.by_level).forEach(level => {
                const levelData = reportData.raw.by_level[level];

                if (levelData.runs && levelData.runs.all_features) {
                    levelData.runs.all_features.forEach(run => {
                        if (run.iterations && run.iterations.scores && run.iterations.scores.length > 0) {
                            primaryModelData.scores.push(...run.iterations.scores);
                        }
                    });
                }
            });

            // Check for iteration data in chartData
            if (primaryModelData.scores.length === 0 && chartData.iterations_by_level) {
                console.log("Trying to extract scores from chartData.iterations_by_level");
                Object.values(chartData.iterations_by_level).forEach(levelScores => {
                    if (Array.isArray(levelScores) && levelScores.length > 0) {
                        primaryModelData.scores.push(...levelScores);
                    }
                });
                console.log(`Extracted ${primaryModelData.scores.length} scores from chartData`);
            }

            if (primaryModelData.scores.length === 0) {
                console.log("No scores found for primary model, checking alternative sources");

                // Try to find scores in initial_results
                if (reportData.initial_results && reportData.initial_results.models) {
                    const primaryModel = Object.values(reportData.initial_results.models).find(
                        model => model.name === primaryModelData.name || model.is_primary
                    );

                    if (primaryModel && primaryModel.evaluation_results && primaryModel.evaluation_results.scores) {
                        console.log("Using scores from initial_results");
                        primaryModelData.scores = primaryModel.evaluation_results.scores;
                    }
                }
            }

            // Only proceed if we have real scores
            if (primaryModelData.scores.length === 0) {
                console.error("No real scores found for primary model");
                // Keep the model but with empty scores - don't generate synthetic data
            }

            const models = [primaryModelData];

            // Extract alternative models if available
            if (reportData.alternative_models) {
                Object.keys(reportData.alternative_models).forEach(modelName => {
                    const modelData = reportData.alternative_models[modelName];

                    const altModelData = {
                        name: modelName,
                        modelType: modelData.model_type || 'Unknown',
                        baseScore: modelData.base_score || 0,
                        scores: []
                    };

                    // Extract scores if raw data is available
                    if (modelData.raw && modelData.raw.by_level) {
                        Object.keys(modelData.raw.by_level).forEach(level => {
                            const levelData = modelData.raw.by_level[level];

                            if (levelData.runs && levelData.runs.all_features) {
                                levelData.runs.all_features.forEach(run => {
                                    if (run.iterations && run.iterations.scores && run.iterations.scores.length > 0) {
                                        altModelData.scores.push(...run.iterations.scores);
                                    }
                                });
                            }
                        });
                    }

                    // Check for iteration data in chartData
                    if (altModelData.scores.length === 0 &&
                        chartData.alternative_models_iterations &&
                        chartData.alternative_models_iterations[modelName]) {

                        console.log(`Trying to extract scores for ${modelName} from chartData.alternative_models_iterations`);
                        Object.values(chartData.alternative_models_iterations[modelName]).forEach(levelScores => {
                            if (Array.isArray(levelScores) && levelScores.length > 0) {
                                altModelData.scores.push(...levelScores);
                            }
                        });
                        console.log(`Extracted ${altModelData.scores.length} scores for ${modelName} from chartData`);
                    }

                    // Try to find scores in initial_results
                    if (altModelData.scores.length === 0 &&
                        reportData.initial_results &&
                        reportData.initial_results.models) {

                        const model = Object.values(reportData.initial_results.models).find(
                            m => m.name === modelName
                        );

                        if (model && model.evaluation_results && model.evaluation_results.scores) {
                            console.log(`Using scores from initial_results for ${modelName}`);
                            altModelData.scores = model.evaluation_results.scores;
                        }
                    }

                    // Only add if real scores are available
                    if (altModelData.scores.length === 0) {
                        console.error(`No real scores found for model ${modelName}`);
                        // Add model anyway but with empty scores - don't generate synthetic data
                    }

                    models.push(altModelData);
                });
            }

            // Ensure at least one model has valid scores
            if (!models.some(model => model.scores && model.scores.length > 0)) {
                console.error("No models with valid scores found");
                return null;
            }

            return {
                models,
                metricName: reportData.metric || 'Score'
            };

        } catch (error) {
            console.error("Error extracting boxplot data:", error);
            return null;
        }
    },
    
    /**
     * Create Plotly boxplot visualization
     * @param {HTMLElement} container - Chart container element
     * @param {Object} chartData - Data for chart
     */
    createPlotlyBoxplot: function(container, chartData) {
        if (typeof Plotly === 'undefined') {
            console.error("Plotly is not available");
            this.showErrorMessage(container, "Plotly library is not available. Charts cannot be displayed.");
            return;
        }
        
        const models = chartData.models;
        const traces = [];
        
        // Define consistent colors for models
        const modelColors = {
            'Primary Model': 'rgba(31, 119, 180, 0.7)',
            'primary_model': 'rgba(31, 119, 180, 0.7)',
            'GLM_CLASSIFIER': 'rgba(255, 127, 14, 0.7)',
            'GAM_CLASSIFIER': 'rgba(44, 160, 44, 0.7)',
            'GBM': 'rgba(214, 39, 40, 0.7)',
            'XGB': 'rgba(148, 103, 189, 0.7)',
            'RANDOM_FOREST': 'rgba(140, 86, 75, 0.7)',
            'SVM': 'rgba(227, 119, 194, 0.7)',
            'NEURAL_NETWORK': 'rgba(127, 127, 127, 0.7)'
        };
        
        // Track valid models
        let validModelCount = 0;
        
        // Create traces for each model
        models.forEach(model => {
            // Skip models without real data
            if (!model.scores || model.scores.length === 0) {
                return;
            }
            
            validModelCount++;
            
            // Clean up model name for display
            const displayName = model.name.replace(/_/g, ' ').trim();
            
            // Get color or generate a deterministic color
            let color = modelColors[model.name];
            if (!color) {
                // Generate a deterministic color based on model name
                const hash = Array.from(model.name).reduce((hash, char) => {
                    return ((hash << 5) - hash) + char.charCodeAt(0);
                }, 0);
                const r = Math.abs(hash) % 200 + 55;
                const g = Math.abs(hash * 31) % 200 + 55;
                const b = Math.abs(hash * 17) % 200 + 55;
                color = `rgba(${r}, ${g}, ${b}, 0.7)`;
            }
            
            // Create violin plot for model
            traces.push({
                type: 'violin',
                y: model.scores,
                x: Array(model.scores.length).fill(displayName),
                name: displayName,
                box: {
                    visible: true,
                    width: 0.6
                },
                meanline: {
                    visible: true
                },
                line: {
                    color: 'black',
                    width: 1
                },
                fillcolor: color,
                opacity: 0.7,
                points: 'all',
                jitter: 0.3,
                pointpos: 0,
                hoverinfo: 'y+x',
                spanmode: 'soft',
                width: 0.5,
                bandwidth: 0.2
            });
        });
        
        // Add base scores as separate markers
        const baseScoreTrace = {
            type: 'scatter',
            mode: 'markers',
            y: models.map(m => m.baseScore),
            x: models.map(m => m.name.replace(/_/g, ' ').trim()),
            name: 'Base Score',
            marker: {
                size: 12,
                symbol: 'diamond',
                color: models.map(m => modelColors[m.name] || 'rgba(31, 119, 180, 0.7)'),
                line: {
                    color: 'white',
                    width: 1
                }
            },
            text: models.map(m => `Base Score: ${m.baseScore.toFixed(4)}`),
            hoverinfo: 'text+y',
        };
        
        traces.push(baseScoreTrace);
        
        // If no valid models, show error
        if (validModelCount === 0) {
            this.showNoDataMessage(container, "Nenhum modelo possui dados reais para visualização");
            return;
        }
        
        // Get metric name
        const metricName = chartData.metricName || 'Score';
        
        // Create layout
        const layout = {
            title: {
                text: `Model Performance Distribution - ${metricName}`,
                font: { size: 20 }
            },
            xaxis: {
                title: 'Models',
                tickangle: 0,
                automargin: true,
            },
            yaxis: {
                title: metricName,
                zeroline: false,
                autorange: true,
                automargin: true
            },
            autosize: true,
            violinmode: 'group',
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
                b: 80
            },
            annotations: [{
                xref: 'paper',
                yref: 'paper',
                x: 0,
                y: -0.15,
                text: 'The boxplots show model performance distribution under perturbation tests. Diamond markers indicate base scores.',
                showarrow: false,
                font: { size: 12 }
            }]
        };
        
        try {
            // Render the visualization
            Plotly.newPlot(container, traces, layout, {
                responsive: true,
                displayModeBar: true,
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
            }).then(() => {
                console.log("Boxplot chart successfully rendered");
                
                // Force a resize event to ensure proper layout
                window.dispatchEvent(new Event('resize'));
            }).catch(error => {
                console.error("Plotly.newPlot failed:", error);
                this.showErrorMessage(container, `Error rendering boxplot: ${error.message}`);
            });
        } catch (error) {
            console.error("Exception during Plotly.newPlot:", error);
            this.showErrorMessage(container, `Error rendering boxplot: ${error.message}`);
        }
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
     * Show no data message in container
     * @param {HTMLElement} element - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(element, message) {
        element.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Dados não disponíveis</h3>
                <p style="color: #333; font-size: 16px; line-height: 1.4;">${message}</p>
                <p style="color: #333; margin-top: 20px; font-size: 14px;">
                    Não serão gerados dados sintéticos ou demonstrativos. Execute testes com iterações múltiplas (n_iterations > 1).
                </p>
            </div>`;
    },
    
    /**
     * Show error message in container
     * @param {HTMLElement} element - Chart container element
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(element, errorMessage) {
        element.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Erro ao criar gráfico</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">${errorMessage}</p>
            </div>`;
    }
};