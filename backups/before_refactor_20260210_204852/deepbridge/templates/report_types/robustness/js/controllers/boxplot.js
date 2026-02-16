// BoxplotController.js
// Updated version that handles all boxplot functionality
// Last updated: 2024-05-07

const BoxplotController = {
    // Track initialization state
    hasInitialized: false,
    
    /**
     * Initialize the boxplot section
     */
    init: function() {
        console.log("Boxplot section initialized");
        
        if (this.hasInitialized) {
            console.log("BoxplotController already initialized, skipping");
            return;
        }
        
        this.hasInitialized = true;
        
        // Verify if boxplot section is visible
        const boxplotSection = document.getElementById('boxplot');
        if (boxplotSection) {
            console.log("Boxplot section found:", {
                isVisible: boxplotSection.classList.contains('active'),
                display: window.getComputedStyle(boxplotSection).display
            });
            
            // Add listener for tab click to handle lazy loading
            const boxplotTabButton = document.querySelector('[data-tab="boxplot"]');
            if (boxplotTabButton) {
                boxplotTabButton.addEventListener('click', () => {
                    console.log("Boxplot tab clicked, initializing charts");
                    this.initializeCharts();
                    this.populateStatsTable();
                });
            }
            
            // Listen for tab change events
            document.addEventListener('tabchange', (event) => {
                if (event.detail && event.detail.tabId === 'boxplot') {
                    console.log("Tab changed to boxplot");
                    this.initializeCharts();
                    this.populateStatsTable();
                }
            });
        }
        
        // Initialize immediately if tab is active
        if (boxplotSection && boxplotSection.classList.contains('active')) {
            this.initializeCharts();
            this.populateStatsTable();
        }
    },
    
    /**
     * Initialize boxplot chart
     */
    initializeCharts: function() {
        const container = document.getElementById('boxplot-chart-container');
        if (!container) {
            console.error("Boxplot chart container not found");
            return;
        }
        
        // Check if Plotly is loaded
        if (typeof Plotly === 'undefined') {
            console.log("Plotly not available, loading from CDN");
            
            const script = document.createElement('script');
            script.src = 'https://cdn.plot.ly/plotly-2.29.1.min.js';
            script.onload = () => {
                console.log("Plotly loaded successfully, rendering chart");
                this.renderBoxplotChart(container);
            };
            script.onerror = () => {
                console.error("Failed to load Plotly");
                this.showErrorMessage(container, "Biblioteca de visualização Plotly não pôde ser carregada");
            };
            
            document.head.appendChild(script);
        } else {
            // Plotly is already available
            this.renderBoxplotChart(container);
        }
    },
    
    /**
     * Render boxplot chart
     * @param {HTMLElement} container - Chart container element
     */
    renderBoxplotChart: function(container) {
        // Extract data from reportData or chartData
        const boxplotData = this.extractBoxplotData();
        
        if (!boxplotData || !boxplotData.models || boxplotData.models.length === 0) {
            console.error("No data available for boxplot visualization");
            this.showNoDataMessage(container, "Não há dados disponíveis para visualização do boxplot");
            return;
        }
        
        const models = boxplotData.models;
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
                console.error(`Model ${model.name} has no scores, skipping`);
                return;
            }
            
            validModelCount++;
            
            // Clean up model name for display
            const displayName = model.name.replace(/_/g, ' ').trim();
            
            // Get color or generate a deterministic color based on model name
            let color;
            if (modelColors[model.name]) {
                color = modelColors[model.name];
            } else {
                // Generate a deterministic color based on the model name
                const hash = Array.from(model.name).reduce((hash, char) => {
                    return ((hash << 5) - hash) + char.charCodeAt(0);
                }, 0);
                const r = Math.abs(hash) % 200 + 55; // 55-255 range to avoid too dark or light
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
            console.error("No models with valid scores found");
            this.showNoDataMessage(container, "Nenhum modelo possui dados reais para visualização");
            return;
        }
        
        // Get metric name
        const metricName = boxplotData.metricName || 
                           window.reportData?.metric ||
                           'Score';
        
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
     * Extract boxplot data from report data
     * @returns {Object} Data for boxplot chart
     */
    extractBoxplotData: function() {
        try {
            // First check if we have processed boxplot data
            if (window.reportData && window.reportData.boxplot_data && window.reportData.boxplot_data.models) {
                console.log("Using server-prepared boxplot data");

                // Filter models to only include those with real data
                const validModels = window.reportData.boxplot_data.models.filter(model =>
                    model.scores && model.scores.length > 0
                );

                // Return null if no valid models
                if (validModels.length === 0) {
                    console.error("No models with valid scores found in boxplot_data");
                    return null;
                }

                return {
                    models: validModels,
                    metricName: window.reportData.metric || 'Score'
                };
            }

            // Check for boxplot data in chartData
            if (window.chartData && window.chartData.boxplot_data && window.chartData.boxplot_data.models) {
                console.log("Using chart data boxplot_data");

                // Filter models to only include those with real data
                const validModels = window.chartData.boxplot_data.models.filter(model =>
                    model.scores && model.scores.length > 0
                );

                // Return null if no valid models
                if (validModels.length === 0) {
                    console.error("No models with valid scores found in chartData.boxplot_data");
                    return null;
                }

                return {
                    models: validModels,
                    metricName: window.reportData?.metric || 'Score'
                };
            }

            // If no boxplot_data, try to extract from raw results
            console.log("No pre-processed boxplot data found, extracting from raw results");

            if (!window.reportData || !window.reportData.raw || !window.reportData.raw.by_level) {
                console.warn("No raw data available for boxplot extraction");
                return null;
            }

            // Get metric name
            const metricName = window.reportData.metric || 'Score';
            console.log(`Using metric: ${metricName}`);

            // Extract primary model data
            const primaryModelData = {
                name: window.reportData.model_name || 'Primary Model',
                modelType: window.reportData.model_type || 'Unknown',
                baseScore: window.reportData.base_score || 0,
                scores: []
            };

            // Extract iteration scores from each perturbation level
            const rawData = window.reportData.raw.by_level;
            Object.keys(rawData).forEach(level => {
                const levelData = rawData[level];

                if (!levelData.runs || !levelData.runs.all_features) {
                    console.log(`Level ${level}: No runs data found`);
                    return;
                }

                // Extract scores from all runs at this level
                levelData.runs.all_features.forEach(run => {
                    if (run.iterations && run.iterations.scores && run.iterations.scores.length > 0) {
                        console.log(`Level ${level}: Found ${run.iterations.scores.length} iteration scores`);
                        primaryModelData.scores.push(...run.iterations.scores);
                    }
                });
            });

            // Check for iteration data in chartData
            if (primaryModelData.scores.length === 0 && window.chartData && window.chartData.iterations_by_level) {
                console.log("Trying to extract primary model scores from chartData.iterations_by_level");
                Object.values(window.chartData.iterations_by_level).forEach(levelScores => {
                    if (Array.isArray(levelScores) && levelScores.length > 0) {
                        primaryModelData.scores.push(...levelScores);
                    }
                });
                console.log(`Extracted ${primaryModelData.scores.length} scores from chartData for primary model`);
            }

            // Try to find data in initial_results
            if (primaryModelData.scores.length === 0 &&
                window.reportData.initial_results &&
                window.reportData.initial_results.models) {

                console.log("Looking for primary model scores in initial_results");
                // Find the primary model
                const primaryModel = Object.values(window.reportData.initial_results.models).find(
                    model => model.name === primaryModelData.name || model.is_primary
                );

                if (primaryModel && primaryModel.evaluation_results && primaryModel.evaluation_results.scores) {
                    console.log(`Found ${primaryModel.evaluation_results.scores.length} scores in initial_results`);
                    primaryModelData.scores = primaryModel.evaluation_results.scores;
                }
            }

            console.log(`Primary model: extracted ${primaryModelData.scores.length} total scores`);

            // Se não temos scores reais para o modelo primário, não gerar dados sintéticos
            if (primaryModelData.scores.length === 0) {
                console.error("Nenhum score real encontrado para o modelo primário. Não serão gerados dados sintéticos.");
                // We keep the array empty - no synthetic data
            }

            const models = [];
            models.push(primaryModelData);

            // Extract alternative models data
            if (window.reportData.alternative_models) {
                console.log("Processing alternative models:", Object.keys(window.reportData.alternative_models));

                Object.keys(window.reportData.alternative_models).forEach(modelName => {
                    const modelData = window.reportData.alternative_models[modelName];
                    console.log(`Processing alternative model ${modelName}, data keys:`, Object.keys(modelData));

                    const altModelData = {
                        name: modelName,
                        modelType: modelData.model_type || 'Unknown',
                        baseScore: modelData.base_score || 0,
                        scores: []
                    };

                    // Extract scores from alternative model's raw data
                    if (modelData.raw && modelData.raw.by_level) {
                        Object.keys(modelData.raw.by_level).forEach(level => {
                            const levelData = modelData.raw.by_level[level];

                            if (levelData.runs && levelData.runs.all_features) {
                                levelData.runs.all_features.forEach(run => {
                                    if (run.iterations && run.iterations.scores && run.iterations.scores.length > 0) {
                                        console.log(`Found ${run.iterations.scores.length} scores for model ${modelName} at level ${level}`);
                                        altModelData.scores.push(...run.iterations.scores);
                                    }
                                });
                            } else if (levelData.overall_result && levelData.overall_result.all_features) {
                                const score = levelData.overall_result.all_features.mean_score;
                                if (score !== undefined) {
                                    console.log(`Using mean_score ${score} from overall_result for level ${level}`);
                                    altModelData.scores.push(score);
                                }
                            }
                        });
                    }

                    // Check for iteration data in chartData
                    if (altModelData.scores.length === 0 &&
                        window.chartData &&
                        window.chartData.alternative_models_iterations &&
                        window.chartData.alternative_models_iterations[modelName]) {

                        console.log(`Trying to extract scores for ${modelName} from chartData.alternative_models_iterations`);
                        Object.values(window.chartData.alternative_models_iterations[modelName]).forEach(levelScores => {
                            if (Array.isArray(levelScores) && levelScores.length > 0) {
                                altModelData.scores.push(...levelScores);
                            }
                        });
                        console.log(`Extracted ${altModelData.scores.length} scores for ${modelName} from chartData`);
                    }

                    // Try to find data in initial_results
                    if (altModelData.scores.length === 0 &&
                        window.reportData.initial_results &&
                        window.reportData.initial_results.models) {

                        console.log(`Looking for ${modelName} scores in initial_results`);
                        // Find the model by name
                        const foundModel = Object.values(window.reportData.initial_results.models).find(
                            model => model.name === modelName
                        );

                        if (foundModel && foundModel.evaluation_results && foundModel.evaluation_results.scores) {
                            console.log(`Found ${foundModel.evaluation_results.scores.length} scores in initial_results for ${modelName}`);
                            altModelData.scores = foundModel.evaluation_results.scores;
                        }
                    }

                    console.log(`Alternative model ${modelName}: extracted ${altModelData.scores.length} scores`);

                    // Se não há scores para o modelo alternativo, mostrar erro e não gerar dados sintéticos
                    if (altModelData.scores.length === 0) {
                        console.error(`Nenhum score encontrado para o modelo alternativo ${modelName}. Não serão gerados dados sintéticos.`);
                        // Still add the model to show its base score - with empty scores array
                    }

                    models.push(altModelData);
                });
            } else {
                console.log("No alternative models found in reportData");

                // Check for alternative models in initial_results
                if (window.reportData.initial_results && window.reportData.initial_results.models) {
                    const initialModels = window.reportData.initial_results.models;
                    console.log(`Found ${Object.keys(initialModels).length} models in initial_results`);

                    Object.values(initialModels).forEach(model => {
                        // Skip the primary model which we already processed
                        if (model.name === primaryModelData.name || model.is_primary) {
                            return;
                        }

                        console.log(`Processing initial_results model: ${model.name}`);

                        const altModelData = {
                            name: model.name,
                            modelType: model.type || 'Unknown',
                            baseScore: model.base_score || 0,
                            scores: []
                        };

                        if (model.evaluation_results && model.evaluation_results.scores) {
                            console.log(`Found ${model.evaluation_results.scores.length} scores for ${model.name}`);
                            altModelData.scores = model.evaluation_results.scores;
                        }

                        if (altModelData.scores.length === 0) {
                            console.error(`No scores found for model ${model.name} in initial_results`);
                        }

                        models.push(altModelData);
                    });
                }
            }

            // Se não houver modelos com scores, não gerar dados sintéticos
            if (models.length === 0 || !models.some(m => m.scores && m.scores.length > 0)) {
                console.error("Nenhum modelo com scores foi encontrado. Não serão criados dados sintéticos.");
                return null;
            }

            console.log(`Extracted data for ${models.length} models`);
            return { models, metricName };
        } catch (error) {
            console.error("Error extracting boxplot data:", error);
            return null; // Não gerar dados sintéticos em caso de erro
        }
    },
    
    /**
     * Populate the statistics table with model data
     */
    populateStatsTable: function() {
        const tableBody = document.getElementById('boxplot-table-body');
        if (!tableBody) return;
        
        try {
            // Clear existing content
            tableBody.innerHTML = '';
            
            // Get boxplot data
            const boxplotData = this.extractBoxplotData();
            if (!boxplotData || !boxplotData.models || boxplotData.models.length === 0) {
                this.showNoTableData(tableBody);
                return;
            }
            
            // Add rows for each model
            boxplotData.models.forEach(model => {
                // Skip models without real data
                if (!model.scores || model.scores.length === 0) return;
                
                // Sort scores for statistics
                const sortedScores = model.scores.slice().sort((a, b) => a - b);
                
                // Calculate basic stats
                const mean = sortedScores.reduce((a, b) => a + b, 0) / sortedScores.length;
                
                // Calculate median
                const mid = Math.floor(sortedScores.length / 2);
                const median = sortedScores.length % 2 === 0 ? 
                    (sortedScores[mid - 1] + sortedScores[mid]) / 2 : 
                    sortedScores[mid];
                
                // Calculate quartiles for IQR
                const q1Index = Math.floor(sortedScores.length * 0.25);
                const q3Index = Math.floor(sortedScores.length * 0.75);
                const q1 = sortedScores[q1Index] || 0;
                const q3 = sortedScores[q3Index] || 0;
                const iqr = q3 - q1;
                
                // Calculate min, max
                const min = sortedScores[0] || 0;
                const max = sortedScores[sortedScores.length - 1] || 0;
                
                // Calculate standard deviation
                let stdDev = 0;
                if (sortedScores.length > 1) {
                    const squaredDiffs = sortedScores.map(val => Math.pow(val - mean, 2));
                    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / sortedScores.length;
                    stdDev = Math.sqrt(variance);
                }
                
                // Create row
                const row = document.createElement('tr');
                
                // Model name - replace all underscores with spaces
                const nameCell = document.createElement('td');
                const displayName = model.name.replace(/_/g, ' ');
                nameCell.textContent = displayName || "Unknown Model";
                nameCell.style.fontWeight = 'bold';
                
                // Highlight primary model
                if (model.name === window.reportData?.model_name || model.name === 'Primary Model') {
                    nameCell.style.color = '#1b78de';
                }
                
                row.appendChild(nameCell);
                
                // Base score
                const baseScoreCell = document.createElement('td');
                baseScoreCell.textContent = model.baseScore ? model.baseScore.toFixed(4) : "N/A";
                row.appendChild(baseScoreCell);
                
                // Median
                const medianCell = document.createElement('td');
                medianCell.textContent = median.toFixed(4);
                row.appendChild(medianCell);
                
                // Mean
                const meanCell = document.createElement('td');
                meanCell.textContent = mean.toFixed(4);
                row.appendChild(meanCell);
                
                // IQR
                const iqrCell = document.createElement('td');
                iqrCell.textContent = iqr.toFixed(4);
                row.appendChild(iqrCell);
                
                // Min
                const minCell = document.createElement('td');
                minCell.textContent = min.toFixed(4);
                row.appendChild(minCell);
                
                // Max
                const maxCell = document.createElement('td');
                maxCell.textContent = max.toFixed(4);
                row.appendChild(maxCell);
                
                // Std Dev
                const stdDevCell = document.createElement('td');
                stdDevCell.textContent = stdDev.toFixed(4);
                row.appendChild(stdDevCell);
                
                // Score drop
                const dropCell = document.createElement('td');
                const baseScore = model.baseScore || 0;
                const dropPercent = (baseScore > 0) ? ((baseScore - median) / baseScore) * 100 : 0;
                dropCell.textContent = dropPercent.toFixed(2) + '%';
                dropCell.className = dropPercent > 5 ? 'text-danger' : (dropPercent > 2 ? 'text-warning' : 'text-success');
                row.appendChild(dropCell);
                
                tableBody.appendChild(row);
            });
        } catch (error) {
            console.error("Error populating stats table:", error);
            this.showTableError(tableBody);
        }
    },
    
    /**
     * Show no data message in container
     * @param {HTMLElement} container - Chart container
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(container, message) {
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border-radius: 8px; margin: 20px auto; max-width: 600px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #d32f2f;">Dados não disponíveis</h3>
                <p style="color: #333; font-size: 16px; line-height: 1.4;">${message}</p>
                <p style="color: #333; margin-top: 20px; font-size: 14px;">
                    Execute testes de robustez com iterações (n_iterations > 1) para visualizar dados reais.
                </p>
            </div>`;
    },
    
    /**
     * Show error message in container
     * @param {HTMLElement} container - Chart container
     * @param {string} errorMessage - Error message to display
     */
    showErrorMessage: function(container, errorMessage) {
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Chart Error</h3>
                <p style="color: #666; font-size: 16px; line-height: 1.4;">${errorMessage}</p>
            </div>`;
    },
    
    /**
     * Show no data message in table
     * @param {HTMLElement} tableBody - Table body element
     */
    showNoTableData: function(tableBody) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center">
                    <div class="loading-info">
                        <div class="loading-icon">⚠️</div>
                        <p>Dados de distribuição não disponíveis para modelos.</p>
                        <p>Execute testes com iterações (n_iterations > 1) para visualizar estatísticas.</p>
                    </div>
                </td>
            </tr>`;
    },
    
    /**
     * Show error message in table
     * @param {HTMLElement} tableBody - Table body element
     */
    showTableError: function(tableBody) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center">
                    <div class="loading-info">
                        <div class="loading-icon">⚠️</div>
                        <p>Erro ao carregar estatísticas dos modelos.</p>
                    </div>
                </td>
            </tr>`;
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    BoxplotController.init();
});