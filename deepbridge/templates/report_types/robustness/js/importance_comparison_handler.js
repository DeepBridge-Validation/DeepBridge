// Importance Comparison Handler
// This is a standalone script that handles the importance comparison visualization

(function() {
    // Create the handler
    window.ImportanceComparisonHandler = {
        /**
         * Initialize the importance comparison visualization
         */
        initialize: function() {
            console.log("Initializing importance comparison handler");
            
            try {
                // Initialize the importance comparison chart
                this.initializeImportanceComparisonChart();
                
                console.log("Importance comparison handler initialized");
            } catch (error) {
                console.error("Error initializing importance comparison handler:", error);
            }
        },
        
        /**
         * Initialize the importance comparison chart using Plotly
         */
        initializeImportanceComparisonChart: function() {
            try {
                const chartContainer = document.getElementById('importance-comparison-chart-plot');
                if (!chartContainer) return;
                
                // Get feature importance data
                const chartData = this.extractImportanceComparisonData();
                if (!chartData || !chartData.features || chartData.features.length === 0) {
                    this.showNoDataMessage(chartContainer, "No feature importance comparison data available");
                    return;
                }
                
                // Create chart
                this.renderImportanceComparisonChart(chartContainer, chartData);
            } catch (error) {
                console.error("Error initializing importance comparison chart:", error);
                const container = document.getElementById('importance-comparison-chart-plot');
                if (container) {
                    this.showErrorMessage(container, "Error initializing chart: " + error.message);
                }
            }
        },
        
        /**
         * Extract importance comparison data from report data
         * @returns {Object} Importance comparison data
         */
        extractImportanceComparisonData: function() {
            try {
                // Try to get data from various sources
                let featureImportance = {};
                let modelFeatureImportance = {};
                
                // Function to check if we have enough usable feature data
                const hasEnoughFeatureData = (data) => {
                    return data && typeof data === 'object' && 
                           Object.keys(data).length > 0 && 
                           Object.keys(data).length < 1000; // Sanity check to avoid giant objects
                };
                
                // Check multiple sources in order of preference
                if (window.reportConfig && hasEnoughFeatureData(window.reportConfig.feature_importance)) {
                    featureImportance = window.reportConfig.feature_importance;
                    modelFeatureImportance = window.reportConfig.model_feature_importance || {};
                    console.log("Using feature importance from reportConfig");
                } 
                else if (window.chartData && hasEnoughFeatureData(window.chartData.feature_importance)) {
                    featureImportance = window.chartData.feature_importance;
                    modelFeatureImportance = window.chartData.model_feature_importance || {};
                    console.log("Using feature importance from chartData");
                }
                else if (window.reportData) {
                    if (hasEnoughFeatureData(window.reportData.feature_importance)) {
                        featureImportance = window.reportData.feature_importance;
                        modelFeatureImportance = window.reportData.model_feature_importance || {};
                        console.log("Using feature importance from reportData");
                    }
                    else if (window.reportData.chart_data_json && typeof window.reportData.chart_data_json === 'string') {
                        // Try to parse the chart_data_json string
                        try {
                            const chartData = JSON.parse(window.reportData.chart_data_json);
                            if (chartData && hasEnoughFeatureData(chartData.feature_importance)) {
                                featureImportance = chartData.feature_importance;
                                modelFeatureImportance = chartData.model_feature_importance || {};
                                console.log("Using feature importance from parsed chart_data_json");
                            }
                        } catch (e) {
                            console.error("Error parsing chart_data_json:", e);
                        }
                    }
                }
                
                // Try FeatureImportanceTableManager as a fallback
                if (Object.keys(featureImportance).length === 0 && 
                    typeof FeatureImportanceTableManager !== 'undefined' && 
                    typeof FeatureImportanceTableManager.extractFeatureData === 'function') {
                    try {
                        const featureData = FeatureImportanceTableManager.extractFeatureData();
                        if (featureData && featureData.length > 0) {
                            // Convert the array format back to object format
                            featureImportance = {};
                            modelFeatureImportance = {};
                            
                            featureData.forEach(item => {
                                featureImportance[item.name] = item.robustness;
                                modelFeatureImportance[item.name] = item.importance;
                            });
                            
                            console.log("Using feature importance from FeatureImportanceTableManager");
                        }
                    } catch (e) {
                        console.error("Error using FeatureImportanceTableManager:", e);
                    }
                }
                
                // Only if there's absolutely no data available, use empty objects
                // We will not generate synthetic data - we'll let the UI show a "no data" message
                if (Object.keys(featureImportance).length === 0) {
                    console.warn("No feature importance data available - returning null");
                    return null;
                }
                
                // Convert to arrays for plotting
                const featureArray = [];
                for (const feature in featureImportance) {
                    // We include all features with robustness values, even if model importance is missing
                    // (we'll just use 0 for missing model importance)
                    featureArray.push({
                        name: feature,
                        robustness: featureImportance[feature],
                        model: modelFeatureImportance[feature] || 0
                    });
                }
                
                // Sort by absolute robustness impact
                featureArray.sort((a, b) => Math.abs(b.robustness) - Math.abs(a.robustness));
                
                // Get top features (up to 10)
                const topFeatures = featureArray.slice(0, 10);
                
                // Check if we have any data
                if (topFeatures.length === 0) {
                    console.warn("No features data available after processing");
                    return null;
                }
                
                return {
                    features: topFeatures.map(f => f.name),
                    robustnessValues: topFeatures.map(f => f.robustness),
                    modelValues: topFeatures.map(f => f.model)
                };
            } catch (error) {
                console.error("Error extracting importance comparison data:", error);
                return null;
            }
        },
        
        /**
         * Render importance comparison chart using Plotly
         * @param {HTMLElement} container - Chart container element
         * @param {Object} chartData - Chart data object
         */
        renderImportanceComparisonChart: function(container, chartData) {
            try {
                // Verify Plotly is available
                if (typeof Plotly === 'undefined') {
                    this.showErrorMessage(container, "Plotly library not available");
                    return;
                }
                
                // Clear container
                container.innerHTML = '';
                
                // Verificar se já existe um gráfico e removê-lo completamente
                Plotly.purge(container);
                
                // Criar gráfico de dispersão (o primeiro estilo que aparece)
                // Este é um estilo de gráfico diferente que mostra a correlação
                // entre a importância do modelo e o impacto de robustez
                
                // Preparar dados para o gráfico de dispersão
                const scatterData = chartData.features.map((feature, index) => {
                    return {
                        feature,
                        model: chartData.modelValues[index], 
                        robustness: chartData.robustnessValues[index]
                    };
                });
                
                // Criar trace de dispersão
                const scatterTrace = {
                    x: chartData.modelValues,
                    y: chartData.robustnessValues,
                    mode: 'markers+text',
                    type: 'scatter',
                    text: chartData.features,
                    textposition: 'top center',
                    textfont: {
                        family: 'Arial, sans-serif',
                        size: 10,
                        color: 'rgba(0, 0, 0, 0.7)'
                    },
                    marker: {
                        size: 12,
                        color: 'rgba(93, 109, 235, 0.8)',
                        line: {
                            color: 'rgba(0, 0, 0, 0.5)',
                            width: 1
                        }
                    },
                    name: 'Features'
                };
                
                // Criar linha de referência diagonal (onde model importance = robustness impact)
                // Encontrar valores mínimos e máximos para os eixos
                const allValues = [...chartData.modelValues, ...chartData.robustnessValues];
                const minVal = Math.min(...allValues.filter(v => !isNaN(v)));
                const maxVal = Math.max(...allValues.filter(v => !isNaN(v)));
                
                const refLine = {
                    x: [minVal, maxVal],
                    y: [minVal, maxVal],
                    mode: 'lines',
                    type: 'scatter',
                    line: {
                        color: 'rgba(200, 200, 200, 0.5)',
                        dash: 'dash',
                        width: 1
                    },
                    showlegend: false
                };
                
                // Layout para gráfico de dispersão
                const layout = {
                    title: 'Model Importance vs Robustness Impact',
                    xaxis: {
                        title: 'Model Importance',
                        zeroline: true
                    },
                    yaxis: {
                        title: 'Robustness Impact',
                        zeroline: true
                    },
                    hovermode: 'closest',
                    legend: {
                        orientation: 'h',
                        y: -0.2
                    },
                    margin: {
                        l: 60,
                        r: 20,
                        t: 60,
                        b: 80
                    },
                    annotations: [{
                        xref: 'paper',
                        yref: 'paper',
                        x: 0.02,
                        y: 0.98,
                        text: 'Features above line: higher robustness impact than model importance',
                        showarrow: false,
                        font: {
                            size: 12,
                            color: 'rgba(0, 0, 0, 0.6)'
                        }
                    }]
                };
                
                // Create plot with scatter chart and reference line
                Plotly.newPlot(container, [scatterTrace, refLine], layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtons: [[
                        'zoom2d', 
                        'pan2d', 
                        'resetScale2d', 
                        'toImage'
                    ]]
                });
                
                console.log("Importance comparison chart rendered successfully");
            } catch (error) {
                console.error("Error rendering importance comparison chart:", error);
                this.showErrorMessage(container, "Error rendering chart: " + error.message);
            }
        },
        
        /**
         * Show no data message in container
         * @param {HTMLElement} container - Container element
         * @param {string} message - Message to display
         */
        showNoDataMessage: function(container, message) {
            container.innerHTML = `
                <div style="padding: 40px; text-align: center; background-color: #f8f9fa; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
                    <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px;">Dados não disponíveis</h3>
                    <p style="color: #666; font-size: 16px; line-height: 1.4;">
                        ${message}
                    </p>
                    <p style="color: #666; font-size: 14px; margin-top: 20px;">
                        Nenhum dado sintético será gerado. Execute testes com cálculo de importância de características habilitado para visualizar dados reais.
                    </p>
                </div>`;
        },
        
        /**
         * Show error message in container
         * @param {HTMLElement} container - Container element
         * @param {string} errorMessage - Error message to display
         */
        showErrorMessage: function(container, errorMessage) {
            container.innerHTML = `
                <div style="padding: 40px; text-align: center; background-color: #fff0f0; border: 1px solid #ffcccc; border-radius: 8px; margin: 20px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 48px; margin-bottom: 20px;">⚠️</div>
                    <h3 style="font-size: 24px; font-weight: bold; margin-bottom: 10px; color: #cc0000;">Chart Error</h3>
                    <p style="color: #666; font-size: 16px; line-height: 1.4;">${errorMessage}</p>
                </div>`;
        }
    };
    
    // Initialize when document is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize after a small delay to ensure everything is loaded
        setTimeout(function() {
            if (document.getElementById('importance-comparison-chart-plot')) {
                window.ImportanceComparisonHandler.initialize();
            }
        }, 500);
    });
})();