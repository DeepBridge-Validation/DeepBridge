// Chart Manager for Overview Section
const ChartManager = {
    /**
     * Initialize perturbation chart
     * @param {string} elementId - Chart container ID
     */
    
    initializePerturbationChart: function(elementId) {
        console.log("Initializing perturbation chart");
        const chartElement = document.getElementById(elementId);
        if (!chartElement) {
            console.error("Chart element not found:", elementId);
            return;
        }
        
        try {
            // Extract data for chart
            const chartData = this.extractPerturbationChartData();
            
            if (!chartData || chartData.levels.length === 0) {
                this.showNoDataMessage(chartElement, "No perturbation data available");
                return;
            }
            
            // Create a horizontal line for the base score
            const baseScores = Array(chartData.levels.length).fill(chartData.baseScore);
            
            // Prepare plot data - first trace is the base score
            const plotData = [
                {
                    x: chartData.levels,
                    y: baseScores,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Base Score',
                    line: {
                        dash: 'dash',
                        width: 2,
                        color: 'rgb(136, 132, 216)'
                    }
                }
            ];
            
            // Add all features score trace if available
            if (chartData.perturbedScores && chartData.perturbedScores.length > 0) {
                plotData.push({
                    x: chartData.levels,
                    y: chartData.perturbedScores,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'All Features Score',
                    line: {
                        width: 3,
                        color: 'rgb(255, 87, 51)'
                    },
                    marker: {
                        size: 8,
                        color: 'rgb(255, 87, 51)'
                    }
                });
            }
            
            // Add worst scores trace if available
            if (chartData.worstScores && chartData.worstScores.length > 0) {
                plotData.push({
                    x: chartData.levels,
                    y: chartData.worstScores,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Worst Score',
                    line: {
                        width: 2,
                        color: 'rgb(199, 0, 57)'
                    },
                    marker: {
                        size: 6,
                        color: 'rgb(199, 0, 57)'
                    }
                });
            }
            
            // Add feature subset scores trace if available
            if (chartData.featureSubsetScores && chartData.featureSubsetScores.length > 0) {
                plotData.push({
                    x: chartData.levels,
                    y: chartData.featureSubsetScores,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Feature Subset Score',
                    line: {
                        width: 2,
                        color: 'rgb(40, 180, 99)'
                    },
                    marker: {
                        size: 6,
                        color: 'rgb(40, 180, 99)'
                    }
                });
            }
            
            // Layout for the chart
            const layout = {
                title: `Model Robustness: Performance Under Perturbation (${chartData.metricName})`,
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
                    orientation: 'h',
                    y: -0.2
                },
                hovermode: 'closest',
                margin: {
                    l: 50,
                    r: 20,
                    t: 60,
                    b: 100
                }
            };
            
            // Create the plot
            Plotly.newPlot(chartElement, plotData, layout, {responsive: true});
            
        } catch (error) {
            console.error("Error creating perturbation chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Extract data for perturbation chart from report data
     */
    extractPerturbationChartData: function() {
        // Primeiro, verificar se temos dados pré-calculados
        if (window.reportData && window.reportData.perturbation_chart_data) {
            console.log("Usando dados pré-calculados do perturbation_chart_data");
            return {
                levels: window.reportData.perturbation_chart_data.levels,
                perturbedScores: window.reportData.perturbation_chart_data.scores,
                worstScores: window.reportData.perturbation_chart_data.worstScores,
                featureSubsetScores: window.reportData.perturbation_chart_data.featureSubsetScores || [],
                baseScore: window.reportData.perturbation_chart_data.baseScore,
                metricName: window.reportData.perturbation_chart_data.metric
            };
        }
        
        console.log("Nenhum dado pré-calculado encontrado, extraindo dados brutos");
        
        // Código original de extração como fallback
        let perturbationLevels = [];
        let perturbedScores = [];
        let worstScores = [];
        let featureSubsetScores = [];
        let baseScore = null;
        let metricName = 'Score';
        
        // Extract data from report data
        if (window.reportData) {
            // Get base score
            if (window.reportConfig && window.reportConfig.baseScore !== undefined) {
                baseScore = window.reportConfig.baseScore;
            } else if (window.reportData.base_score !== undefined) {
                baseScore = window.reportData.base_score;
            }
            
            // Get metric name
            if (window.reportConfig && window.reportConfig.metric) {
                metricName = window.reportConfig.metric;
            } else if (window.reportData.metric) {
                metricName = window.reportData.metric;
            }
            
            // Extract data from raw perturbation results
            if (window.reportData.raw && window.reportData.raw.by_level) {
                const rawData = window.reportData.raw.by_level;
                
                // Sort levels numerically
                perturbationLevels = Object.keys(rawData)
                    .sort((a, b) => parseFloat(a) - parseFloat(b))
                    .map(parseFloat);
                
                // Get perturbed scores (all features)
                perturbedScores = perturbationLevels.map(level => {
                    const levelStr = level.toString();
                    if (rawData[levelStr] && 
                        rawData[levelStr].overall_result && 
                        rawData[levelStr].overall_result.all_features) {
                        return rawData[levelStr].overall_result.all_features.mean_score;
                    }
                    return null;
                });
                
                // Get worst scores
                worstScores = perturbationLevels.map(level => {
                    const levelStr = level.toString();
                    if (rawData[levelStr] && 
                        rawData[levelStr].overall_result && 
                        rawData[levelStr].overall_result.all_features) {
                        return rawData[levelStr].overall_result.all_features.worst_score;
                    }
                    return null;
                });
                
                // Get feature subset scores
                featureSubsetScores = perturbationLevels.map(level => {
                    const levelStr = level.toString();
                    if (rawData[levelStr] && 
                        rawData[levelStr].overall_result && 
                        rawData[levelStr].overall_result.feature_subset) {
                        return rawData[levelStr].overall_result.feature_subset.mean_score;
                    }
                    return null;
                });
            }
        }
        
        return {
            levels: perturbationLevels,
            perturbedScores: perturbedScores,
            worstScores: worstScores,
            featureSubsetScores: featureSubsetScores,
            baseScore: baseScore,
            metricName: metricName
        };
    },
    
    /**
     * Initialize worst score chart
     * @param {string} elementId - Chart container ID
     */
    initializeWorstScoreChart: function(elementId) {
        const chartElement = document.getElementById(elementId);
        if (!chartElement) return;
        
        try {
            // Extract data
            const chartData = this.extractPerturbationChartData();
            
            if (!chartData || chartData.levels.length === 0 || !chartData.worstScores || chartData.worstScores.length === 0) {
                this.showNoDataMessage(chartElement, "No worst score data available");
                return;
            }
            
            // Create trace for worst scores
            const worstTrace = {
                x: chartData.levels,
                y: chartData.worstScores,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Worst Score',
                line: {
                    width: 3,
                    color: 'rgb(199, 0, 57)'
                },
                marker: {
                    size: 8,
                    color: 'rgb(199, 0, 57)'
                }
            };
            
            const data = [worstTrace];
            
            // Add base score trace if available
            if (chartData.baseScore !== null) {
                const baseScoreTrace = {
                    x: chartData.levels,
                    y: Array(chartData.levels.length).fill(chartData.baseScore),
                    type: 'scatter',
                    mode: 'lines',
                    line: {
                        dash: 'dash',
                        width: 2,
                        color: 'rgb(136, 132, 216)'
                    },
                    name: 'Base Score'
                };
                data.push(baseScoreTrace);
            }
            
            // Layout
            const layout = {
                title: `Worst-Case Performance Under Perturbation (${chartData.metricName})`,
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
                    l: 50,
                    r: 20,
                    t: 60,
                    b: 100
                }
            };
            
            // Create plot
            Plotly.newPlot(chartElement, data, layout, {responsive: true});
            
        } catch (error) {
            console.error("Error creating worst score chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Initialize mean score chart
     * @param {string} elementId - Chart container ID
     */
    initializeMeanScoreChart: function(elementId) {
        const chartElement = document.getElementById(elementId);
        if (!chartElement) return;
        
        try {
            // Extract data
            const chartData = this.extractPerturbationChartData();
            
            if (!chartData || chartData.levels.length === 0 || !chartData.perturbedScores || chartData.perturbedScores.length === 0) {
                this.showNoDataMessage(chartElement, "No mean score data available");
                return;
            }
            
            // Create trace for mean scores
            const meanTrace = {
                x: chartData.levels,
                y: chartData.perturbedScores,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Mean Score',
                line: {
                    width: 3,
                    color: 'rgb(255, 87, 51)'
                },
                marker: {
                    size: 8,
                    color: 'rgb(255, 87, 51)'
                }
            };
            
            const data = [meanTrace];
            
            // Add base score trace if available
            if (chartData.baseScore !== null) {
                const baseScoreTrace = {
                    x: chartData.levels,
                    y: Array(chartData.levels.length).fill(chartData.baseScore),
                    type: 'scatter',
                    mode: 'lines',
                    line: {
                        dash: 'dash',
                        width: 2,
                        color: 'rgb(136, 132, 216)'
                    },
                    name: 'Base Score'
                };
                data.push(baseScoreTrace);
            }
            
            // Layout
            const layout = {
                title: `Mean Performance Under Perturbation (${chartData.metricName})`,
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
                    orientation: 'h',
                    y: -0.2
                },
                hovermode: 'closest',
                margin: {
                    l: 50,
                    r: 20,
                    t: 60,
                    b: 100
                }
            };
            
            // Create plot
            Plotly.newPlot(chartElement, data, layout, {responsive: true});
            
        } catch (error) {
            console.error("Error creating mean score chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Initialize model comparison chart
     * @param {string} elementId - Chart container ID
     */
    initializeModelComparisonChart: function(elementId) {
        const chartElement = document.getElementById(elementId);
        if (!chartElement) return;
        
        try {
            // Check if we have alternative models data
            if (!window.reportData || !window.reportConfig || !window.reportConfig.hasAlternativeModels) {
                this.showNoDataMessage(chartElement, "No model comparison data available");
                return;
            }
            
            // Extract data for chart
            const chartData = this.extractModelComparisonData();
            
            if (!chartData || chartData.models.length <= 1) {
                this.showNoDataMessage(chartElement, "Insufficient model data for comparison");
                return;
            }
            
            // Create base score bars
            const baseScoreTrace = {
                x: chartData.models,
                y: chartData.baseScores,
                type: 'bar',
                name: 'Base Score',
                marker: {
                    color: 'rgb(41, 128, 185)'
                }
            };
            
            // Create robustness score bars
            const robustnessScoreTrace = {
                x: chartData.models,
                y: chartData.robustnessScores,
                type: 'bar',
                name: 'Robustness Score',
                marker: {
                    color: 'rgb(46, 204, 113)'
                }
            };
            
            // Create the plot data
            const plotData = [baseScoreTrace, robustnessScoreTrace];
            
            // Layout
            const layout = {
                title: 'Model Comparison Overview',
                barmode: 'group',
                xaxis: {
                    title: 'Models',
                    tickangle: -45
                },
                yaxis: {
                    title: 'Score',
                    range: [0, 1.1]
                },
                legend: {
                    orientation: 'h',
                    y: -0.2
                },
                margin: {
                    l: 50,
                    r: 20,
                    t: 60,
                    b: 150
                }
            };
            
            // Create the plot
            Plotly.newPlot(chartElement, plotData, layout, {responsive: true});
            
        } catch (error) {
            console.error("Error creating model comparison chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Extract data for model comparison
     */
    extractModelComparisonData: function() {
        const models = [];
        const baseScores = [];
        const robustnessScores = [];
        
        if (!window.reportData) return null;
        
        // Add primary model
        let primaryModelName = 'Primary Model';
        if (window.reportData.model_name) {
            primaryModelName = window.reportData.model_name;
        } else if (window.reportConfig && window.reportConfig.modelName) {
            primaryModelName = window.reportConfig.modelName;
        }
        
        let primaryBaseScore = 0;
        if (window.reportData.base_score !== undefined) {
            primaryBaseScore = window.reportData.base_score;
        } else if (window.reportConfig && window.reportConfig.baseScore !== undefined) {
            primaryBaseScore = window.reportConfig.baseScore;
            console.log("Gráfico - Usando baseScore do reportConfig:", primaryBaseScore);
        }
        
        // CORREÇÃO: Garantir que estamos usando o valor correto do robustness_score
        let primaryRobustnessScore = 0;
        if (typeof window.reportData.robustness_score === 'number') {
            primaryRobustnessScore = window.reportData.robustness_score;
            console.log("Gráfico - Usando robustness_score do modelo primário:", primaryRobustnessScore);
        } else if (typeof window.reportData.score === 'number') {
            // Fallback to score if robustness_score is not available
            primaryRobustnessScore = window.reportData.score;
            console.log("Gráfico - Usando score do modelo primário como fallback:", primaryRobustnessScore);
        } else if (window.reportConfig && typeof window.reportConfig.robustnessScore === 'number') {
            primaryRobustnessScore = window.reportConfig.robustnessScore;
            console.log("Gráfico - Usando robustnessScore do reportConfig:", primaryRobustnessScore);
        }
        
        models.push(primaryModelName);
        baseScores.push(primaryBaseScore);
        robustnessScores.push(primaryRobustnessScore);
        
        // Add alternative models
        if (window.reportData.alternative_models) {
            Object.entries(window.reportData.alternative_models).forEach(([name, data]) => {
                models.push(name);
                baseScores.push(data.base_score || 0);
                
                // CORREÇÃO: Garantir que usamos os valores corretos para modelos alternativos
                let altScore = 0;
                if (typeof data.robustness_score === 'number') {
                    altScore = data.robustness_score;
                    console.log(`Gráfico - Modelo alternativo ${name} robustness_score:`, altScore);
                } else if (typeof data.score === 'number') {
                    // Fallback to score if robustness_score is not available
                    altScore = data.score;
                    console.log(`Gráfico - Modelo alternativo ${name} score (fallback):`, altScore);
                }
                robustnessScores.push(altScore);
            });
        }
        
        return {
            models,
            baseScores,
            robustnessScores
        };
    },
    
    /**
     * Initialize model level details chart
     * @param {string} elementId - Chart container ID
     */
    initializeModelLevelDetailsChart: function(elementId) {
        const chartElement = document.getElementById(elementId);
        if (!chartElement) return;
        
        try {
            // Check if we have alternative models data
            if (!window.reportData || !window.reportConfig || !window.reportConfig.hasAlternativeModels) {
                this.showNoDataMessage(chartElement, "No model comparison data available");
                return;
            }
            
            // Extract data for model performance across perturbation levels
            const chartData = this.extractModelLevelDetailsData();
            
            if (!chartData || chartData.levels.length === 0 || Object.keys(chartData.modelScores).length <= 1) {
                this.showNoDataMessage(chartElement, "Insufficient data for model comparison by level");
                return;
            }
            
            // Create a trace for each model
            const plotData = [];
            const colors = ['rgb(255, 87, 51)', 'rgb(41, 128, 185)', 'rgb(142, 68, 173)', 'rgb(39, 174, 96)', 'rgb(243, 156, 18)'];
            let colorIndex = 0;
            
            // Add primary model first
            if (chartData.modelScores['primary']) {
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
            
            // Add alternative models
            for (const modelId in chartData.modelScores) {
                if (modelId === 'primary') continue; // Skip primary model as it's already added
                
                plotData.push({
                    x: chartData.levels,
                    y: chartData.modelScores[modelId],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: chartData.modelNames[modelId] || modelId,
                    line: {
                        width: 2,
                        color: colors[colorIndex % colors.length]
                    },
                    marker: {
                        size: 6,
                        color: colors[colorIndex % colors.length]
                    }
                });
                colorIndex++;
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
                    l: 50,
                    r: 20,
                    t: 60,
                    b: 100
                }
            };
            
            // Create plot
            Plotly.newPlot(chartElement, plotData, layout, {responsive: true});
            
        } catch (error) {
            console.error("Error creating model level details chart:", error);
            this.showErrorMessage(chartElement, error.message);
        }
    },
    
    /**
     * Extract data for model level details
     */
    extractModelLevelDetailsData: function() {
        let levels = [];
        const modelScores = {};
        const modelNames = {};
        let metricName = 'Score';
        
        if (!window.reportData) return null;
        
        // Get metric name
        if (window.reportConfig && window.reportConfig.metric) {
            metricName = window.reportConfig.metric;
        } else if (window.reportData.metric) {
            metricName = window.reportData.metric;
        }
        
        // Usar níveis diretamente dos dados pré-processados, se disponíveis
        if (window.reportData.perturbation_chart_data && 
            window.reportData.perturbation_chart_data.levels && 
            window.reportData.perturbation_chart_data.levels.length > 0) {
            
            levels = window.reportData.perturbation_chart_data.levels.map(l => parseFloat(l));
            console.log("Usando níveis dos dados pré-processados para gráficos:", levels);
            
            // Verificar dados dos modelos alternativos
            if (window.reportData.perturbation_chart_data.alternativeModels) {
                const altModels = window.reportData.perturbation_chart_data.alternativeModels;
                console.log("Modelos alternativos disponíveis:", Object.keys(altModels));
                
                // Verificar que cada modelo alternativo tem o mesmo número de scores
                Object.entries(altModels).forEach(([name, data]) => {
                    if (data.scores) {
                        console.log(`Modelo ${name} tem ${data.scores.length} scores para ${levels.length} níveis`);
                    }
                });
            }
        } else {
            // Fallback: coletar níveis dos dados raw
            const allLevels = new Set();
                
            // Coletar níveis do modelo principal
            if (window.reportData.raw && window.reportData.raw.by_level) {
                Object.keys(window.reportData.raw.by_level)
                    .forEach(level => allLevels.add(parseFloat(level)));
            }
            
            // Coletar níveis dos modelos alternativos
            if (window.reportData.alternative_models) {
                Object.values(window.reportData.alternative_models).forEach(model => {
                    if (model.raw && model.raw.by_level) {
                        Object.keys(model.raw.by_level)
                            .forEach(level => allLevels.add(parseFloat(level)));
                    }
                });
            }
            
            // Transformar Set em array e ordenar
            levels = Array.from(allLevels).sort((a, b) => a - b);
            console.log("Usando níveis coletados manualmente:", levels);
        }
        console.log("Níveis coletados para comparação de modelos:", levels);
        
        // Extrair dados do modelo principal
        if (window.reportData.raw && window.reportData.raw.by_level) {
            const rawData = window.reportData.raw.by_level;
            
            // Verificar primeiro se podemos usar dados pré-processados para o modelo primário
            let primaryScores;
            if (window.reportData.perturbation_chart_data && 
                window.reportData.perturbation_chart_data.scores &&
                window.reportData.perturbation_chart_data.scores.length === levels.length) {
                
                console.log("Usando scores pré-processados para o modelo primário");
                primaryScores = window.reportData.perturbation_chart_data.scores;
            } else {
                // Caso contrário, extrair dos dados raw
                primaryScores = levels.map(level => {
                    const levelStr = level.toString();
                    if (rawData[levelStr] && 
                        rawData[levelStr].overall_result && 
                        rawData[levelStr].overall_result.all_features) {
                        return rawData[levelStr].overall_result.all_features.mean_score;
                    }
                    return null;
                });
                
                // Adicionar log para debugging dos valores nulos
                if (primaryScores.includes(null)) {
                    console.log("Modelo primário tem valores null:", primaryScores);
                    console.log("Níveis correspondentes:", levels);
                    console.log("Dados raw do modelo primário:", JSON.stringify(rawData, null, 2));
                }
            }
            
            modelScores['primary'] = primaryScores;
            modelNames['primary'] = window.reportData.model_name || 'Primary Model';
        }
        
        // Adicionar modelos alternativos
        if (window.reportData.alternative_models) {
            Object.entries(window.reportData.alternative_models).forEach(([name, data]) => {
                if (data.raw && data.raw.by_level) {
                    const rawData = data.raw.by_level;
                    
                    // Verificar primeiro se podemos usar dados pré-processados
                    let scores;
                    if (window.reportData.perturbation_chart_data && 
                        window.reportData.perturbation_chart_data.alternativeModels && 
                        window.reportData.perturbation_chart_data.alternativeModels[name] &&
                        window.reportData.perturbation_chart_data.alternativeModels[name].scores &&
                        window.reportData.perturbation_chart_data.alternativeModels[name].scores.length === levels.length) {
                        
                        console.log(`Usando scores pré-processados para o modelo alternativo ${name}`);
                        scores = window.reportData.perturbation_chart_data.alternativeModels[name].scores;
                    } else {
                        // Caso contrário, extrair dos dados raw
                        scores = levels.map(level => {
                            const levelStr = level.toString();
                            if (rawData[levelStr] && 
                                rawData[levelStr].overall_result && 
                                rawData[levelStr].overall_result.all_features) {
                                return rawData[levelStr].overall_result.all_features.mean_score;
                            }
                            return null;
                        });
                        
                        // Adicionar log para debugging dos valores nulos
                        if (scores.includes(null)) {
                            console.log(`Modelo ${name} tem valores null:`, scores);
                            console.log(`Níveis correspondentes para ${name}:`, levels);
                            console.log(`Dados raw do modelo ${name}:`, JSON.stringify(rawData, null, 2));
                        }
                    }
                    
                    modelScores[name] = scores;
                    modelNames[name] = name;
                }
            });
        }
        
        return {
            levels,
            modelScores,
            modelNames,
            metricName
        };
    },
    
    /**
     * Show no data message in chart container
     * @param {HTMLElement} element - Chart container element
     * @param {string} message - Message to display
     */
    showNoDataMessage: function(element, message) {
        element.innerHTML = `
            <div class="data-unavailable">
                <div class="data-message">
                    <span class="message-icon">📊</span>
                    <h3>No Data Available</h3>
                    <p>${message}</p>
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
            <div style='padding: 20px; color: red;'>
                Error creating chart: ${errorMessage}
            </div>`;
    },
    
    /**
     * Format score value to a readable string
     * @param {number} score - The score value to format
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted score
     */
    formatScore: function(score, decimals = 4) {
        if (score === null || score === undefined) return 'N/A';
        return score.toFixed(decimals);
    }
};