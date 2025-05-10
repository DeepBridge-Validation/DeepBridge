/**
 * ScoreDistributionChartManager - Gerenciador de gráficos específicos para a aba de detalhes
 * Fornece funções para inicializar e atualizar visualizações de distribuição e radar 
 */
window.ScoreDistributionChartManager = {
    /**
     * Inicializar o gráfico de distribuição de pontuações
     * @param {string} containerId - ID do elemento HTML contendo o gráfico
     * @param {Array} scores - Array de pontuações para plotar
     * @param {number} baseScore - Pontuação base para linha de referência
     */
    initializeScoreDistributionChart: function(containerId, scores, baseScore) {
        console.log("Initializing score distribution chart");
        
        const chartElement = document.getElementById(containerId);
        if (!chartElement) {
            console.error("Score distribution chart element not found:", containerId);
            return;
        }
        
        // Verificar se Plotly está disponível
        if (typeof Plotly === 'undefined') {
            console.error("Plotly library not loaded. Chart cannot be displayed.");
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: red;">
                    Plotly library not loaded. Chart cannot be displayed.
                </div>`;
            return;
        }
        
        // Verificar se temos pontuações para plotar
        if (!scores || scores.length === 0) {
            console.warn("No scores available for distribution chart");
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #666;">
                    Não foram encontradas pontuações para este nível de perturbação.
                </div>`;
            return;
        }
        
        // Criar histograma
        const trace = {
            x: scores,
            type: 'histogram',
            marker: {
                color: 'rgba(66, 135, 245, 0.7)',
                line: {
                    color: 'rgba(66, 135, 245, 1)',
                    width: 1
                }
            },
            opacity: 0.7,
            name: 'Distribuição de Pontuações'
        };
        
        // Adicionar linha vertical para pontuação base
        const baseScoreLine = {
            type: 'line',
            x0: baseScore,
            x1: baseScore,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: 'rgba(255, 0, 0, 0.7)',
                width: 2,
                dash: 'dash'
            },
            name: 'Pontuação Base'
        };
        
        // Layout do gráfico
        const layout = {
            title: 'Distribuição de Pontuações',
            xaxis: {
                title: 'Pontuação',
                tickformat: '.3f',
                automargin: true
            },
            yaxis: {
                title: 'Frequência',
                automargin: true
            },
            shapes: [baseScoreLine],
            annotations: [{
                x: baseScore,
                y: 1,
                xref: 'x',
                yref: 'paper',
                text: 'Pontuação Base',
                showarrow: true,
                arrowhead: 2,
                ax: 0,
                ay: -40,
                bgcolor: 'rgba(255, 255, 255, 0.8)',
                font: {
                    color: 'red'
                }
            }],
            margin: {
                l: 50,
                r: 20,
                t: 40,
                b: 50
            },
            bargap: 0.05,
            bargroupgap: 0.2
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false
        };
        
        // Renderizar gráfico
        Plotly.newPlot(chartElement, [trace], layout, config);
        
        console.log("Score distribution chart initialized");
    },
    
    /**
     * Inicializar o gráfico radar de métricas
     * @param {string} containerId - ID do elemento HTML contendo o gráfico
     * @param {Array} models - Array de objetos de modelo com métricas
     */
    initializeMetricsRadarChart: function(containerId, models) {
        console.log("Initializing metrics radar chart");
        
        const chartElement = document.getElementById(containerId);
        if (!chartElement) {
            console.error("Metrics radar chart element not found:", containerId);
            return;
        }
        
        // Verificar se Plotly está disponível
        if (typeof Plotly === 'undefined') {
            console.error("Plotly library not loaded. Chart cannot be displayed.");
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: red;">
                    Plotly library not loaded. Chart cannot be displayed.
                </div>`;
            return;
        }
        
        // Verificar se temos modelos com métricas
        if (!models || models.length === 0) {
            console.warn("No models available for radar chart");
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #666;">
                    Não foram encontrados dados de métricas para exibir.
                </div>`;
            return;
        }
        
        // Obter métricas comuns para todos os modelos
        const commonMetrics = this.getCommonMetrics(models);
        if (commonMetrics.length === 0) {
            console.warn("No common metrics found for radar chart");
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #666;">
                    Não foram encontradas métricas comuns entre os modelos.
                </div>`;
            return;
        }
        
        // Preparar dados para o gráfico radar
        const traces = [];
        
        // Criar trace para cada modelo
        models.forEach((model, index) => {
            const values = commonMetrics.map(metric => model.metrics[metric] || 0);
            // Fechar o radar conectando de volta ao primeiro ponto
            values.push(values[0]);
            
            // Criar labels das métricas
            const labels = commonMetrics.map(metric => this.formatMetricName(metric));
            // Fechar o radar conectando de volta ao primeiro label
            labels.push(labels[0]);
            
            traces.push({
                type: 'scatterpolar',
                r: values,
                theta: labels,
                fill: 'toself',
                name: model.name,
                opacity: 0.7
            });
        });
        
        // Layout do gráfico
        const layout = {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 1.05],
                    tickformat: '.2f'
                }
            },
            showlegend: true,
            legend: {
                x: 0,
                y: 1,
                orientation: 'h'
            },
            margin: {
                l: 50,
                r: 50,
                t: 30,
                b: 30
            },
            paper_bgcolor: 'rgb(255,255,255)',
            plot_bgcolor: 'rgb(252,252,252)',
            font: {
                family: 'Arial, sans-serif'
            },
            autosize: true
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false
        };
        
        // Renderizar gráfico
        Plotly.newPlot(chartElement, traces, layout, config);
        
        console.log("Metrics radar chart initialized with", traces.length, "models");
    },
    
    /**
     * Atualizar o gráfico radar para mostrar apenas um modelo
     * @param {string} containerId - ID do elemento HTML contendo o gráfico
     * @param {Object} model - Objeto de modelo com métricas
     * @param {Array} commonMetrics - Array de nomes de métricas comuns
     */
    updateRadarChartForModel: function(containerId, model, commonMetrics) {
        console.log("Updating radar chart for model:", model.name);
        
        const chartElement = document.getElementById(containerId);
        if (!chartElement || typeof Plotly === 'undefined') {
            console.error("Chart element or Plotly not available");
            return;
        }
        
        if (!commonMetrics || commonMetrics.length === 0) {
            commonMetrics = this.getCommonMetrics([model]);
            if (commonMetrics.length === 0) {
                console.error("No metrics available for model");
                return;
            }
        }
        
        // Preparar dados para o gráfico
        const values = commonMetrics.map(metric => model.metrics[metric] || 0);
        values.push(values[0]); // Fechar o radar
        
        const labels = commonMetrics.map(metric => this.formatMetricName(metric));
        labels.push(labels[0]); // Fechar o radar
        
        const trace = {
            type: 'scatterpolar',
            r: values,
            theta: labels,
            fill: 'toself',
            name: model.name,
            opacity: 0.7
        };
        
        // Layout do gráfico
        const layout = {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 1.05],
                    tickformat: '.2f'
                }
            },
            showlegend: true,
            legend: {
                x: 0,
                y: 1,
                orientation: 'h'
            },
            margin: {
                l: 50,
                r: 50,
                t: 30,
                b: 30
            },
            paper_bgcolor: 'rgb(255,255,255)',
            plot_bgcolor: 'rgb(252,252,252)',
            font: {
                family: 'Arial, sans-serif'
            },
            autosize: true
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false
        };
        
        // Renderizar gráfico
        Plotly.newPlot(chartElement, [trace], layout, config);
        
        console.log("Radar chart updated for model:", model.name);
    },
    
    /**
     * Obter métricas comuns para todos os modelos
     * @param {Array} models - Array de objetos de modelo
     * @returns {Array} Array de nomes de métricas comuns
     */
    getCommonMetrics: function(models) {
        if (!models || models.length === 0) {
            return [];
        }
        
        // Conjunto de métricas padrão para verificar
        const standardMetrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];
        
        // Verificar quais métricas estão disponíveis em pelo menos um modelo
        if (models.length === 1) {
            // Se temos apenas um modelo, retornar todas as métricas disponíveis
            const metrics = models[0].metrics || {};
            return standardMetrics.filter(metric => metrics[metric] !== undefined);
        }
        
        // Verificar quais métricas estão disponíveis em todos os modelos
        const availableMetrics = [];
        
        for (const metric of standardMetrics) {
            let allModelsHaveMetric = true;
            
            for (const model of models) {
                const metrics = model.metrics || {};
                if (metrics[metric] === undefined) {
                    allModelsHaveMetric = false;
                    break;
                }
            }
            
            if (allModelsHaveMetric) {
                availableMetrics.push(metric);
            }
        }
        
        return availableMetrics;
    },
    
    /**
     * Formatar nome de métrica para exibição
     * @param {string} metric - Nome da métrica
     * @returns {string} Nome formatado
     */
    formatMetricName: function(metric) {
        const metricMap = {
            'accuracy': 'Acurácia',
            'precision': 'Precisão',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'roc_auc': 'ROC AUC'
        };
        
        return metricMap[metric] || metric;
    }
};