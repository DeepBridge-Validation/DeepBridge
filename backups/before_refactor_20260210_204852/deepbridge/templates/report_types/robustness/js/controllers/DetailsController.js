/**
 * DetailsController - Controlador para a aba de detalhes no relatório de robustez
 * Responsável por gerenciar a visualização de detalhes de perturbações e métricas do modelo
 */
window.DetailsController = {
    // Dados carregados
    modelData: {},
    configData: {},
    perturbationData: {},
    
    /**
     * Inicializar o controlador
     */
    init: function() {
        console.log("DetailsController initialized");
        
        // Carregar dados
        this.loadData();
        
        // Inicializar tabela de comparação de modelos
        this.initModelsTable();
        
        // Inicializar o container de resultados de perturbação
        this.initPerturbationResults();
        
        // Inicializar o gráfico radar de métricas
        this.initMetricsRadarChart();
        
        // Configurar event listeners
        this.setupEventListeners();
        
        console.log("Details page initialization complete");
    },
    
    /**
     * Carregar dados necessários
     */
    loadData: function() {
        console.log("Loading details data");
        
        // Carregar dados de modelos
        this.loadModelData();
        
        // Carregar dados de perturbação
        this.loadPerturbationData();
    },
    
    /**
     * Carregar dados de modelos
     */
    loadModelData: function() {
        // Tentar initial_results primeiro (de transformer)
        if (window.reportData && window.reportData.initial_results) {
            console.log("Found initial_results data for details");
            if (window.reportData.initial_results.models) {
                this.modelData = window.reportData.initial_results.models;
                this.configData = window.reportData.initial_results.config || {};
            }
        } 
        // Tentar chart_data.initial_results
        else if (window.chartData && window.chartData.initial_results) {
            if (window.chartData.initial_results.models) {
                this.modelData = window.chartData.initial_results.models;
                this.configData = window.chartData.initial_results.config || {};
            }
        }
        // Tentar radar_chart_data
        else if (window.chartData && window.chartData.radar_chart_data) {
            if (window.chartData.radar_chart_data.models) {
                this.modelData = window.chartData.radar_chart_data.models;
            }
        }
        // Tentar generic model data
        else if (window.reportData) {
            this.modelData = window.reportData.models || {};
            this.configData = window.reportData.config || {};
        }
        
        // Verificar se temos dados de modelo
        if (Object.keys(this.modelData).length === 0) {
            console.warn("No model data found for details view");
            this.showDataError("Não foi possível carregar dados de modelos para visualização detalhada.");
        } else {
            console.log("Loaded details data for", Object.keys(this.modelData).length, "models");
            
            // Atualizar informações do dataset
            this.updateDatasetInfo();
        }
    },
    
    /**
     * Carregar dados de perturbação
     */
    loadPerturbationData: function() {
        // Verificar fontes diferentes para obter dados de perturbação
        if (window.reportData && window.reportData.raw) {
            this.perturbationData = window.reportData.raw;
            console.log("Using perturbation data from reportData.raw");
        } else if (window.chartData && window.chartData.raw) {
            this.perturbationData = window.chartData.raw;
            console.log("Using perturbation data from chartData.raw");
        } else if (window.reportData && window.reportData.perturbation_chart_data) {
            this.perturbationData = window.reportData.perturbation_chart_data;
            console.log("Using perturbation data from perturbation_chart_data");
        } else if (window.chartData && window.chartData.perturbation_chart_data) {
            this.perturbationData = window.chartData.perturbation_chart_data;
            console.log("Using perturbation data from chartData.perturbation_chart_data");
        }

        // Verificar se temos dados de perturbação
        if (!this.perturbationData || Object.keys(this.perturbationData).length === 0) {
            console.warn("No perturbation data found");
            this.showPerturbationDataError("Não foram encontrados dados de perturbação para exibir.");
        }
    },
    
    /**
     * Atualizar informações do dataset na UI
     */
    updateDatasetInfo: function() {
        // Elementos de informação do dataset
        const samplesElement = document.getElementById('n-samples');
        const featuresElement = document.getElementById('n-features');
        const testSizeElement = document.getElementById('test-size');
        
        // Elementos de informação do teste
        const testsListElement = document.getElementById('tests-list');
        const verboseElement = document.getElementById('verbose-status');
        
        // Verificar se temos info do dataset
        if (this.configData && this.configData.dataset_info) {
            const datasetInfo = this.configData.dataset_info;
            
            // Atualizar info do dataset
            if (samplesElement) {
                samplesElement.textContent = datasetInfo.n_samples || 'N/A';
            }
            
            if (featuresElement) {
                featuresElement.textContent = datasetInfo.n_features || 'N/A';
            }
            
            if (testSizeElement && datasetInfo.test_size !== undefined) {
                testSizeElement.textContent = (datasetInfo.test_size * 100) + '%';
            }
        } else if (window.reportData) {
            // Tentar obter informações diretamente do reportData
            if (samplesElement) {
                samplesElement.textContent = window.reportData.test_sample_count || 'N/A';
            }
            
            if (featuresElement) {
                const features = window.reportData.features || [];
                featuresElement.textContent = features.length || 'N/A';
            }
            
            if (testSizeElement) {
                testSizeElement.textContent = window.reportData.test_size || 'N/A';
            }
        } else {
            // Definir espaços reservados
            if (samplesElement) samplesElement.textContent = 'N/A';
            if (featuresElement) featuresElement.textContent = 'N/A';
            if (testSizeElement) testSizeElement.textContent = 'N/A';
        }
        
        // Verificar se temos info de configuração de teste
        if (this.configData && this.configData.tests) {
            // Atualizar info de teste
            if (testsListElement) {
                testsListElement.textContent = this.configData.tests.join(', ');
            }
            
            if (verboseElement) {
                verboseElement.textContent = this.configData.verbose ? 'Sim' : 'Não';
            }
        } else if (window.reportData && window.reportData.tests) {
            // Tentar obter info diretamente do reportData
            if (testsListElement) {
                testsListElement.textContent = Array.isArray(window.reportData.tests) 
                    ? window.reportData.tests.join(', ') 
                    : 'Robustez';
            }
            
            if (verboseElement) {
                verboseElement.textContent = window.reportData.verbose ? 'Sim' : 'Não';
            }
        } else {
            // Definir espaços reservados
            if (testsListElement) testsListElement.textContent = 'Robustez';
            if (verboseElement) verboseElement.textContent = 'N/A';
        }
    },
    
    /**
     * Configurar event listeners
     */
    setupEventListeners: function() {
        // Model selector change event
        const selector = document.getElementById('model-selector');
        if (selector) {
            selector.addEventListener('change', (e) => {
                const modelId = e.target.value;
                this.handleModelSelection(modelId);
            });
        }

        // Table header sort event
        const headers = document.querySelectorAll('#models-table th.sortable');
        if (headers.length > 0) {
            headers.forEach(header => {
                header.addEventListener('click', (e) => {
                    const sortBy = e.currentTarget.getAttribute('data-sort');
                    this.sortModelsTable(sortBy);
                });
            });
        }

        // Setup subtabs navigation
        const detailsTabs = document.querySelectorAll('.details-tab');
        if (detailsTabs.length > 0) {
            detailsTabs.forEach(tab => {
                tab.addEventListener('click', (e) => {
                    const tabId = e.currentTarget.getAttribute('data-subtab');
                    this.switchSubtab(tabId);
                });
            });
        }

        // Apply the same styling and behavior as chart-selector buttons
        document.querySelectorAll('.details-tabs').forEach(tabContainer => {
            // Add matching classes for consistent styling
            tabContainer.classList.add('chart-selector');

            // Convert li elements to button elements with the right classes
            const tabs = tabContainer.querySelectorAll('.details-tab');
            tabs.forEach(tab => {
                tab.classList.add('chart-selector-option');
            });
        });
    },

    /**
     * Switch between subtabs in the details section
     * @param {string} tabId - ID of the subtab to switch to
     */
    switchSubtab: function(tabId) {
        console.log("Switching to subtab:", tabId);

        // Update tab buttons
        document.querySelectorAll('.details-tab').forEach(tab => {
            if (tab.getAttribute('data-subtab') === tabId) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });

        // Update tab contents
        document.querySelectorAll('.subtab-content').forEach(content => {
            if (content.id === tabId) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });

        // Adjust charts after tab switch for proper rendering
        window.setTimeout(() => {
            window.dispatchEvent(new Event('resize'));

            // Initialize specific charts based on the active subtab
            if (tabId === 'model-metrics') {
                this.initMetricsRadarChart();
            } else if (tabId === 'perturbation-analysis') {
                // If there's a level selector, trigger a change to refresh the charts
                const levelSelector = document.getElementById('perturbation-level-selector');
                if (levelSelector) {
                    const event = new Event('change');
                    levelSelector.dispatchEvent(event);
                }
            }
        }, 50);
    },
    
    /**
     * Inicializar tabela de comparação de modelos
     */
    initModelsTable: function() {
        const tableBody = document.getElementById('models-table-body');
        if (!tableBody) {
            console.error("Models table body element not found in details view");
            return;
        }
        
        // Verificar se temos dados de modelo
        if (Object.keys(this.modelData).length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="empty-table-message">
                        Não foram encontrados dados de modelos para exibir.
                    </td>
                </tr>`;
            return;
        }
        
        // Converter dados de modelo para array para ordenação
        const modelsArray = Object.entries(this.modelData).map(([key, model]) => ({
            id: key,
            name: model.name || key,
            type: model.type || 'Desconhecido',
            metrics: model.metrics || {}
        }));
        
        // Ordenar por nome do modelo por padrão
        modelsArray.sort((a, b) => a.name.localeCompare(b.name));
        
        // Gerar HTML das linhas da tabela
        const rows = modelsArray.map(model => {
            const metrics = model.metrics || {};
            
            return `
                <tr data-model-id="${model.id}">
                    <td>${model.name}</td>
                    <td>${model.type}</td>
                    <td class="numeric">${this.formatMetric(metrics.accuracy)}</td>
                    <td class="numeric">${this.formatMetric(metrics.roc_auc)}</td>
                    <td class="numeric">${this.formatMetric(metrics.f1)}</td>
                    <td class="numeric">${this.formatMetric(metrics.precision)}</td>
                    <td class="numeric">${this.formatMetric(metrics.recall)}</td>
                </tr>`;
        }).join('');
        
        // Atualizar corpo da tabela
        tableBody.innerHTML = rows;
        
        console.log("Models table initialized in details view with", modelsArray.length, "models");
    },
    
    /**
     * Inicializar container de resultados de perturbação
     */
    initPerturbationResults: function() {
        console.log("Initializing perturbation results container");
        
        const container = document.getElementById('perturbation-results-container');
        if (!container) {
            console.error("Perturbation results container not found");
            return;
        }
        
        // Verificar se temos dados de perturbação válidos
        if (!this.perturbationData || !this.perturbationData.by_level) {
            this.showPerturbationDataError("Não foram encontrados dados de perturbação detalhados.");
            return;
        }
        
        // Obter níveis de perturbação
        const levels = Object.keys(this.perturbationData.by_level).sort((a, b) => parseFloat(a) - parseFloat(b));
        if (levels.length === 0) {
            this.showPerturbationDataError("Não foram encontrados níveis de perturbação.");
            return;
        }
        
        // Construir interface de resultados de perturbação
        let html = `
            <div class="perturbation-results">
                <div class="perturbation-controls">
                    <div class="form-group">
                        <label for="perturbation-level-selector">Perturbation Level:</label>
                        <select id="perturbation-level-selector" class="form-control">
                            ${levels.map(level => `<option value="${level}">${parseFloat(level).toFixed(2)}</option>`).join('')}
                        </select>
                    </div>
                </div>
                
                <div class="perturbation-summary">
                    <h4>Perturbation Summary <span id="current-level-display"></span></h4>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value" id="mean-score-value">-</div>
                            <div class="metric-label">Average Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="worst-score-value">-</div>
                            <div class="metric-label">Worst Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="score-drop-value">-</div>
                            <div class="metric-label">Performance Drop</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="iterations-value">-</div>
                            <div class="metric-label">Iterations</div>
                        </div>
                    </div>
                </div>
                
                <div class="perturbation-charts">
                    <div class="chart-container">
                        <h4>Score Distribution</h4>
                        <div id="score-distribution-chart" class="chart-plot" style="height: 300px;"></div>
                    </div>
                </div>
            </div>
        `;
        
        // Atualizar container
        container.innerHTML = html;
        
        // Configurar event listener para o seletor de nível
        const levelSelector = document.getElementById('perturbation-level-selector');
        if (levelSelector) {
            levelSelector.addEventListener('change', (e) => {
                const level = e.target.value;
                this.updatePerturbationDetails(level);
            });
            
            // Inicializar com o primeiro nível
            this.updatePerturbationDetails(levels[0]);
        }
        
        console.log("Perturbation results container initialized");
    },
    
    /**
     * Atualizar detalhes de perturbação para um nível específico
     * @param {string} level - Nível de perturbação selecionado
     */
    updatePerturbationDetails: function(level) {
        console.log("Updating perturbation details for level:", level);
        
        // Atualizar display de nível atual
        const levelDisplay = document.getElementById('current-level-display');
        if (levelDisplay) {
            levelDisplay.textContent = `(Nível ${parseFloat(level).toFixed(2)})`;
        }
        
        // Obter dados para este nível
        const levelData = this.perturbationData.by_level[level];
        if (!levelData || !levelData.overall_result) {
            console.error("No data found for perturbation level:", level);
            return;
        }
        
        // Obter resultados globais
        const overallResult = levelData.overall_result;
        const allFeaturesResult = overallResult.all_features || {};
        
        // Atualizar métricas de sumário
        this.updateSummaryMetrics(allFeaturesResult, level);
        
        // Atualizar tabela de características
        this.updateFeatureTable(levelData, level);
        
        // Atualizar gráfico de distribuição de pontuações
        this.updateScoreDistributionChart(levelData, level);
    },
    
    /**
     * Atualizar métricas de sumário para um nível
     * @param {Object} result - Resultado geral para o nível
     * @param {string} level - Nível de perturbação
     */
    updateSummaryMetrics: function(result, level) {
        // Obter elementos de métricas
        const meanScoreElement = document.getElementById('mean-score-value');
        const worstScoreElement = document.getElementById('worst-score-value');
        const scoreDropElement = document.getElementById('score-drop-value');
        const iterationsElement = document.getElementById('iterations-value');
        
        // Calcular queda de pontuação
        const baseScore = this.getBaseScore();
        const meanScore = result.mean_score || 0;
        const scoreDrop = baseScore - meanScore;
        const scoreDropPercent = (scoreDrop / baseScore * 100);
        
        // Atualizar elementos
        if (meanScoreElement) {
            meanScoreElement.textContent = this.formatMetric(meanScore);
        }
        
        if (worstScoreElement) {
            worstScoreElement.textContent = this.formatMetric(result.worst_score);
        }
        
        if (scoreDropElement) {
            scoreDropElement.textContent = `${this.formatMetric(scoreDropPercent)}%`;
            
            // Colorir baseado na queda
            if (scoreDropPercent > 20) {
                scoreDropElement.classList.add('text-danger');
                scoreDropElement.classList.remove('text-warning', 'text-success');
            } else if (scoreDropPercent > 10) {
                scoreDropElement.classList.add('text-warning');
                scoreDropElement.classList.remove('text-danger', 'text-success');
            } else {
                scoreDropElement.classList.add('text-success');
                scoreDropElement.classList.remove('text-danger', 'text-warning');
            }
        }
        
        if (iterationsElement) {
            const iterations = this.getIterationCount(level);
            iterationsElement.textContent = iterations;
        }
    },
    
    /**
     * Atualizar tabela de características para um nível
     * @param {Object} levelData - Dados para o nível de perturbação
     * @param {string} level - Nível de perturbação
     */
    updateFeatureTable: function(levelData, level) {
        const tableBody = document.getElementById('feature-perturbation-body');
        if (!tableBody) {
            console.error("Feature perturbation table body not found");
            return;
        }
        
        // Verificar se temos dados de característica
        if (!levelData.feature_results || Object.keys(levelData.feature_results).length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="empty-table-message">
                        Não foram encontrados dados de perturbação por característica para este nível.
                    </td>
                </tr>`;
            return;
        }
        
        // Obter pontuação base
        const baseScore = this.getBaseScore();
        
        // Converter dados de características para array para ordenação
        const featuresArray = Object.entries(levelData.feature_results).map(([name, result]) => {
            const meanScore = result.mean_score || 0;
            const worstScore = result.worst_score || 0;
            const impact = baseScore - meanScore;
            const dropPercent = (impact / baseScore * 100);
            
            return {
                name: name,
                meanScore: meanScore,
                worstScore: worstScore,
                impact: impact,
                dropPercent: dropPercent
            };
        });
        
        // Ordenar por impacto (maior para menor)
        featuresArray.sort((a, b) => b.impact - a.impact);
        
        // Gerar linhas HTML
        const rows = featuresArray.map(feature => {
            // Determinar classe de cor baseado na queda
            let dropClass = 'text-success';
            if (feature.dropPercent > 20) {
                dropClass = 'text-danger';
            } else if (feature.dropPercent > 10) {
                dropClass = 'text-warning';
            }
            
            return `
                <tr>
                    <td>${feature.name}</td>
                    <td class="numeric">${this.formatMetric(feature.impact)}</td>
                    <td class="numeric">${this.formatMetric(feature.worstScore)}</td>
                    <td class="numeric ${dropClass}">${this.formatMetric(feature.dropPercent)}%</td>
                </tr>`;
        }).join('');
        
        // Atualizar corpo da tabela
        tableBody.innerHTML = rows;
    },
    
    /**
     * Atualizar gráfico de distribuição de pontuações
     * @param {Object} levelData - Dados para o nível de perturbação
     * @param {string} level - Nível de perturbação
     */
    updateScoreDistributionChart: function(levelData, level) {
        const chartElement = document.getElementById('score-distribution-chart');
        if (!chartElement) {
            console.error("Score distribution chart element not found");
            return;
        }
        
        // Verificar se Plotly está disponível
        if (typeof Plotly === 'undefined') {
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: red;">
                    Plotly library not loaded. Chart cannot be displayed.
                </div>`;
            return;
        }
        
        // Obter pontuações de iteração
        const iterationScores = this.getIterationScores(level);
        if (!iterationScores || iterationScores.length === 0) {
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #666;">
                    Não foram encontradas pontuações de iteração para este nível.
                </div>`;
            return;
        }
        
        // Criar histograma
        const trace = {
            x: iterationScores,
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
        const baseScore = this.getBaseScore();
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
                tickformat: '.3f'
            },
            yaxis: {
                title: 'Frequência'
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
        
        // Renderizar gráfico
        Plotly.newPlot(chartElement, [trace], layout, {responsive: true});
    },
    
    /**
     * Inicializar gráfico radar de métricas
     */
    initMetricsRadarChart: function() {
        console.log("Initializing metrics radar chart");
        
        const chartElement = document.getElementById('metrics-radar-chart');
        if (!chartElement) {
            console.error("Metrics radar chart element not found");
            return;
        }
        
        // Verificar se Plotly está disponível
        if (typeof Plotly === 'undefined') {
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: red;">
                    Plotly library not loaded. Chart cannot be displayed.
                </div>`;
            return;
        }
        
        // Verificar se temos dados de modelo com métricas
        if (Object.keys(this.modelData).length === 0) {
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #666;">
                    Não foram encontrados dados de métricas de modelo.
                </div>`;
            return;
        }
        
        // Obter métricas comuns para todos os modelos
        const commonMetrics = this.getCommonMetrics();
        if (commonMetrics.length === 0) {
            chartElement.innerHTML = `
                <div style="padding: 20px; text-align: center; color: #666;">
                    Não foram encontradas métricas comuns entre os modelos.
                </div>`;
            return;
        }
        
        // Preparar dados para o gráfico radar
        const traces = [];
        
        // Converter dados de modelo para array
        const modelsArray = Object.entries(this.modelData).map(([key, model]) => ({
            id: key,
            name: model.name || key,
            type: model.type || 'Desconhecido',
            metrics: model.metrics || {}
        }));
        
        // Criar trace para cada modelo
        modelsArray.forEach((model, index) => {
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
                    range: [0, 1]
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
            }
        };
        
        // Renderizar gráfico
        Plotly.newPlot(chartElement, traces, layout, {responsive: true});
    },
    
    /**
     * Obter métricas comuns para todos os modelos
     * @returns {Array} Array de nomes de métricas comuns
     */
    getCommonMetrics: function() {
        // Conjunto de métricas padrão para verificar
        const standardMetrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];
        
        // Verificar quais métricas estão disponíveis em todos os modelos
        const availableMetrics = [];
        
        for (const metric of standardMetrics) {
            let allModelsHaveMetric = true;
            
            for (const modelData of Object.values(this.modelData)) {
                const metrics = modelData.metrics || {};
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
    },
    
    /**
     * Obter pontuação base do modelo
     * @returns {number} Pontuação base
     */
    getBaseScore: function() {
        if (window.reportData && window.reportData.base_score !== undefined) {
            return window.reportData.base_score;
        } else if (window.chartData && window.chartData.base_score !== undefined) {
            return window.chartData.base_score;
        } else if (window.chartData && window.chartData.perturbation_chart_data && 
                  window.chartData.perturbation_chart_data.baseScore !== undefined) {
            return window.chartData.perturbation_chart_data.baseScore;
        }
        
        // Verificar no primeiro modelo nas métricas
        if (Object.keys(this.modelData).length > 0) {
            const firstModel = Object.values(this.modelData)[0];
            const metrics = firstModel.metrics || {};
            if (metrics.accuracy !== undefined) {
                return metrics.accuracy;
            }
        }
        
        return 0;
    },
    
    /**
     * Obter pontuações de iteração para um nível
     * @param {string} level - Nível de perturbação
     * @returns {Array} Array de pontuações
     */
    getIterationScores: function(level) {
        // Verificar diferentes fontes para obter pontuações de iteração
        if (window.reportData && window.reportData.iterations_by_level && 
            window.reportData.iterations_by_level[level]) {
            return window.reportData.iterations_by_level[level];
        } else if (window.chartData && window.chartData.iterations_by_level && 
                 window.chartData.iterations_by_level[level]) {
            return window.chartData.iterations_by_level[level];
        }
        
        // Tentar extrair das execuções brutas
        if (this.perturbationData && this.perturbationData.by_level && 
            this.perturbationData.by_level[level] && this.perturbationData.by_level[level].runs) {
            const runs = this.perturbationData.by_level[level].runs;
            if (runs.all_features) {
                const allScores = [];
                runs.all_features.forEach(run => {
                    if (run.iterations && Array.isArray(run.iterations.scores)) {
                        allScores.push(...run.iterations.scores);
                    }
                });
                return allScores;
            }
        }
        
        return [];
    },
    
    /**
     * Obter contagem de iterações para um nível
     * @param {string} level - Nível de perturbação
     * @returns {number} Contagem de iterações
     */
    getIterationCount: function(level) {
        const scores = this.getIterationScores(level);
        if (scores && Array.isArray(scores)) {
            return scores.length;
        }
        
        if (window.reportData && window.reportData.n_iterations) {
            return window.reportData.n_iterations;
        }
        
        return 0;
    },
    
    /**
     * Lidar com mudança de seleção de modelo
     * @param {string} modelId - ID do modelo selecionado
     */
    handleModelSelection: function(modelId) {
        console.log("Model selection changed to:", modelId);
        
        // Destacar modelo selecionado na tabela
        const tableRows = document.querySelectorAll('#models-table-body tr');
        tableRows.forEach(row => {
            if (modelId === 'all' || row.getAttribute('data-model-id') === modelId) {
                row.classList.remove('inactive-row');
            } else {
                row.classList.add('inactive-row');
            }
        });
        
        // Atualizar gráfico radar para mostrar apenas o modelo selecionado
        if (modelId !== 'all') {
            this.updateRadarChartForModel(modelId);
        } else {
            // Mostrar todos os modelos no gráfico
            this.initMetricsRadarChart();
        }
    },
    
    /**
     * Atualizar gráfico radar para mostrar apenas um modelo
     * @param {string} modelId - ID do modelo
     */
    updateRadarChartForModel: function(modelId) {
        const chartElement = document.getElementById('metrics-radar-chart');
        if (!chartElement || typeof Plotly === 'undefined') {
            return;
        }
        
        // Encontrar modelo
        let selectedModel = null;
        for (const [id, model] of Object.entries(this.modelData)) {
            if (id === modelId) {
                selectedModel = {
                    id: id,
                    name: model.name || id,
                    type: model.type || 'Desconhecido',
                    metrics: model.metrics || {}
                };
                break;
            }
        }
        
        if (!selectedModel) {
            console.error("Selected model not found:", modelId);
            return;
        }
        
        // Obter métricas comuns
        const commonMetrics = this.getCommonMetrics();
        if (commonMetrics.length === 0) {
            return;
        }
        
        // Preparar dados para o gráfico
        const values = commonMetrics.map(metric => selectedModel.metrics[metric] || 0);
        values.push(values[0]); // Fechar o radar
        
        const labels = commonMetrics.map(metric => this.formatMetricName(metric));
        labels.push(labels[0]); // Fechar o radar
        
        const trace = {
            type: 'scatterpolar',
            r: values,
            theta: labels,
            fill: 'toself',
            name: selectedModel.name,
            opacity: 0.7
        };
        
        // Layout do gráfico
        const layout = {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 1]
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
            }
        };
        
        // Renderizar gráfico
        Plotly.newPlot(chartElement, [trace], layout, {responsive: true});
    },
    
    /**
     * Ordenar tabela de modelos pelo campo especificado
     * @param {string} sortBy - Campo para ordenar
     */
    sortModelsTable: function(sortBy) {
        const tableBody = document.getElementById('models-table-body');
        if (!tableBody) {
            return;
        }
        
        // Resetar indicadores de ordenação em todos os cabeçalhos
        document.querySelectorAll('#models-table th.sortable').forEach(header => {
            header.classList.remove('sort-asc', 'sort-desc');
        });
        
        // Obter elemento de cabeçalho para a coluna ordenada
        const header = document.querySelector(`#models-table th[data-sort="${sortBy}"]`);
        if (!header) {
            return;
        }
        
        // Determinar direção de ordenação
        let sortDirection = 'asc';
        if (header.classList.contains('sort-asc')) {
            sortDirection = 'desc';
        }
        
        // Definir indicador de ordenação na coluna ativa
        header.classList.add(`sort-${sortDirection}`);
        
        // Converter dados de modelo para array para ordenação
        const modelsArray = Object.entries(this.modelData).map(([key, model]) => ({
            id: key,
            name: model.name || key,
            type: model.type || 'Desconhecido',
            metrics: model.metrics || {}
        }));
        
        // Ordenar o array
        modelsArray.sort((a, b) => {
            let valueA, valueB;
            
            if (sortBy === 'name') {
                valueA = a.name;
                valueB = b.name;
                return sortDirection === 'asc' ? 
                    valueA.localeCompare(valueB) : 
                    valueB.localeCompare(valueA);
            } else if (sortBy === 'type') {
                valueA = a.type;
                valueB = b.type;
                return sortDirection === 'asc' ? 
                    valueA.localeCompare(valueB) : 
                    valueB.localeCompare(valueA);
            } else {
                // Assume que é uma métrica
                valueA = a.metrics[sortBy] || 0;
                valueB = b.metrics[sortBy] || 0;
                return sortDirection === 'asc' ? 
                    valueA - valueB : 
                    valueB - valueA;
            }
        });
        
        // Gerar HTML das linhas da tabela
        const rows = modelsArray.map(model => {
            const metrics = model.metrics || {};
            
            return `
                <tr data-model-id="${model.id}">
                    <td>${model.name}</td>
                    <td>${model.type}</td>
                    <td class="numeric">${this.formatMetric(metrics.accuracy)}</td>
                    <td class="numeric">${this.formatMetric(metrics.roc_auc)}</td>
                    <td class="numeric">${this.formatMetric(metrics.f1)}</td>
                    <td class="numeric">${this.formatMetric(metrics.precision)}</td>
                    <td class="numeric">${this.formatMetric(metrics.recall)}</td>
                </tr>`;
        }).join('');
        
        // Atualizar corpo da tabela
        tableBody.innerHTML = rows;
        
        // Reaplicar filtro de modelo se necessário
        const selectedModel = document.getElementById('model-selector').value;
        if (selectedModel !== 'all') {
            this.handleModelSelection(selectedModel);
        }
    },
    
    /**
     * Formatar valor de métrica para exibição
     * @param {number} value - Valor da métrica
     * @returns {string} - Valor formatado
     */
    formatMetric: function(value) {
        if (value === undefined || value === null) {
            return 'N/A';
        }
        return value.toFixed(4);
    },
    
    /**
     * Mostrar erro quando os dados não podem ser carregados
     * @param {string} message - Mensagem de erro
     */
    showDataError: function(message) {
        console.error("Data error:", message);
        
        // Encontrar elementos de container para mostrar o erro
        const chartContainer = document.getElementById('metrics-radar-chart');
        const tableBody = document.getElementById('models-table-body');
        
        // Mostrar erro no container do gráfico
        if (chartContainer) {
            chartContainer.innerHTML = `
                <div class="error-container">
                    <div class="error-icon">⚠️</div>
                    <h3 class="error-title">Erro ao Carregar Dados</h3>
                    <p class="error-message">${message}</p>
                </div>`;
        }
        
        // Mostrar erro na tabela
        if (tableBody) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="empty-table-message">
                        ${message}
                    </td>
                </tr>`;
        }
    },
    
    /**
     * Mostrar erro quando os dados de perturbação não podem ser carregados
     * @param {string} message - Mensagem de erro
     */
    showPerturbationDataError: function(message) {
        console.error("Perturbation data error:", message);
        
        // Encontrar container de resultados de perturbação
        const container = document.getElementById('perturbation-results-container');
        if (container) {
            container.innerHTML = `
                <div class="error-container">
                    <div class="error-icon">⚠️</div>
                    <h3 class="error-title">Erro ao Carregar Dados de Perturbação</h3>
                    <p class="error-message">${message}</p>
                    <p class="error-help">Verifique se os testes de robustez foram executados com coleta de dados detalhada.</p>
                </div>`;
        }
    }
};

// Inicializar controlador quando DOM for carregado
document.addEventListener('DOMContentLoaded', function() {
    // Pequeno atraso para garantir que outros scripts foram carregados
    setTimeout(function() {
        if (typeof DetailsController !== 'undefined' && 
            typeof DetailsController.init === 'function') {
            DetailsController.init();
        } else {
            console.error("DetailsController not available");
        }
    }, 100);
});