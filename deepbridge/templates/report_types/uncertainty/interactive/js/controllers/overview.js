// Overview Controller
const OverviewController = {
    init: function() {
        console.log("Overview section initialized");
        
        // Initialize chart selector
        this.initChartSelector('performance_charts_selector', '.performance-charts-container .chart-container');
        
        // Initialize model comparison selector if it exists
        this.initChartSelector('model_comparison_selector', '.model-comparison-section .chart-container');
        
        // Initialize tabs in overview
        this.initResultTabs();
        
        // Initialize charts immediately
        this.initCharts();
    },
    
    initChartSelector: function(selectorId, containerSelector) {
        const chartSelector = document.getElementById(selectorId);
        if (!chartSelector) return;
        
        const options = chartSelector.querySelectorAll('.chart-selector-option');
        options.forEach(option => {
            option.addEventListener('click', function() {
                // Remove active from all options
                options.forEach(opt => opt.classList.remove('active'));
                
                // Add active to clicked option
                this.classList.add('active');
                
                // Show corresponding chart
                const chartType = this.getAttribute('data-chart-type');
                const containers = document.querySelectorAll(containerSelector);
                
                containers.forEach(chart => {
                    chart.classList.remove('active');
                });
                
                const targetChart = document.querySelector(`${containerSelector}[data-chart-type="${chartType}"]`);
                if (targetChart) {
                    targetChart.classList.add('active');
                }
            });
        });
    },
    
    initResultTabs: function() {
        const resultTabsContainer = document.getElementById('result_tables_tabs');
        if (!resultTabsContainer) return;
        
        const tabButtons = resultTabsContainer.querySelectorAll('.tab');
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons
                tabButtons.forEach(btn => btn.classList.remove('active'));
                
                // Add active class to clicked button
                this.classList.add('active');
                
                // Hide all tab contents
                const tabContents = document.querySelectorAll('.results-tables-section .tab-content');
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Show target tab content
                const targetId = this.getAttribute('data-tab');
                const targetContent = document.getElementById(targetId);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
            });
        });
    },
    
    initCharts: function() {
        console.log("Initializing overview charts");
        
        // Try to initialize all charts
        setTimeout(() => {
            if (typeof Plotly !== 'undefined') {
                this.initializePerturbationChart();
                this.initializeWorstScoreChart();
                this.initializeMeanScoreChart();
                
                // Initialize model comparison charts if available
                if (window.reportConfig && window.reportConfig.hasAlternativeModels) {
                    this.initializeModelComparisonChart();
                    this.initializeModelLevelDetailsChart();
                    this.populateModelComparisonTable();
                }
                
                // Fill the perturbation tables
                this.fillPerturbationTables();
            } else {
                this.showChartError();
            }
        }, 500);
    },
    
    showChartError: function() {
        const chartContainers = document.querySelectorAll('.chart-plot');
        chartContainers.forEach(container => {
            container.innerHTML = "<div style='padding: 20px; text-align: center; color: red;'>Plotly library not loaded. Charts cannot be displayed.</div>";
        });
    },
    
    fillPerturbationTables: function() {
        // Fill the raw perturbation table
        try {
            const rawTableBody = document.getElementById('raw-perturbation-data');
            if (!rawTableBody) return;
            
            // Clear existing content
            rawTableBody.innerHTML = '';
            
            if (window.reportData && window.reportData.raw && window.reportData.raw.by_level) {
                const rawData = window.reportData.raw.by_level;
                const levels = Object.keys(rawData).sort((a, b) => parseFloat(a) - parseFloat(b));
                
                let baseScore = window.reportData.base_score || 0;
                
                levels.forEach(level => {
                    if (rawData[level] && rawData[level].overall_result) {
                        const result = rawData[level].overall_result;
                        let levelData;
                        
                        if (result.all_features) {
                            levelData = result.all_features;
                        } else if (result.feature_subset) {
                            levelData = result.feature_subset;
                        } else {
                            return; // Skip if no data
                        }
                        
                        const row = document.createElement('tr');
                        
                        // Level
                        const levelCell = document.createElement('td');
                        levelCell.textContent = level;
                        row.appendChild(levelCell);
                        
                        // Base score
                        const baseScoreCell = document.createElement('td');
                        baseScoreCell.textContent = baseScore.toFixed(3);
                        row.appendChild(baseScoreCell);
                        
                        // Perturbed score
                        const perturbedScoreCell = document.createElement('td');
                        perturbedScoreCell.textContent = levelData.mean_score ? levelData.mean_score.toFixed(3) : 'N/A';
                        row.appendChild(perturbedScoreCell);
                        
                        // Impact
                        const impactCell = document.createElement('td');
                        impactCell.textContent = levelData.impact ? (levelData.impact * 100).toFixed(2) + '%' : 'N/A';
                        row.appendChild(impactCell);
                        
                        // Worst score
                        const worstScoreCell = document.createElement('td');
                        worstScoreCell.textContent = levelData.worst_score ? levelData.worst_score.toFixed(3) : 'N/A';
                        row.appendChild(worstScoreCell);
                        
                        rawTableBody.appendChild(row);
                    }
                });
            }
        } catch (error) {
            console.error("Error filling perturbation tables:", error);
        }
    },
    
    // Chart initialization methods would be here (moved to charts/overview.js)
    initializePerturbationChart: function() {
        ChartManager.initializePerturbationChart('perturbation-chart-plot');
    },
    
    initializeWorstScoreChart: function() {
        ChartManager.initializeWorstScoreChart('worst-score-chart-plot');
    },
    
    initializeMeanScoreChart: function() {
        ChartManager.initializeMeanScoreChart('mean-score-chart-plot');
    },
    
    initializeModelComparisonChart: function() {
        ChartManager.initializeModelComparisonChart('model-comparison-chart-plot');
    },
    
    initializeModelLevelDetailsChart: function() {
        ChartManager.initializeModelLevelDetailsChart('model-level-details-chart-plot');
    },
    
    populateModelComparisonTable: function() {
        const tableBody = document.getElementById('model-comparison-table')?.querySelector('tbody');
        if (!tableBody) return;
        
        try {
            // Clear existing content
            tableBody.innerHTML = '';
            
            // Check if we have alternative models data
            if (!window.reportData || !window.reportConfig || !window.reportConfig.hasAlternativeModels) {
                this.showNoModelComparisonDataMessage(tableBody);
                return;
            }
            
            // Get model data
            const models = this.extractModelComparisonData();
            
            if (models.length === 0) {
                this.showNoModelComparisonDataMessage(tableBody);
                return;
            }
            
            // Sort models by robustness score
            models.sort((a, b) => b.robustnessScore - a.robustnessScore);
            
            // Add a row for each model
            models.forEach(model => {
                const row = document.createElement('tr');
                
                // Model name
                const nameCell = document.createElement('td');
                nameCell.textContent = model.name;
                
                // Base score
                const baseScoreCell = document.createElement('td');
                baseScoreCell.textContent = model.baseScore.toFixed(3);
                
                // Robustness score
                const robustnessScoreCell = document.createElement('td');
                // Usar o valor real do robustness_score
                const score = typeof model.robustnessScore === 'number' ? model.robustnessScore : 0;
                robustnessScoreCell.textContent = score.toFixed(3);
                
                // Raw impact
                const rawImpactCell = document.createElement('td');
                rawImpactCell.textContent = (model.rawImpact * 100).toFixed(2) + '%';
                
                // Quantile impact
                const quantileImpactCell = document.createElement('td');
                quantileImpactCell.textContent = (model.quantileImpact * 100).toFixed(2) + '%';
                
                row.appendChild(nameCell);
                row.appendChild(baseScoreCell);
                row.appendChild(robustnessScoreCell);
                row.appendChild(rawImpactCell);
                row.appendChild(quantileImpactCell);
                
                tableBody.appendChild(row);
            });
            
        } catch (error) {
            console.error("Error populating model comparison table:", error);
            this.showErrorMessage(tableBody);
        }
    },
    
    showNoModelComparisonDataMessage: function(tableBody) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 5;
        cell.innerHTML = `
            <div class="no-data-message">
                <p><strong>No model comparison data available</strong></p>
                <p>Run robustness test with compare() method to see model comparison data.</p>
            </div>
        `;
        cell.style.textAlign = 'center';
        row.appendChild(cell);
        tableBody.appendChild(row);
    },
    
    showErrorMessage: function(tableBody) {
        tableBody.innerHTML = '';
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 5;
        cell.textContent = 'Error loading model comparison data';
        cell.style.textAlign = 'center';
        cell.style.color = 'red';
        row.appendChild(cell);
        tableBody.appendChild(row);
    },
    
    extractModelComparisonData: function() {
        const models = [];
        
        // Add primary model
        let primaryModelName = 'Primary Model';
        let primaryModelData = null;
        
        if (window.reportData.primary_model) {
            primaryModelData = window.reportData.primary_model;
            primaryModelName = primaryModelData.model_name || window.reportConfig.modelName || 'Primary Model';
        } else {
            // Primary model data is at the top level
            primaryModelData = window.reportData;
            primaryModelName = window.reportConfig.modelName || 'Primary Model';
        }
        
        if (primaryModelData) {
            // CORREÇÃO: Garantir que usamos os valores corretos do robustness_score
            // Se o valor estiver presente diretamente em primaryModelData, use-o
            let robustnessScore = 0;
            if (typeof primaryModelData.robustness_score === 'number') {
                robustnessScore = primaryModelData.robustness_score;
                console.log("Usando robustness_score do modelo primário:", robustnessScore);
            } else if (window.reportConfig && typeof window.reportConfig.robustnessScore === 'number') {
                robustnessScore = window.reportConfig.robustnessScore;
                console.log("Usando robustnessScore do reportConfig:", robustnessScore);
            }
            
            models.push({
                name: primaryModelName,
                baseScore: primaryModelData.base_score || 0,
                robustnessScore: robustnessScore,
                rawImpact: primaryModelData.avg_raw_impact || window.reportConfig.rawImpact || 0,
                quantileImpact: primaryModelData.avg_quantile_impact || window.reportConfig.quantileImpact || 0
            });
        }
        
        // Add alternative models
        if (window.reportData.alternative_models) {
            Object.entries(window.reportData.alternative_models).forEach(([name, data]) => {
                // CORREÇÃO: Garantir que usamos os valores corretos do robustness_score para modelos alternativos
                const altScore = typeof data.robustness_score === 'number' ? data.robustness_score : 0;
                console.log(`Modelo alternativo ${name} robustness_score:`, altScore);
                
                models.push({
                    name: name,
                    baseScore: data.base_score || 0,
                    robustnessScore: altScore,
                    rawImpact: data.avg_raw_impact || 0,
                    quantileImpact: data.avg_quantile_impact || 0
                });
            });
        }
        
        return models;
    }
};