// Feature Importance Controller
window.FeatureImportanceController = {
    init: function() {
        console.log("Feature Importance section initialized");
        
        // Initialize sort selector
        this.initSortSelector();
        
        // Initialize chart selector if it exists
        this.initChartSelector('feature_importance_selector', '.feature-importance-section .chart-container');
        
        // Initialize charts
        this.initCharts();
        
        // Fill the feature importance table
        this.fillFeatureImportanceTable();
    },
    
    initSortSelector: function() {
        const sortSelector = document.getElementById('feature_sort_selector');
        if (!sortSelector) return;
        
        sortSelector.addEventListener('change', (e) => {
            const sortBy = e.target.value;
            this.sortFeatureTable(sortBy);
            
            // Re-initialize charts with new sort order
            this.initCharts(sortBy);
        });
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
    
    initCharts: function(sortBy = 'robustness') {
        console.log("Initializing feature importance charts");
        
        // Try to initialize all charts
        setTimeout(() => {
            if (typeof Plotly !== 'undefined') {
                this.initializeFeatureImportanceChart();
                this.initializeImportanceComparisonChart();
            } else {
                this.showChartError();
            }
        }, 500);
    },
    
    initializeFeatureImportanceChart: function() {
        console.log("Inicializando gráfico de feature importance");
        // Verificar se o elemento existe com o ID correto
        const chartElement = document.getElementById('feature-importance-chart');
        if (chartElement) {
            console.log("Elemento do gráfico encontrado:", chartElement.id);
            FeatureImportanceChartManager.initializeFeatureImportanceChart('feature-importance-chart');
        } else {
            console.error("Elemento do gráfico não encontrado: 'feature-importance-chart'");
            // Verificar todos os IDs dos elementos .chart-plot para debugging
            const allChartElements = document.querySelectorAll('.chart-plot');
            console.log("Elementos de gráfico disponíveis:", Array.from(allChartElements).map(el => el.id));
        }
    },
    
    initializeImportanceComparisonChart: function() {
        FeatureImportanceChartManager.initializeImportanceComparisonChart('importance-comparison-chart-plot');
    },
    
    showChartError: function() {
        const chartContainers = document.querySelectorAll('.chart-plot');
        chartContainers.forEach(container => {
            container.innerHTML = "<div style='padding: 20px; text-align: center; color: red;'>Plotly library not loaded. Charts cannot be displayed.</div>";
        });
    },
    
    fillFeatureImportanceTable: function() {
        console.log("Preenchendo tabela de feature importance");
        const tableBody = document.getElementById('feature-impact-data');
        if (!tableBody) {
            console.error("Elemento da tabela não encontrado: 'feature-impact-data'");
            return;
        }
        
        try {
            // Clear existing content
            tableBody.innerHTML = '';
            
            if (window.reportData) {
                const featureImportance = window.reportData.feature_importance || {};
                const modelFeatureImportance = window.reportData.model_feature_importance || {};
                
                // Convert to array and sort by robustness impact (absolute value)
                const featureArray = Object.keys(featureImportance).map(feature => ({
                    name: feature,
                    robustnessImportance: featureImportance[feature],
                    modelImportance: modelFeatureImportance[feature] || 0
                }));
                
                // Sort by absolute robustness importance value
                featureArray.sort((a, b) => Math.abs(b.robustnessImportance) - Math.abs(a.robustnessImportance));
                
                // Add rows to table
                featureArray.forEach(feature => {
                    const row = document.createElement('tr');
                    
                    // Feature name
                    const nameCell = document.createElement('td');
                    nameCell.textContent = feature.name;
                    row.appendChild(nameCell);
                    
                    // Robustness importance
                    const robustnessCell = document.createElement('td');
                    robustnessCell.textContent = feature.robustnessImportance.toFixed(4);
                    row.appendChild(robustnessCell);
                    
                    // Add progress bar for robustness importance
                    const robustnessBarCell = document.createElement('td');
                    const maxValue = Math.max(...featureArray.map(f => Math.abs(f.robustnessImportance)));
                    const percentage = Math.abs(feature.robustnessImportance) / maxValue * 100;
                    
                    // Color based on positive/negative impact
                    const barColor = feature.robustnessImportance >= 0 ? '#8884d8' : '#d88884';
                    
                    robustnessBarCell.innerHTML = `
                        <div class="progress-container">
                            <div class="progress-bar" style="width: ${percentage}%; background-color: ${barColor};"></div>
                        </div>
                    `;
                    row.appendChild(robustnessBarCell);
                    
                    // Model importance
                    const modelCell = document.createElement('td');
                    modelCell.textContent = feature.modelImportance.toFixed(4);
                    row.appendChild(modelCell);
                    
                    // Add progress bar for model importance
                    const modelBarCell = document.createElement('td');
                    const modelMaxValue = Math.max(...featureArray.map(f => f.modelImportance));
                    const modelPercentage = feature.modelImportance / modelMaxValue * 100;
                    
                    modelBarCell.innerHTML = `
                        <div class="progress-container">
                            <div class="progress-bar" style="width: ${modelPercentage}%; background-color: #82ca9d;"></div>
                        </div>
                    `;
                    row.appendChild(modelBarCell);
                    
                    tableBody.appendChild(row);
                });
            }
        } catch (error) {
            console.error("Error filling feature importance table:", error);
            this.showTableError(tableBody);
        }
    },
    
    sortFeatureTable: function(sortBy) {
        const tableBody = document.getElementById('feature-impact-data');
        if (!tableBody) return;
        
        try {
            // Get all rows
            const rows = Array.from(tableBody.querySelectorAll('tr'));
            
            // Sort rows based on the selected criteria
            rows.sort((a, b) => {
                const aValue = parseFloat(a.children[sortBy === 'robustness' ? 1 : 3].textContent);
                const bValue = parseFloat(b.children[sortBy === 'robustness' ? 1 : 3].textContent);
                
                if (sortBy === 'robustness') {
                    // Sort by absolute value for robustness
                    return Math.abs(bValue) - Math.abs(aValue);
                } else {
                    // Sort by value for model importance
                    return bValue - aValue;
                }
            });
            
            // Clear the table
            tableBody.innerHTML = '';
            
            // Add sorted rows
            rows.forEach(row => tableBody.appendChild(row));
            
        } catch (error) {
            console.error("Error sorting feature table:", error);
        }
    },
    
    showTableError: function(tableBody) {
        tableBody.innerHTML = '';
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 5;
        cell.textContent = 'Error loading feature importance data';
        cell.style.textAlign = 'center';
        cell.style.color = 'red';
        row.appendChild(cell);
        tableBody.appendChild(row);
    }
};