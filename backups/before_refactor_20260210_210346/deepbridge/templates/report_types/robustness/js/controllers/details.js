/**
 * Overview Controller
 * Handles data and UI interactions for the model overview page
 */
window.OverviewController = {
    /**
     * Initialize the controller
     */
    init: function() {
        console.log("OverviewController initialized");
        
        // Load model data
        this.loadModelData();
        
        // Initialize the model selector
        this.initializeModelSelector();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize charts
        this.initializeCharts();
        
        // Update the dataset info
        this.updateDatasetInfo();
        
        // Fill the models table
        this.fillModelsTable();
        
        console.log("Overview page initialization complete");
    },
    
    /**
     * Load model data from global variables
     */
    loadModelData: function() {
        console.log("Loading model data");
        this.modelData = {};
        this.configData = {};
        
        // Try initial_results data first (from transformer)
        if (window.reportData && window.reportData.initial_results) {
            console.log("Found initial_results data:", Object.keys(window.reportData.initial_results).join(', '));
            if (window.reportData.initial_results.models) {
                this.modelData = window.reportData.initial_results.models;
                this.configData = window.reportData.initial_results.config || {};
                console.log("Using data from initial_results, found models:", Object.keys(this.modelData).length);
            }
        } 
        // Try chart_data.initial_results
        else if (window.chartData && window.chartData.initial_results) {
            console.log("Found initial_results in chartData");
            if (window.chartData.initial_results.models) {
                this.modelData = window.chartData.initial_results.models;
                this.configData = window.chartData.initial_results.config || {};
                console.log("Using data from chartData.initial_results, found models:", Object.keys(this.modelData).length);
            }
        }
        // Try radar_chart_data
        else if (window.chartData && window.chartData.radar_chart_data) {
            console.log("Found radar_chart_data");
            if (window.chartData.radar_chart_data.models) {
                this.modelData = window.chartData.radar_chart_data.models;
                console.log("Using data from radar_chart_data, found models:", Object.keys(this.modelData).length);
            }
        }
        // Try generic model data
        else if (window.reportData) {
            this.modelData = window.reportData.models || {};
            this.configData = window.reportData.config || {};
            console.log("Using data from reportData");
        } else if (window.chartData) {
            this.modelData = window.chartData.models || {};
            this.configData = window.chartData.config || {};
            console.log("Using data from chartData");
        } else if (window.config) {
            this.modelData = window.config.models || {};
            this.configData = window.config;
            console.log("Using data from config");
        } else {
            console.warn("No model data found in any data source");
        }
        
        // Debug the data that was found
        console.log("Model data keys:", Object.keys(this.modelData));
        if (Object.keys(this.modelData).length > 0) {
            const firstModel = Object.values(this.modelData)[0];
            console.log("First model data:", JSON.stringify(firstModel).substring(0, 200) + "...");
        }
        
        // Check if we have model data
        if (Object.keys(this.modelData).length === 0) {
            console.warn("Empty model data");
            this.showDataError("Não foi possível carregar dados de modelos.");
        } else {
            console.log("Loaded data for", Object.keys(this.modelData).length, "models");
        }
    },
    
    /**
     * Initialize the model selector dropdown
     */
    initializeModelSelector: function() {
        console.log("Initializing model selector");
        const selector = document.getElementById('model-selector');
        if (!selector) {
            console.error("Model selector element not found");
            return;
        }
        
        // Keep the 'all' option as first choice
        let options = ['<option value="all">Todos os Modelos</option>'];
        
        // Add options for each model
        Object.entries(this.modelData).forEach(([key, model]) => {
            options.push(`<option value="${key}">${model.name} (${model.type})</option>`);
        });
        
        // Update selector options
        selector.innerHTML = options.join('');
        
        console.log("Model selector initialized with", Object.keys(this.modelData).length, "models");
    },
    
    /**
     * Set up event listeners for UI interactions
     */
    setupEventListeners: function() {
        console.log("Setting up event listeners");
        
        // Model selector change event
        const selector = document.getElementById('model-selector');
        if (selector) {
            selector.addEventListener('change', (e) => {
                const modelId = e.target.value;
                this.handleModelSelection(modelId);
            });
        }
        
        // Table header sort event
        const headers = document.querySelectorAll('.data-table th.sortable');
        if (headers.length > 0) {
            headers.forEach(header => {
                header.addEventListener('click', (e) => {
                    const sortBy = e.currentTarget.getAttribute('data-sort');
                    this.sortModelsTable(sortBy);
                });
            });
        }
        
        // Tab navigation
        const tabLinks = document.querySelectorAll('.main-nav a');
        if (tabLinks.length > 0) {
            tabLinks.forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const tabId = e.currentTarget.getAttribute('data-tab');
                    this.switchTab(tabId);
                });
            });
        }
        
        console.log("Event listeners setup complete");
    },
    
    /**
     * Initialize charts on the overview page
     */
    initializeCharts: function() {
        console.log("Initializing overview charts");
        
        // Use the chart manager to initialize charts
        if (typeof window.OverviewChartsManager !== 'undefined' && 
            typeof window.OverviewChartsManager.initializeOverviewCharts === 'function') {
            window.OverviewChartsManager.initializeOverviewCharts();
        } else {
            console.error("OverviewChartsManager not available");
        }
    },
    
    /**
     * Update dataset info in the UI
     */
    updateDatasetInfo: function() {
        console.log("Updating dataset info");
        
        // Get dataset info elements
        const samplesElement = document.getElementById('n-samples');
        const featuresElement = document.getElementById('n-features');
        const testSizeElement = document.getElementById('test-size');
        
        // Get test info elements
        const testsListElement = document.getElementById('tests-list');
        const verboseElement = document.getElementById('verbose-status');
        
        // Check if we have dataset info
        if (this.configData && this.configData.dataset_info) {
            const datasetInfo = this.configData.dataset_info;
            
            // Update dataset info
            if (samplesElement) {
                samplesElement.textContent = datasetInfo.n_samples || 'N/A';
            }
            
            if (featuresElement) {
                featuresElement.textContent = datasetInfo.n_features || 'N/A';
            }
            
            if (testSizeElement && datasetInfo.test_size !== undefined) {
                testSizeElement.textContent = (datasetInfo.test_size * 100) + '%';
            }
        } else {
            console.warn("No dataset info available");
            
            // Set placeholders
            if (samplesElement) samplesElement.textContent = 'N/A';
            if (featuresElement) featuresElement.textContent = 'N/A';
            if (testSizeElement) testSizeElement.textContent = 'N/A';
        }
        
        // Check if we have test config info
        if (this.configData && this.configData.tests) {
            // Update test info
            if (testsListElement) {
                testsListElement.textContent = this.configData.tests.join(', ');
            }
            
            if (verboseElement) {
                verboseElement.textContent = this.configData.verbose ? 'Sim' : 'Não';
            }
        } else {
            console.warn("No test config info available");
            
            // Set placeholders
            if (testsListElement) testsListElement.textContent = 'N/A';
            if (verboseElement) verboseElement.textContent = 'N/A';
        }
        
        console.log("Dataset info updated");
    },
    
    /**
     * Fill the models comparison table
     */
    fillModelsTable: function() {
        console.log("Filling models table");
        
        const tableBody = document.getElementById('models-table-body');
        if (!tableBody) {
            console.error("Models table body element not found");
            return;
        }
        
        // Check if we have model data
        if (Object.keys(this.modelData).length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="empty-table-message">
                        Não foram encontrados dados de modelos para exibir.
                    </td>
                </tr>`;
            return;
        }
        
        // Convert model data to array for sorting
        const modelsArray = Object.entries(this.modelData).map(([key, model]) => ({
            id: key,
            name: model.name || key,
            type: model.type || 'Desconhecido',
            metrics: model.metrics || {}
        }));
        
        // Sort by model name by default
        modelsArray.sort((a, b) => a.name.localeCompare(b.name));
        
        // Generate table rows HTML
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
        
        // Update table body
        tableBody.innerHTML = rows;
        
        console.log("Models table filled with", modelsArray.length, "models");
    },
    
    /**
     * Format metric value for display
     * @param {number} value - Metric value
     * @returns {string} - Formatted value
     */
    formatMetric: function(value) {
        if (value === undefined || value === null) {
            return 'N/A';
        }
        return value.toFixed(4);
    },
    
    /**
     * Handle model selection change
     * @param {string} modelId - Selected model ID
     */
    handleModelSelection: function(modelId) {
        console.log("Model selection changed to:", modelId);
        
        // Highlight selected model in the table
        const tableRows = document.querySelectorAll('#models-table-body tr');
        tableRows.forEach(row => {
            if (modelId === 'all' || row.getAttribute('data-model-id') === modelId) {
                row.classList.remove('inactive-row');
            } else {
                row.classList.add('inactive-row');
            }
        });
        
        // You could refresh charts here based on selection
        // For example:
        // this.refreshMetricsChart(modelId);
    },
    
    /**
     * Sort the models table by the specified column
     * @param {string} sortBy - Column to sort by
     */
    sortModelsTable: function(sortBy) {
        console.log("Sorting table by:", sortBy);
        
        // Reset sort indicators on all header cells
        document.querySelectorAll('#models-table th.sortable').forEach(header => {
            header.classList.remove('sort-asc', 'sort-desc');
        });
        
        // Get the header element for the sorted column
        const header = document.querySelector(`#models-table th[data-sort="${sortBy}"]`);
        
        // Determine sort direction
        let sortDirection = 'asc';
        if (header.classList.contains('sort-asc')) {
            sortDirection = 'desc';
        }
        
        // Set sort indicator on active column
        header.classList.add(`sort-${sortDirection}`);
        
        // Convert model data to array for sorting
        const modelsArray = Object.entries(this.modelData).map(([key, model]) => ({
            id: key,
            name: model.name || key,
            type: model.type || 'Desconhecido',
            metrics: model.metrics || {}
        }));
        
        // Sort the array
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
                // Assume it's a metric
                valueA = a.metrics[sortBy] || 0;
                valueB = b.metrics[sortBy] || 0;
                return sortDirection === 'asc' ? 
                    valueA - valueB : 
                    valueB - valueA;
            }
        });
        
        // Generate table rows HTML
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
        
        // Update table body
        const tableBody = document.getElementById('models-table-body');
        if (tableBody) {
            tableBody.innerHTML = rows;
        }
        
        // Reapply model filter if needed
        const selectedModel = document.getElementById('model-selector').value;
        if (selectedModel !== 'all') {
            this.handleModelSelection(selectedModel);
        }
    },
    
    /**
     * Switch between tabs
     * @param {string} tabId - ID of the tab to switch to
     */
    switchTab: function(tabId) {
        console.log("Switching to tab:", tabId);
        
        // Update tab links
        document.querySelectorAll('.main-nav li').forEach(tab => {
            if (tab.querySelector(`a[data-tab="${tabId}"]`)) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
        
        // Update tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            if (content.id === tabId) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
        
        // Dispatch custom event for tab change
        const event = new CustomEvent('tabchange', {
            detail: { tabId: tabId }
        });
        document.dispatchEvent(event);
    },
    
    /**
     * Show error message when data cannot be loaded
     * @param {string} message - Error message
     */
    showDataError: function(message) {
        console.error("Data error:", message);
        
        // Find container elements to show the error
        const chartContainer = document.getElementById('metrics-radar-chart');
        const tableBody = document.getElementById('models-table-body');
        
        // Show error in chart container
        if (chartContainer) {
            chartContainer.innerHTML = `
                <div class="error-container">
                    <div class="error-icon">⚠️</div>
                    <h3 class="error-title">Erro ao Carregar Dados</h3>
                    <p class="error-message">${message}</p>
                </div>`;
        }
        
        // Show error in table
        if (tableBody) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="empty-table-message">
                        ${message}
                    </td>
                </tr>`;
        }
    }
};

// Initialize controller when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Small delay to ensure other scripts are loaded
    setTimeout(function() {
        if (typeof OverviewController !== 'undefined' && 
            typeof OverviewController.init === 'function') {
            OverviewController.init();
        } else {
            console.error("OverviewController not available");
        }
    }, 100);
});